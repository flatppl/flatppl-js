'use strict';

// AST → FlatPIR-JSON lowering.
//
// =====================================================================
// Why we lower
// =====================================================================
//
// FlatPIR is the canonical IR for FlatPPL (see flatppl-design/docs/11-flatpir.md).
// Whereas the surface FlatPPL AST captures syntax (operators, indexing,
// field access, comments, source positions), FlatPIR captures *semantics*
// in a uniform shape: every expression is either a literal, a reference,
// or a call. That uniformity is what later passes — the sampler, the
// codegen-to-tfjs-or-Rust we may someday do, content-addressed cache
// signatures — want.
//
// We use a JSON-shaped representation of FlatPIR (not the canonical
// S-expression text). The JSON form mirrors FlatPIR's structural rules
// directly; an S-expression printer on top is a small follow-on if
// cross-tool interop ever needs it.
//
// =====================================================================
// JSON shapes
// =====================================================================
//
// Literals:
//   { kind: 'lit',   value: <number|string|boolean>,                   loc }
//
// Built-in symbols (constants `pi`, `inf`, `im`; sets `reals`, `posreals`,
// …; the special `all` slice marker):
//   { kind: 'const', name: <string>,                                   loc }
//
// References:
//   { kind: 'ref',   ns:   'self' | '%local' | <module-alias>,
//                    name: <string>,                                   loc }
//
// Hole (bare `_` inside `fn(...)`):
//   { kind: 'hole',                                                    loc }
//
// Call (built-in or user-defined). Built-ins use `op`; user-defined calls
// use `target`. Positional args go in `args`; keyword args into `kwargs`
// (object — unordered per FlatPIR `%kwarg` semantics). Both are optional
// — only present when non-empty.
//   { kind: 'call', op:     <builtin-name>,
//                   args?:  [<expr>, …],
//                   kwargs?: { <name>: <expr>, … },
//                   loc, meta? }
//   { kind: 'call', target: { ns, name },           // user-defined head
//                   args?, kwargs?, loc, meta? }
//
// Calls with ordered named entries (record, joint, jointchain, cartprod,
// table) use a `fields: [{name, value}, …]` array because their order is
// part of the structure (mirrors FlatPIR `%field`). They may carry
// positional `args` too.
//   { kind: 'call', op: 'record', fields: [{name, value}, …], loc }
//
// Reified callables (`functionof`, `kernelof`, `fn`) introduce a parameter
// scope. We surface `params` (a list of names) and the `body` expression
// directly; references inside the body to those names lower as
// `{ ns: '%local', name }`. The surface kwarg names are also preserved
// in `paramKwargs` for callsite-keyword matching at higher layers.
//   { kind: 'call', op: 'functionof',
//                   params:      [<name>, …],
//                   paramKwargs: [<surface-kwarg-name>, …],   // parallel to params
//                   body:        <expr>,
//                   loc }
//
// `fn(<expr>)` is a special case: bare `_` holes inside the body are the
// implicit parameters; we don't extract a params list.
//
// Module loads (`load_module`, `standard_module`) carry their substitution
// kwargs as `assigns` (object), distinct from `kwargs` because they're
// resolved at load time, not call time.
//   { kind: 'call', op: 'load_module', args: ["..."], assigns: {…}, loc }
//
// =====================================================================
// Operator desugaring
// =====================================================================
//
// FlatPIR canonical form has no operators — they all lower to function
// calls. We do the same lowering during AST→IR rather than at consumer
// time, so every IR consumer (sampler, codegen, signature-hash) sees one
// uniform shape.
//
//   a + b       →  { kind: 'call', op: 'add', args: [a, b] }
//   -x          →  { kind: 'call', op: 'neg', args: [x] }
//   a[i, j]     →  { kind: 'call', op: 'get', args: [a, i, j] }
//   a.field     →  { kind: 'call', op: 'get_field', args: [a, "field"] }
//   a in S      →  { kind: 'call', op: 'in',  args: [a, S] }
//
// =====================================================================
// Scope tracking
// =====================================================================
//
// Reified callables introduce an inner `%local` scope. We pass `ctx`
// through every recursive `lowerExpr` call; `ctx.localScope` is a Set of
// names visible as `%local`. Inside `functionof`/`kernelof` bodies we
// extend the scope with the new params before recursing.
//
// For ordinary identifiers, the resolution rule is:
//   1. If the name is in `ctx.localScope`         → `{ ref: '%local', name }`
//   2. Else if it's a known builtin constant/set  → `{ const: name }`
//   3. Else if it's `true` or `false`             → `{ lit: <bool> }`
//   4. Otherwise                                  → `{ ref: 'self', name }`
//
// (Step 4 is correct even for "unknown" names — the analyzer flags
// undefined-variable diagnostics separately. The IR pass doesn't perform
// name resolution beyond local-vs-self.)

const builtins = require('./builtins');

const { CONSTANTS, SETS, BOOL_LITERALS, ALL_KNOWN } = builtins;

// ---------------------------------------------------------------------
// Public API

/**
 * Lower a single AST expression into FlatPIR-JSON. Pure: returns a fresh
 * IR tree, doesn't touch the input AST.
 *
 * @param {object} node - an AST expression node
 * @param {object} [ctx] - lowering context
 * @param {Set<string>} [ctx.localScope] - names currently bound as `%local`
 * @returns {object} IR-JSON expression
 */
function lowerExpr(node, ctx) {
  ctx = ctx || { localScope: null };
  return _lowerExpr(node, ctx);
}

/**
 * Lower an AssignStatement's RHS. Convenience wrapper around `lowerExpr`.
 */
function lowerBinding(stmt, ctx) {
  if (!stmt || stmt.type !== 'AssignStatement') {
    throw new Error(`lower: expected AssignStatement, got ${stmt?.type}`);
  }
  return lowerExpr(stmt.value, ctx);
}

// ---------------------------------------------------------------------
// Operator → builtin-name mapping

const BIN_OP_MAP = {
  '+':  'add',
  '-':  'sub',
  '*':  'mul',
  '/':  'div',
  '<':  'lt',
  '<=': 'le',
  '>':  'gt',
  '>=': 'ge',
  '==': 'eq',
  '!=': 'ne',
  'in': 'in',
};

const UN_OP_MAP = {
  '-': 'neg',
  '+': 'pos',
};

// ---------------------------------------------------------------------
// Built-in calls that carry ordered named entries (`%field` in FlatPIR).
// These produce IR with a `fields` array preserving source order.

const FIELD_FORMS = new Set([
  'record',
  'joint',
  'jointchain',
  'cartprod',
  'table',
]);

// Reified callables — introduce `%local` parameter scope.
const REIFICATION_FORMS = new Set([
  'functionof',
  'kernelof',
  // `fn` is a reification but doesn't have keyword params (uses bare holes
  // instead) — handled separately.
]);

// Module-load forms — kwargs are substitutions (`%assign`), not call kwargs.
const MODULE_LOAD_FORMS = new Set([
  'load_module',
  'standard_module',
]);

// ---------------------------------------------------------------------
// Core dispatch

function _lowerExpr(node, ctx) {
  switch (node.type) {
    case 'NumberLiteral':
      return { kind: 'lit', value: node.value, loc: node.loc };

    case 'StringLiteral':
      return { kind: 'lit', value: node.value, loc: node.loc };

    case 'BoolLiteral':
      return { kind: 'lit', value: node.value, loc: node.loc };

    case 'ConstantRef':
      // pi, inf, im — bare-symbol builtins per FlatPIR.
      return { kind: 'const', name: node.name, loc: node.loc };

    case 'SetRef':
      // reals, posreals, integers, booleans, … — also bare-symbol builtins.
      return { kind: 'const', name: node.name, loc: node.loc };

    case 'SliceAll':
      // The `:` slice marker. Per spec, lowers to bare `all`.
      return { kind: 'const', name: 'all', loc: node.loc };

    case 'Hole':
      return { kind: 'hole', loc: node.loc };

    case 'Placeholder':
      // Inside a reified scope, surface `_x_` references the param of the
      // same name. The analyzer enforces that placeholders only appear
      // inside functionof/kernelof, so we can assume the local scope is
      // populated. Param names preserve the trailing-underscore convention
      // per FlatPIR §"Function parameter lists".
      return {
        kind: 'ref',
        ns:   '%local',
        name: '_' + node.name + '_',
        loc:  node.loc,
      };

    case 'Identifier':
      return _lowerIdentifier(node, ctx);

    case 'ArrayLiteral':
      // [1, 2, 3] — lowers to a `(vector ...)` call per FlatPIR composite-
      // literal convention.
      return {
        kind: 'call',
        op:   'vector',
        args: node.elements.map(e => _lowerExpr(e, ctx)),
        loc:  node.loc,
      };

    case 'TupleLiteral':
      // (a, b) — lowers to a `(tuple ...)` call.
      return {
        kind: 'call',
        op:   'tuple',
        args: node.elements.map(e => _lowerExpr(e, ctx)),
        loc:  node.loc,
      };

    case 'BinaryExpr':
      return _lowerBinaryExpr(node, ctx);

    case 'UnaryExpr':
      return _lowerUnaryExpr(node, ctx);

    case 'IndexExpr':
      // a[i, j, …] → (get a i j …)
      return {
        kind: 'call',
        op:   'get',
        args: [
          _lowerExpr(node.object, ctx),
          ...node.indices.map(i => _lowerExpr(i, ctx)),
        ],
        loc: node.loc,
      };

    case 'FieldAccess':
      // a.field → (get_field a "field")
      return {
        kind: 'call',
        op:   'get_field',
        args: [
          _lowerExpr(node.object, ctx),
          { kind: 'lit', value: node.field, loc: node.loc },
        ],
        loc: node.loc,
      };

    case 'CallExpr':
      return _lowerCallExpr(node, ctx);

    default:
      throw new Error(`lower: unsupported AST node type '${node.type}'`);
  }
}

function _lowerIdentifier(node, ctx) {
  const { name, loc } = node;

  // 1. %local scope wins (innermost reified scope's params).
  if (ctx.localScope && ctx.localScope.has(name)) {
    return { kind: 'ref', ns: '%local', name, loc };
  }

  // 2. Boolean literals (`true` / `false` are parsed as Identifiers in
  // some FlatPPL fronts, by name — check defensively).
  if (BOOL_LITERALS.has(name)) {
    return { kind: 'lit', value: name === 'true', loc };
  }

  // 3. Built-in constants (pi, inf, im) and sets (reals, posreals, …).
  // Both lower to bare-symbol `const` per FlatPIR.
  if (CONSTANTS.has(name) || SETS.has(name)) {
    return { kind: 'const', name, loc };
  }

  // 4. Default: a self-module reference. The analyzer separately flags
  // undefined-variable diagnostics; the IR pass doesn't second-guess.
  return { kind: 'ref', ns: 'self', name, loc };
}

function _lowerBinaryExpr(node, ctx) {
  const op = BIN_OP_MAP[node.op];
  if (!op) {
    throw new Error(`lower: unknown binary operator '${node.op}'`);
  }
  return {
    kind: 'call',
    op,
    args: [_lowerExpr(node.left, ctx), _lowerExpr(node.right, ctx)],
    loc:  node.loc,
  };
}

function _lowerUnaryExpr(node, ctx) {
  const op = UN_OP_MAP[node.op];
  if (!op) {
    throw new Error(`lower: unknown unary operator '${node.op}'`);
  }
  return {
    kind: 'call',
    op,
    args: [_lowerExpr(node.operand, ctx)],
    loc:  node.loc,
  };
}

function _lowerCallExpr(node, ctx) {
  if (!node.callee || node.callee.type !== 'Identifier') {
    // Higher-order or computed callees aren't part of the FlatPPL surface
    // grammar today. If we ever add them, they'd lower with a
    // computed-target form. For now, refuse loudly.
    throw new Error(`lower: unsupported callee type '${node.callee?.type}'`);
  }
  const calleeName = node.callee.name;

  // Special-case dispatchers, in priority order:

  if (REIFICATION_FORMS.has(calleeName)) {
    return _lowerReification(calleeName, node, ctx);
  }
  if (calleeName === 'fn') {
    return _lowerFn(node, ctx);
  }
  if (FIELD_FORMS.has(calleeName)) {
    return _lowerFieldsForm(calleeName, node, ctx);
  }
  if (MODULE_LOAD_FORMS.has(calleeName)) {
    return _lowerModuleLoad(calleeName, node, ctx);
  }

  // General call: built-in (we know its name) vs user-defined (we don't).
  // The analyzer's collected `definedNames` set could refine this, but
  // for IR purposes the head-shape distinction (op vs target) is enough.
  // Built-in names are listed in `builtins.ALL_KNOWN`.

  const args = [];
  const kwargs = {};
  let hasKwargs = false;
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      kwargs[arg.name] = _lowerExpr(arg.value, ctx);
      hasKwargs = true;
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }

  const out = { kind: 'call', loc: node.loc };
  if (ALL_KNOWN.has(calleeName)) {
    // Built-in: bare-symbol head per FlatPIR.
    out.op = calleeName;
  } else {
    // User-defined: routed via (%ref self <name>).
    out.target = { ns: 'self', name: calleeName };
  }
  if (args.length > 0)  out.args   = args;
  if (hasKwargs)        out.kwargs = kwargs;
  return out;
}

// ---------------------------------------------------------------------
// Reification: functionof, kernelof
//
// Surface form:
//   functionof(<body>, p1 = <Ident or Placeholder>, p2 = …)
//
// Each kwarg `kwName = value` declares a parameter:
//   - If `value` is an Identifier `id`           → param name is `id`
//   - If `value` is a Placeholder `_x_`          → param name is `_x_`
//     (with leading + trailing underscores, per FlatPIR §"Function
//     parameter lists" round-trip convention).
//
// We track BOTH the param name (visible as `%local` inside the body) and
// the surface kwarg name (used by callers at callsites). They differ for
// placeholder-form params and coincide for identifier-form params.

function _lowerReification(op, node, ctx) {
  const args = node.args;
  if (args.length === 0) {
    throw new Error(`lower: ${op} requires at least one argument (the body)`);
  }
  if (args[0].type === 'KeywordArg') {
    throw new Error(`lower: ${op}'s first argument must be the body, not a kwarg`);
  }

  const params = [];
  const paramKwargs = [];
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg.type !== 'KeywordArg') {
      throw new Error(`lower: ${op} parameters must be keyword args (got ${arg.type})`);
    }
    let paramName;
    if (arg.value.type === 'Identifier') {
      paramName = arg.value.name;
    } else if (arg.value.type === 'Placeholder') {
      paramName = '_' + arg.value.name + '_';
    } else {
      throw new Error(
        `lower: ${op} parameter '${arg.name}' must be bound to an identifier ` +
        `or placeholder, not ${arg.value.type}`
      );
    }
    params.push(paramName);
    paramKwargs.push(arg.name);
  }

  // Inner scope: existing %local plus the new params. Used for the body.
  const innerLocal = new Set(ctx.localScope || []);
  for (const p of params) innerLocal.add(p);
  const innerCtx = { ...ctx, localScope: innerLocal };

  return {
    kind:        'call',
    op,
    params,
    paramKwargs,
    body:        _lowerExpr(args[0], innerCtx),
    loc:         node.loc,
  };
}

// ---------------------------------------------------------------------
// `fn(<body>)` — the holes inside `body` are implicit parameters.
//
// The analyzer separately validates that holes appear only inside fn(...).
// We don't extract a params list here; the holes carry their position
// through `kind: 'hole'`. A future consumer that wants explicit param
// names can scan the body for holes (see analyzer's `countHoles`).

function _lowerFn(node, ctx) {
  if (node.args.length !== 1 || node.args[0].type === 'KeywordArg') {
    throw new Error(`lower: fn() requires exactly one expression argument`);
  }
  return {
    kind: 'call',
    op:   'fn',
    body: _lowerExpr(node.args[0], ctx),
    loc:  node.loc,
  };
}

// ---------------------------------------------------------------------
// Forms with ordered named entries (record, joint, jointchain, cartprod, table).
//
// Surface forms can mix positional and keyword arguments:
//   joint(M1, M2)           — purely positional
//   joint(a = M, b = N)     — purely keyword (= ordered fields)
//   joint(M1, b = N)        — mixed (rarer)
//
// Per FlatPIR §"Structural named entries", named entries use `(%field name value)`
// rather than `(%kwarg)` because order is part of the structure. Our IR
// shape: `fields: [{name, value}, …]` for the keyword half (preserving
// source order), `args: […]` for any positional half.

function _lowerFieldsForm(op, node, ctx) {
  const args = [];
  const fields = [];
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      fields.push({ name: arg.name, value: _lowerExpr(arg.value, ctx) });
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }
  const out = { kind: 'call', op, loc: node.loc };
  if (args.length > 0)   out.args   = args;
  if (fields.length > 0) out.fields = fields;
  return out;
}

// ---------------------------------------------------------------------
// Module-load forms.
//
// `load_module("path", <kwargs as substitutions>)` — kwargs are the
// `%assign` substitutions for the loaded module's free inputs. They're
// resolved at load time, distinct from runtime call kwargs, so we use a
// dedicated `assigns` field rather than `kwargs`.
//
// `standard_module(name, version)` — purely positional (just the path/version).

function _lowerModuleLoad(op, node, ctx) {
  const args = [];
  const assigns = {};
  let hasAssigns = false;
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      assigns[arg.name] = _lowerExpr(arg.value, ctx);
      hasAssigns = true;
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }
  const out = { kind: 'call', op, loc: node.loc };
  if (args.length > 0) out.args = args;
  if (hasAssigns)      out.assigns = assigns;
  return out;
}

// ---------------------------------------------------------------------

module.exports = {
  lowerExpr,
  lowerBinding,

  // Exported for tests / introspection
  _internal: {
    BIN_OP_MAP,
    UN_OP_MAP,
    FIELD_FORMS,
    REIFICATION_FORMS,
    MODULE_LOAD_FORMS,
  },
};
