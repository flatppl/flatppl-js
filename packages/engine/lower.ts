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
// directly. Per spec §11, only PLACEHOLDER params (`_x_`-class — lambda /
// fn-hole formals with no module binding) live in the `%local` namespace;
// body references to IDENTIFIER-form boundary inputs stay plain `self`/
// module refs designating the cut nodes — the params list carries the
// cut, and boundary-aware consumers apply the §04 input substitution at
// evaluation. The surface kwarg names are preserved in `paramKwargs` for
// callsite-keyword matching at higher layers.
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
// extend the scope with the new PLACEHOLDER params before recursing
// (identifier-form boundary names stay out — their body refs lower as
// ordinary `self` refs, spec §11).
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

import type { IRNode } from './engine-types';

const builtins = require('./builtins.ts');

const { CONSTANTS, SETS, BOOL_LITERALS, ALL_KNOWN, BUILTIN_FUNCTIONS, DISTRIBUTIONS } = builtins;

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
function lowerExpr(node: any, ctx?: any): IRNode {
  ctx = ctx || { localScope: null };
  return _lowerExpr(node, ctx);
}

/**
 * Lower an AssignStatement's RHS. Convenience wrapper around `lowerExpr`.
 */
function lowerBinding(stmt: any, ctx?: any): IRNode {
  if (!stmt || stmt.type !== 'AssignStatement') {
    throw new Error(`lower: expected AssignStatement, got ${stmt?.type}`);
  }
  return lowerExpr(stmt.value, ctx);
}

// ---------------------------------------------------------------------
// Operator → builtin-name mapping

const BIN_OP_MAP: Record<string, string> = {
  '+':  'add',
  '-':  'sub',
  '*':  'mul',
  // Spec §07: `/` is `divide` (true division, real/complex result), NOT
  // `div` (which is ⌊a/b⌋ integer floor division). Mapping `/`→`div`
  // mis-typed every `/` as integer-preserving (it routes through the
  // BINARY_ARITH_OPS unifyArith path) even though the runtime already
  // computed a/b. `divide` has a REAL signature and is not in
  // BINARY_ARITH_OPS, so it types via inferGenericCall → real.
  '/':  'divide',
  '<':  'lt',
  '<=': 'le',
  '>':  'gt',
  '>=': 'ge',
  '==': 'equal',
  '!=': 'unequal',
  'in': 'in',
};

const UN_OP_MAP: Record<string, string> = {
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

function _lowerExpr(node: any, ctx: any): IRNode {
  switch (node.type) {
    case 'NumberLiteral': {
      // Preserve the lexical integer-vs-real distinction. JS Numbers
      // can't tell `1.0` from `1` once parsed (both === 1), so we
      // record `numType` from the source text. Per FlatPIR spec
      // (Literal values §): integer if the digit pattern has no
      // decimal point or exponent, else real.
      const raw = String(node.raw != null ? node.raw : node.value);
      // Integer if the source lexeme is a decimal integer or a hex
      // literal (with optional underscore digit separators). Anything
      // with a decimal point or exponent classifies as real.
      const isInt = /^[+-]?(\d(_?\d)*|0x[0-9a-fA-F](_?[0-9a-fA-F])*)$/.test(raw);
      const numType = isInt ? 'integer' : 'real';
      return { kind: 'lit', value: node.value, numType, loc: node.loc };
    }

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

    case 'SliceOnly':
      // The `!` singleton-axis marker (spec §07). Lowers to bare
      // `only` — same shape as `all` so the evaluator's `get` op can
      // dispatch on the sentinel.
      return { kind: 'const', name: 'only', loc: node.loc };

    case 'Hole':
      return { kind: 'hole', loc: node.loc };

    case 'AxisRef': {
      // Axis label `.name` (spec §05) used by `aggregate` and
      // `metricsum`. Per FlatPIR §11 the bare axis lowers to
      // `(%axis <name>)`; variance-marked axes (only legal inside
      // `metricsum`, spec §04 §sec:metricsum) lower to
      // `(%uaxis <name>)` (contravariant / upper) or
      // `(%laxis <name>)` (covariant / lower). The analyzer enforces
      // that axes only appear inside an enclosing aggregate / metricsum;
      // the IR shape carries the axis name + variance so the metricsum
      // lift pass (which rewrites to `aggregate(sum, ...)` with metric /
      // inv(metric) factor insertions) can read the variance directly.
      const ir: any = { kind: 'axis', name: node.name, loc: node.loc };
      if (node.variance) ir.variance = node.variance;
      return ir;
    }

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
        args: node.elements.map((e: any) => _lowerExpr(e, ctx)),
        loc:  node.loc,
      };

    case 'TupleLiteral':
      // (a, b) — lowers to a `(tuple ...)` call.
      return {
        kind: 'call',
        op:   'tuple',
        args: node.elements.map((e: any) => _lowerExpr(e, ctx)),
        loc:  node.loc,
      };

    case 'BinaryExpr':
      return _lowerBinaryExpr(node, ctx);

    case 'UnaryExpr':
      return _lowerUnaryExpr(node, ctx);

    case 'IndexExpr':
      // a[i, j, …] → (get a i j …) for FlatPPL/FlatPPJ (1-based),
      // (get0 a i j …) for FlatPPY (0-based). The lowering op was
      // chosen at parse time and stored on the node so this code
      // path stays variant-agnostic.
      return {
        kind: 'call',
        op:   node.indexOp || 'get',
        args: [
          _lowerExpr(node.object, ctx),
          ...node.indices.map((i: any) => _lowerExpr(i, ctx)),
        ],
        loc: node.loc,
      };

    case 'FieldAccess': {
      // Per spec §04 name resolution: `self.foo` is the explicit
      // current-module reference; `base.foo` is the explicit built-in
      // reference; `mod.foo` (mod a loaded module) is the cross-
      // module reference. None of those are data-access — they're
      // name-resolution refs, lowered to spec §11's `(%ref ns name)`
      // / bare-headed (for base) shapes. Per spec §03/§07, only
      // record / table / tuple values get the data-access form
      // `get_field(value, "field")`.
      //
      // Discriminator (in order):
      //   - `self.X`  → (%ref self X)
      //   - `base.X`  → bare-headed when X is a known builtin (spec
      //                  §04: built-ins lower to bare form, never to
      //                  `(%ref base X)`); else surface error.
      //   - `mod.X`   → (%ref mod X) when `mod` is module-typed.
      //   - anything else → get_field(object, "field") (record /
      //                       table / tuple access).
      if (node.object && node.object.type === 'Identifier') {
        const objName = node.object.name;
        if (objName === 'self') {
          return {
            kind: 'ref', ns: 'self', name: node.field, loc: node.loc,
          };
        }
        if (objName === 'base') {
          // Spec §11: explicit built-in refs share the same bare-
          // headed IR shape as implicit built-in calls. The lowerer
          // emits a `ref` here with no `ns` — downstream consumers
          // that need to resolve a base-ref look it up by name in
          // the built-ins catalog. (Most code paths see `base.foo`
          // exclusively in HEAD position of a call, where _lower
          // CallExpr handles it; bare value-position `base.foo`
          // produces a callable ref the caller is expected to apply
          // immediately.)
          return {
            kind: 'ref', ns: 'base', name: node.field, loc: node.loc,
          };
        }
        if (ctx && ctx.moduleNames && ctx.moduleNames.has(objName)) {
          return {
            kind: 'ref', ns: objName, name: node.field, loc: node.loc,
          };
        }
      }
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
    }

    case 'CallExpr':
      return _lowerCallExpr(node, ctx);

    default:
      throw new Error(`lower: unsupported AST node type '${node.type}'`);
  }
}

function _lowerIdentifier(node: any, ctx: any): IRNode {
  const { name, loc } = node;

  // 1. %local scope wins (innermost reified scope's params).
  if (ctx.localScope && ctx.localScope.has(name)) {
    return { kind: 'ref', ns: '%local', name, loc };
  }

  // 2. Boolean literals — defensive path for any Identifier-typed
  // name that happens to spell a boolean across any variant. The
  // canonical path is the parser emitting AST.BoolLiteral directly.
  if (BOOL_LITERALS.has(name)) {
    return { kind: 'lit', value: (name === 'true' || name === 'True'), loc };
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

// Dot-notation broadcast (spec §05 / §04 higher-order): a Binary/Unary
// node tagged `broadcast:true` by the parser (`A .+ B`, `.- x`, `a .< b`,
// …) lowers to `broadcast(<functionof wrapping the op>, <operands…>)`.
// The synthesized functionof is byte-identical in shape to what
// `fn(<op>(_, …))` lowers to, so the entire existing broadcast path
// (`_resolveFn`'s functionof branch, `_broadcastApply`, kernel-broadcast
// in the materialiser) applies unchanged — no new runtime surface. This
// is also the single place that knows the operator→builtin map
// (BIN_OP_MAP / UN_OP_MAP); the parser never duplicates it.
function _synthOpFunctionof(opName: string, arity: number) {
  const params: any[] = [], paramKwargs: any[] = [], paramSources: any[] = [], bodyArgs: any[] = [];
  for (let i = 1; i <= arity; i++) {
    const p = `_arg${i}_`;
    params.push(p);
    paramKwargs.push(`arg${i}`);
    paramSources.push({ kind: 'placeholder', name: p });
    bodyArgs.push({ kind: 'ref', ns: '%local', name: p });
  }
  return {
    kind: 'call', op: 'functionof',
    params, paramKwargs, paramSources,
    body: { kind: 'call', op: opName, args: bodyArgs },
  };
}

function _lowerBinaryExpr(node: any, ctx: any): any {
  const op = BIN_OP_MAP[node.op];
  if (!op) {
    throw new Error(`lower: unknown binary operator '${node.op}'`);
  }
  const args: any[] = [_lowerExpr(node.left, ctx), _lowerExpr(node.right, ctx)];
  if (node.broadcast) {
    return {
      kind: 'call', op: 'broadcast',
      args: [_synthOpFunctionof(op, 2), ...args],
      loc:  node.loc,
    };
  }
  return { kind: 'call', op, args, loc: node.loc };
}

function _lowerUnaryExpr(node: any, ctx: any): any {
  const op = UN_OP_MAP[node.op];
  if (!op) {
    throw new Error(`lower: unknown unary operator '${node.op}'`);
  }
  // Literal-folding for negated numeric literals. Without this,
  // `-3` lowers to `neg(lit(3, integer))` which typeinfers as
  // `real` (the neg signature widens integers to real); inside an
  // array literal `[1, -3, 5]` the unification then fails with
  // "array element type mismatch: integer vs real". Folding `-N` /
  // `+N` to `lit(-N)` / `lit(N)` at lower-time preserves the
  // integer/real numType and matches how `28` itself lowers.
  // (Non-broadcast unary `-`/`+` on a literal only — broadcast
  // dotted-unary and unary on non-lit operands keep their original
  // shape.)
  if (!node.broadcast && (node.op === '-' || node.op === '+')
      && node.operand && node.operand.type === 'NumberLiteral'
      && typeof node.operand.value === 'number') {
    // Re-use the same integer-detection logic NumberLiteral uses
    // (raw-source pattern, not Number.isInteger — `1.0` is integer-
    // valued but spec-real).
    const raw = String(node.operand.raw != null ? node.operand.raw : node.operand.value);
    const isInt = /^[+-]?(\d(_?\d)*|0x[0-9a-fA-F](_?[0-9a-fA-F])*)$/.test(raw);
    const v = node.op === '-' ? -node.operand.value : node.operand.value;
    return { kind: 'lit', value: v, numType: isInt ? 'integer' : 'real', loc: node.loc };
  }
  const args: any[] = [_lowerExpr(node.operand, ctx)];
  if (node.broadcast) {
    return {
      kind: 'call', op: 'broadcast',
      args: [_synthOpFunctionof(op, 1), ...args],
      loc:  node.loc,
    };
  }
  return { kind: 'call', op, args, loc: node.loc };
}

function _lowerCallExpr(node: any, ctx: any): any {
  if (!node.callee || node.callee.type !== 'Identifier') {
    // Special case: `broadcasted(f)(args...)` — the spec §04
    // curried form `broadcasted(f)(args) ≡ broadcast(f, args)`.
    // Lower this directly to a `broadcast` IR, avoiding the
    // synthesised inner `fn(f(_, _, …))` wrapper the runtime lift
    // would otherwise add. This makes inline forms like
    // `fn(broadcasted(f)(a, _, c))` lower cleanly (without this,
    // the outer call's non-Identifier callee would land in the
    // lit-null catch in pir.lowerToModule). The body of the
    // resulting `broadcast` keeps `f` as a bare ref / call expr;
    // the runtime broadcast handler and inferBroadcast both
    // resolve user-fn refs directly, so no wrapper is needed.
    if (node.callee && node.callee.type === 'CallExpr'
        && node.callee.callee && node.callee.callee.type === 'Identifier'
        && node.callee.callee.name === 'broadcasted') {
      const bcArgs = (node.callee.args || []).filter(
        (a: any) => a && a.type !== 'KeywordArg');
      if (bcArgs.length === 1) {
        const f = _lowerExpr(bcArgs[0], ctx);
        const lowered: any = { kind: 'call', op: 'broadcast', loc: node.loc };
        const posArgs: any[] = [];
        const kwargs: Record<string, any> = {};
        let hasKwargs = false;
        for (const a of node.args || []) {
          if (a.type === 'KeywordArg') {
            kwargs[a.name] = _lowerExpr(a.value, ctx);
            hasKwargs = true;
          } else {
            posArgs.push(_lowerExpr(a, ctx));
          }
        }
        lowered.args = [f, ...posArgs];
        if (hasKwargs) lowered.kwargs = kwargs;
        return lowered;
      }
    }
    // Expression-headed user call (spec §11, b070d0a / §05 Postfix
    // Call*): the callee is an expression that must evaluate to a
    // user-defined callable — an inline reification in the common case
    // (`functionof(e, p = a)(2.5)`, `(x -> 2*x)(3.0)`), or a chained
    // call (`f(x)(y)`). Lower to the callee-form user call: the head
    // lives in `callee` (a child IR node), structurally distinct from
    // the ref-headed `target` form (a {ns,name} string pair). The
    // "must evaluate to a callable" condition is inference's job —
    // the lowerer stays liberal (matching the FlatPIR reader).
    if (node.callee && node.callee.type === 'CallExpr') {
      const lowered: any = {
        kind: 'call',
        callee: _lowerExpr(node.callee, ctx),
        loc: node.loc,
      };
      const posArgs: any[] = [];
      const kw: Record<string, any> = {};
      let hasKw = false;
      for (const a of node.args || []) {
        if (a.type === 'KeywordArg') {
          kw[a.name] = _lowerExpr(a.value, ctx);
          hasKw = true;
        } else {
          posArgs.push(_lowerExpr(a, ctx));
        }
      }
      lowered.args = posArgs;
      if (hasKw) lowered.kwargs = kw;
      return lowered;
    }
    // Otherwise: refuse loudly (a computed callee shape outside the
    // grammar — e.g. an indexing-produced callable — has no IR form yet).
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
  if (calleeName === 'broadcast') {
    return _lowerBroadcast(node, ctx);
  }
  if (calleeName === 'builtin_logdensityof') {
    return _lowerBuiltinLogdensityof(node, ctx);
  }
  if (calleeName === 'builtin_touniform' || calleeName === 'builtin_fromuniform'
      || calleeName === 'builtin_tonormal' || calleeName === 'builtin_fromnormal') {
    return _lowerBuiltinTransport(calleeName, node, ctx);
  }
  if (calleeName === 'builtin_sample') {
    return _lowerBuiltinSample(node, ctx);
  }

  // General call: built-in (we know its name) vs user-defined (we don't).
  // The analyzer's collected `definedNames` set could refine this, but
  // for IR purposes the head-shape distinction (op vs target) is enough.
  // Built-in names are listed in `builtins.ALL_KNOWN`.

  const args: any[] = [];
  const kwargs: Record<string, any> = {};
  let hasKwargs = false;
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      kwargs[arg.name] = _lowerExpr(arg.value, ctx);
      hasKwargs = true;
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }

  const out: any = { kind: 'call', loc: node.loc };
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

// broadcast(callable, operands…) — spec §04 higher-order. Mirrors the
// general-call path but normalizes a *bare builtin-function* callee
// (`broadcast(add, A, B)` — the literal lowering the spec mandates for
// `A .+ B`, and the shape every parser-desugared dotted operator
// `.^ .&& .|| .!` produces) into the SAME functionof a
// `fn(add(_, …))` would lower to. The runtime then resolves it via
// the existing functionof branch of `_resolveFn` — no new runtime
// surface. Untouched, by design:
//   • distributions (`Normal`, …) — not in BUILTIN_FUNCTIONS, so
//     kernel-broadcast in the materialiser still sees the bare name;
//   • user functions / `fn(…)` / `functionof(…)` — not bare
//     Identifiers in BUILTIN_FUNCTIONS, lowered as before.
function _lowerBroadcast(node: any, ctx: any) {
  const calleeArg = node.args && node.args[0];
  if (!calleeArg) {
    // `broadcast()` — invalid; let the evaluator raise the proper
    // "no function argument" error (parity with the old path).
    return { kind: 'call', op: 'broadcast', loc: node.loc };
  }
  const posArgs: any[] = [];
  const kwargs: Record<string, any> = {};
  const kwOrder: string[] = [];
  let hasKwargs = false;
  for (const arg of node.args.slice(1)) {
    if (arg.type === 'KeywordArg') {
      kwargs[arg.name] = _lowerExpr(arg.value, ctx);
      kwOrder.push(arg.name);
      hasKwargs = true;
    } else {
      posArgs.push(_lowerExpr(arg, ctx));
    }
  }
  let head: any;
  if (calleeArg.type === 'Identifier'
      && BUILTIN_FUNCTIONS.has(calleeArg.name)) {
    const arity = hasKwargs ? kwOrder.length : posArgs.length;
    head = _synthOpFunctionof(calleeArg.name, arity);
    // Keyword form: match the synthesized params to the caller's
    // kwarg names (in order) so `_broadcastApply`'s kwarg path binds.
    if (hasKwargs) head.paramKwargs = kwOrder.slice();
  } else {
    head = _lowerExpr(calleeArg, ctx);
  }
  const out: any = {
    kind: 'call', op: 'broadcast',
    args: [head, ...posArgs], loc: node.loc,
  };
  if (hasKwargs) out.kwargs = kwargs;
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

// builtin_logdensityof(kernel, kernel_input, x) — FlatPDL primitive,
// spec §07 §sec:measure-eval-prims. The first arg must be a bare
// distribution Identifier (built-in kernel constructor — `Normal`,
// `MvNormal`, …); lower it to a `{kind:'lit', value:<name>}` so the
// analyzer doesn't see it as an undefined variable and the evaluator
// can read the kernel name directly without resolving a ref. The
// other two args lower normally.
function _lowerBuiltinLogdensityof(node: any, ctx: any): any {
  const args = node.args || [];
  if (args.length !== 3) {
    throw new Error(`lower: builtin_logdensityof expects 3 positional `
      + `arguments (kernel, kernel_input, x), got ${args.length}`);
  }
  for (const a of args) {
    if (a.type === 'KeywordArg') {
      throw new Error('lower: builtin_logdensityof does not accept keyword arguments');
    }
  }
  const kernelArg = args[0];
  if (kernelArg.type !== 'Identifier' || !DISTRIBUTIONS.has(kernelArg.name)) {
    throw new Error('lower: builtin_logdensityof: kernel argument must be a built-in '
      + 'distribution name (e.g. Normal, MvNormal); got '
      + (kernelArg.type === 'Identifier' ? `'${kernelArg.name}'` : kernelArg.type));
  }
  return {
    kind: 'call', op: 'builtin_logdensityof',
    args: [
      { kind: 'lit', value: kernelArg.name, loc: kernelArg.loc },
      _lowerExpr(args[1], ctx),
      _lowerExpr(args[2], ctx),
    ],
    loc: node.loc,
  };
}

// Shared lowerer for `builtin_touniform` / `builtin_fromuniform` /
// `builtin_tonormal` / `builtin_fromnormal`. Same 3-arg signature
// (kernel, kernel_input, x|u|z) and same kernel-name normalisation
// as `_lowerBuiltinLogdensityof`. Pulled into one helper because the
// four functions differ only in their op name.
function _lowerBuiltinTransport(op: string, node: any, ctx: any): any {
  const args = node.args || [];
  if (args.length !== 3) {
    throw new Error(`lower: ${op} expects 3 positional arguments `
      + `(kernel, kernel_input, point), got ${args.length}`);
  }
  for (const a of args) {
    if (a.type === 'KeywordArg') {
      throw new Error(`lower: ${op} does not accept keyword arguments`);
    }
  }
  const kernelArg = args[0];
  if (kernelArg.type !== 'Identifier' || !DISTRIBUTIONS.has(kernelArg.name)) {
    throw new Error(`lower: ${op}: kernel argument must be a built-in `
      + 'distribution name (e.g. Normal, MvNormal); got '
      + (kernelArg.type === 'Identifier' ? `'${kernelArg.name}'` : kernelArg.type));
  }
  return {
    kind: 'call', op,
    args: [
      { kind: 'lit', value: kernelArg.name, loc: kernelArg.loc },
      _lowerExpr(args[1], ctx),
      _lowerExpr(args[2], ctx),
    ],
    loc: node.loc,
  };
}

// builtin_sample(rngstate, kernel, kernel_input, n, m, ...) — FlatPDL
// primitive. Variadic trailing dims (each a positive-integer expression).
// Returns the spec-defined tuple `(X, new_rngstate)`. We keep the
// kernel_input IR intact so the evaluator can pass it to the measure walker (no
// kwarg pre-evaluation, no special record-only restriction at lower
// time — the evaluator enforces shape constraints with clearer errors).
function _lowerBuiltinSample(node: any, ctx: any): any {
  const args = node.args || [];
  if (args.length < 3) {
    throw new Error('lower: builtin_sample expects at least 3 positional '
      + 'arguments (rngstate, kernel, kernel_input, [n, m, ...]); got '
      + args.length);
  }
  for (const a of args) {
    if (a.type === 'KeywordArg') {
      throw new Error('lower: builtin_sample does not accept keyword arguments');
    }
  }
  const kernelArg = args[1];
  if (kernelArg.type !== 'Identifier' || !DISTRIBUTIONS.has(kernelArg.name)) {
    throw new Error('lower: builtin_sample: kernel argument must be a built-in '
      + 'distribution name (e.g. Normal, MvNormal); got '
      + (kernelArg.type === 'Identifier' ? `'${kernelArg.name}'` : kernelArg.type));
  }
  const lowered: any[] = [
    _lowerExpr(args[0], ctx),
    { kind: 'lit', value: kernelArg.name, loc: kernelArg.loc },
    _lowerExpr(args[2], ctx),
  ];
  for (let i = 3; i < args.length; i++) lowered.push(_lowerExpr(args[i], ctx));
  return { kind: 'call', op: 'builtin_sample', args: lowered, loc: node.loc };
}

function _lowerReification(op: string, node: any, ctx: any): any {
  const args = node.args;
  if (args.length === 0) {
    throw new Error(`lower: ${op} requires at least one argument (the body)`);
  }
  if (args[0].type === 'KeywordArg') {
    throw new Error(`lower: ${op}'s first argument must be the body, not a kwarg`);
  }

  const params: any[] = [];
  const paramKwargs: any[] = [];
  // paramSources records the structural origin of each boundary kwarg
  // so downstream UI (profile-plot auto-range) can resolve it to an
  // empirical range or a set restriction *retroactively* — at plot
  // time we don't have samples; here we do have the boundary's
  // surface form (Identifier vs Placeholder). The viewer fetches
  // getMeasure(name) for 'binding' sources to compute a 4-σ-quantile
  // range; for 'placeholder' sources it consults the corresponding
  // elementof binding's set restriction.
  const paramSources: any[] = [];
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg.type !== 'KeywordArg') {
      throw new Error(`lower: ${op} parameters must be keyword args (got ${arg.type})`);
    }
    let paramName: string;
    let source: any;
    if (arg.value.type === 'Identifier') {
      paramName = arg.value.name;
      // An identifier that designates a real module node is a boundary
      // CUT (spec §04 — "the trace stops at nodes a and d"); one that
      // names nothing is a pure FORMAL, semantically a placeholder (the
      // spec §04 lambda rule binds formals as `x = _x_`; the engine
      // accepts the bare-identifier shorthand `x = x`). The distinction
      // drives the body namespace: cut refs stay `self` (spec §11),
      // formals live in `%local`. Without `ctx.bindingNames` (legacy
      // re-lower fallback paths) default to the cut reading — the
      // dual-ns consumers handle either shape.
      if (ctx.bindingNames && !ctx.bindingNames.has(paramName)) {
        source = { kind: 'placeholder', name: paramName };
      } else {
        source = { kind: 'binding', name: arg.value.name };
      }
    } else if (arg.value.type === 'Placeholder') {
      paramName = '_' + arg.value.name + '_';
      source = { kind: 'placeholder', name: paramName };
    } else {
      throw new Error(
        `lower: ${op} parameter '${arg.name}' must be bound to an identifier ` +
        `or placeholder, not ${arg.value.type}`
      );
    }
    params.push(paramName);
    paramKwargs.push(arg.name);
    paramSources.push(source);
  }

  // Inner scope: existing %local plus the new PLACEHOLDER params only
  // (spec §11: `%local` is the placeholder namespace — `_x_`-class names
  // with no module binding). Identifier-form boundary refs in the body
  // stay plain `self`/module refs designating the cut nodes; the entry
  // list (`params`/`paramKwargs`/`paramSources`) carries the cut, and
  // the §04 input substitution is applied at evaluation/lowering by the
  // boundary-aware consumers (scope-aware collectSelfRefs etc.), never
  // encoded in the namespace.
  const innerLocal = new Set(ctx.localScope || []);
  for (let i = 0; i < params.length; i++) {
    if (paramSources[i].kind === 'placeholder') innerLocal.add(params[i]);
  }
  const innerCtx = { ...ctx, localScope: innerLocal };

  let body: any = _lowerExpr(args[0], innerCtx);

  // kernelof(x, kw) ≡ functionof(lawof(x), kw) per spec §sec:kernelof
  // line 421-422. The boundary substitution applies BEFORE the inner
  // lawof is interpreted — exactly what we get by wrapping the body
  // in lawof inside the same scope and emitting functionof. After
  // this rewrite, downstream IR consumers (type inference,
  // orchestrator) see one uniform reification form.
  let outOp = op;
  if (op === 'kernelof') {
    body = { kind: 'call', op: 'lawof', args: [body], loc: args[0].loc };
    outOp = 'functionof';
  }

  return {
    kind:        'call',
    op:          outOp,
    params,
    paramKwargs,
    paramSources,
    body,
    loc:         node.loc,
  };
}

// ---------------------------------------------------------------------
// `fn(<body>)` is sugar for `functionof(<body-with-placeholders>,
// arg1=_arg1_, arg2=_arg2_, ...)` per spec §sec:fn (lines 618-636).
// We perform that lowering here so downstream IR consumers see only
// `functionof` and the holes are gone — every parameter is named.
//
// Hole numbering is left-to-right reading order on the AST. We walk
// the body, replace each Hole AST node with a Placeholder named
// `_argN_`, count holes, then synthesize the equivalent functionof
// IR with `params` / `paramKwargs` matching the holes' indices.

function _lowerFn(node: any, ctx: any) {
  if (node.args.length !== 1 || node.args[0].type === 'KeywordArg') {
    throw new Error(`lower: fn() requires exactly one expression argument`);
  }
  const bodyAst = node.args[0];

  // Walk the body in reading order; replace each Hole with a
  // Placeholder named `arg1`, `arg2`, … (inner-name convention,
  // matching the parser; _lowerReification will wrap with the
  // surrounding underscores). We deep-clone so the original AST
  // isn't touched — other consumers (DAG render, source diagnostics)
  // still see the user's `fn(_)` form.
  let counter = 0;
  function rewriteHoles(astNode: any): any {
    if (astNode == null || typeof astNode !== 'object') return astNode;
    if (Array.isArray(astNode)) return astNode.map(rewriteHoles);
    if (astNode.type === 'Hole') {
      counter++;
      return { type: 'Placeholder', name: 'arg' + counter, loc: astNode.loc };
    }
    const out: Record<string, any> = {};
    for (const k in astNode) out[k] = rewriteHoles(astNode[k]);
    return out;
  }
  const rewrittenBody = rewriteHoles(bodyAst);
  const numHoles = counter;

  // Build a synthesized `functionof(<body>, arg1=_arg1_, …)` AST and
  // delegate to the existing reification lowering. One code path,
  // no duplicate scope-tracking logic.
  const kwargs: any[] = [];
  for (let i = 1; i <= numHoles; i++) {
    kwargs.push({
      type: 'KeywordArg',
      name: 'arg' + i,
      value: { type: 'Placeholder', name: 'arg' + i, loc: bodyAst.loc },
      loc: bodyAst.loc,
    });
  }
  const synthesized = {
    type:   'CallExpr',
    callee: { type: 'Identifier', name: 'functionof', loc: node.callee.loc },
    args:   [rewrittenBody, ...kwargs],
    loc:    node.loc,
  };
  return _lowerReification('functionof', synthesized, ctx);
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

function _lowerFieldsForm(op: string, node: any, ctx: any) {
  const args: any[] = [];
  const fields: any[] = [];
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      fields.push({ name: arg.name, value: _lowerExpr(arg.value, ctx) });
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }
  const out: any = { kind: 'call', op, loc: node.loc };
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

function _lowerModuleLoad(op: string, node: any, ctx: any) {
  const args: any[] = [];
  const assigns: Record<string, any> = {};
  let hasAssigns = false;
  for (const arg of node.args) {
    if (arg.type === 'KeywordArg') {
      assigns[arg.name] = _lowerExpr(arg.value, ctx);
      hasAssigns = true;
    } else {
      args.push(_lowerExpr(arg, ctx));
    }
  }
  const out: any = { kind: 'call', op, loc: node.loc };
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
