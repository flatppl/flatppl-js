'use strict';

// FlatPIR structural type inference, Phase 2 — walks the analyzer's
// bindings map, attaches an `inferredType` to each binding, and
// returns the diagnostics found along the way (type mismatches at
// call sites, unknown references, cycles).
//
// Design
// ======
// One pass over the bindings, with on-demand recursion for refs.
// Cycle detection via a `visiting` set: if inferring binding A leads
// back to A before A's type is known, the cycle is broken with
// (%failed "cyclic …") and downstream uses see `failed`. Once a
// binding is resolved its type is memoised on the binding info so
// the same binding isn't walked twice.
//
// Inference rules (high level)
// ----------------------------
//   * NumberLiteral        — INTEGER if lexically integer-shaped, else REAL.
//   * StringLiteral        — STRING.
//   * BoolLiteral          — BOOLEAN.
//   * ConstantRef pi/inf   — REAL.   im → COMPLEX.   true/false → BOOLEAN.
//   * SetRef <name>        — internal "set" marker (consumed by elementof).
//   * Identifier <name>    — recursively infer the named binding.
//   * Hole / Placeholder   — %any.
//   * ArrayLiteral         — array<unify of elements, length-N>.
//   * TupleLiteral         — tuple<elementwise>.
//   * BinaryExpr/UnaryExpr — desugared to the matching call op via
//                            BIN_OP_TO_NAME / UN_OP_TO_NAME.
//   * CallExpr <op>        — look up signature, unify args, instantiate
//                            result. Handful of "special" ops (record,
//                            joint, tuple, lawof, elementof) get bespoke
//                            handling because their structural shape
//                            depends on actuals (e.g. record's fields).
//   * IndexExpr/FieldAccess — partially handled (record field access
//                              works; array indexing returns deferred).
//
// Subtyping
// ---------
// Numeric promotion (booleans ⊂ integers ⊂ reals → complexes) lives
// in types.js's unify() — the inference pass is unaware of it. So a
// kwarg signature `mu: real` accepts an integer literal `0` cleanly,
// matching the spec's "may use these canonical embeddings implicitly".
//
// What's NOT here
// ---------------
//   * User-defined function calls ((%call (%ref ...) args)) — deferred
//     until kernels/functions are implemented in the engine.
//   * load_data, load_module — return %deferred for now; load_data
//     specifically should yield a %table with %nrows %dynamic, but we
//     don't have the %table category implemented yet.
//   * Cross-module inference (load_module into another module).
//   * Phase information — already lives in analyzer.computePhases.

const T = require('./types');
const builtins = require('./builtins');

// Lexical-form integer detection. NumberLiteral.raw preserves the
// source text; if there's no decimal point or exponent, the literal
// is integer-typed per spec §sec:flatpir Literal values.
function isIntegerLiteral(raw) {
  if (raw == null) return false;
  return /^[+-]?\d+$/.test(String(raw));
}

// Built-in constant → type. Mirrors CONSTANTS in builtins.js. The
// analyzer surfaces these as ConstantRef nodes; we map by name.
const CONST_TYPES = {
  pi:    T.REAL,
  inf:   T.REAL,
  im:    T.COMPLEX,
  true:  T.BOOLEAN,
  false: T.BOOLEAN,
};

// Set name → element type. Used by elementof: `elementof(reals)` is
// real-typed, `elementof(integers)` is integer-typed. Refinement
// (posreals, unitinterval, etc.) collapses to its structural category
// per spec §sec:flatpir "Sets and types are distinct".
const SET_VALUE_TYPES = {
  reals: T.REAL, posreals: T.REAL, nonnegreals: T.REAL, unitinterval: T.REAL,
  integers: T.INTEGER, posintegers: T.INTEGER, nonnegintegers: T.INTEGER,
  booleans: T.BOOLEAN,
  complexes: T.COMPLEX,
  rngstates: T.any(),
  anything: T.any(),
};

// Surface AST operators → built-in op names. Mirrors BIN_OP_MAP /
// UN_OP_MAP in lower.js but kept locally to avoid pulling lower.js
// (which requires the full builtins map) just for two tables.
const BIN_OP_TO_NAME = {
  '+':  'add',  '-':  'sub',  '*':  'mul',  '/': 'div',
  '<':  'lt',   '<=': 'le',   '>':  'gt',   '>=': 'ge',
  '==': 'equal', '!=': 'unequal',
};
const UN_OP_TO_NAME = { '-': 'neg', '+': 'pos' };

/**
 * Run structural type inference over `bindings` (the analyzer's
 * Map<string, BindingInfo>). Mutates each binding to set
 * `binding.inferredType`. Returns a flat array of diagnostics in the
 * standard {severity, message, loc} shape (matches existing analyzer
 * diagnostics so they merge into one stream).
 *
 * Idempotent: re-running on the same map is a no-op (the second pass
 * sees `inferredType` already set and short-circuits).
 */
function inferTypes(bindings) {
  const diagnostics = [];
  const visiting = new Set();   // names currently being inferred (cycle guard)
  const visited  = new Set();   // names whose type is finalised

  for (const [name] of bindings) inferBinding(name);
  return diagnostics;

  function inferBinding(name) {
    const binding = bindings.get(name);
    if (!binding) return T.failed('unknown binding "' + name + '"');
    if (visited.has(name))  return binding.inferredType || T.deferred();
    if (visiting.has(name)) {
      const t = T.failed('cyclic type inference at "' + name + '"');
      binding.inferredType = t;
      return t;
    }
    visiting.add(name);
    const t = (binding.node && binding.node.value)
      ? inferExpr(binding.node.value)
      : T.failed('"' + name + '" has no RHS');
    visiting.delete(name);
    visited.add(name);
    binding.inferredType = t;
    return t;
  }

  function inferExpr(astNode) {
    if (!astNode) return T.failed('null expression');
    switch (astNode.type) {
      case 'NumberLiteral':
        return isIntegerLiteral(astNode.raw) ? T.INTEGER : T.REAL;
      case 'StringLiteral': return T.STRING;
      case 'BoolLiteral':   return T.BOOLEAN;
      case 'ConstantRef':   return CONST_TYPES[astNode.name] || T.any();
      case 'SetRef':        return setMarker(astNode.name);
      case 'Identifier':    return inferIdentifier(astNode);
      case 'Hole':
      case 'Placeholder':   return T.any();
      case 'ArrayLiteral':  return inferArrayLiteral(astNode);
      case 'TupleLiteral':  return T.tuple(astNode.elements.map(inferExpr));
      case 'BinaryExpr':    return inferOp(BIN_OP_TO_NAME[astNode.op], astNode.op,
                                            [astNode.left, astNode.right], {}, astNode.loc);
      case 'UnaryExpr':     return inferOp(UN_OP_TO_NAME[astNode.op],  astNode.op,
                                            [astNode.operand], {}, astNode.loc);
      case 'CallExpr':      return inferCall(astNode);
      case 'FieldAccess':   return inferFieldAccess(astNode);
      case 'IndexExpr':     return T.deferred();   // shape-aware indexing TBD
    }
    return T.deferred();
  }

  function inferIdentifier(node) {
    if (bindings.has(node.name))           return inferBinding(node.name);
    if (builtins.isConstant(node.name))    return CONST_TYPES[node.name] || T.any();
    if (builtins.isSet(node.name))         return setMarker(node.name);
    // Names like 'self' / 'base' / load_module aliases shouldn't appear
    // bare in expressions; the analyzer already flagged unknown refs as
    // warnings. We mark as failed so dependents propagate cleanly.
    return T.failed('undefined name "' + node.name + '"');
  }

  function inferArrayLiteral(node) {
    if (node.elements.length === 0) return T.array(1, [0], T.any());
    const elemTypes = node.elements.map(inferExpr);
    let s = new Map();
    let elem = elemTypes[0];
    for (let i = 1; i < elemTypes.length; i++) {
      const next = T.unify(elem, elemTypes[i], s);
      if (next == null) {
        diagnostics.push({
          severity: 'error',
          message: 'array element type mismatch: '
            + T.show(elem) + ' vs ' + T.show(elemTypes[i]),
          loc: node.elements[i].loc,
        });
        return T.failed('array element mismatch');
      }
      s = next;
      elem = T.substitute(elem, s);
    }
    return T.array(1, [node.elements.length], T.substitute(elem, s));
  }

  function inferCall(node) {
    if (!node.callee || node.callee.type !== 'Identifier') return T.deferred();
    const op = node.callee.name;
    // Split positional vs kwargs (KeywordArg nodes are kwargs).
    const positional = [];
    const kwargs = {};
    for (const a of (node.args || [])) {
      if (a && a.type === 'KeywordArg') kwargs[a.name] = a.value;
      else positional.push(a);
    }
    return inferOp(op, op, positional, kwargs, node.loc);
  }

  // Generic call-site inference. `op` is the registry key; `displayOp`
  // is what we put in diagnostic text (for binary/unary ops we want
  // the surface symbol, not the lowered name).
  function inferOp(op, displayOp, positional, kwargs, callLoc) {
    if (!op) return T.failed('unsupported operator "' + displayOp + '"');

    // Special-cased ops whose result type depends on actuals in ways
    // the static signature table can't express.
    switch (op) {
      case 'elementof': return inferElementof(positional, callLoc);
      case 'lawof':     return inferLawof(positional, callLoc);
      case 'record':    return inferRecord(kwargs);
      case 'joint':     return inferJoint(kwargs, callLoc);
      case 'tuple':     return T.tuple(positional.map(inferExpr));
    }

    const sig = T.signatureOf(op);
    if (!sig) return T.deferred();   // unknown / not yet typed → don't error

    let s = new Map();

    // Positional arguments
    if (sig.args !== null) {
      const rawN = sig.args.length;
      const got  = positional.length;
      const variadic = sig.variadic === 'positional';
      const fixedN = variadic ? rawN - 1 : rawN;
      if (variadic) {
        if (got < fixedN) return arityError(displayOp, '≥' + fixedN, got, callLoc);
      } else if (got !== rawN) {
        return arityError(displayOp, rawN, got, callLoc);
      }
      // Check the fixed positions
      for (let i = 0; i < fixedN; i++) {
        const at = inferExpr(positional[i]);
        const next = T.unify(sig.args[i], at, s);
        if (next == null) return argError(displayOp, i, sig.args[i], at, positional[i].loc);
        s = next;
      }
      // Check the variadic tail (every remaining actual unifies with
      // the last expected type).
      if (variadic) {
        const tail = sig.args[rawN - 1];
        for (let i = fixedN; i < got; i++) {
          const at = inferExpr(positional[i]);
          const next = T.unify(tail, at, s);
          if (next == null) return argError(displayOp, i, tail, at, positional[i].loc);
          s = next;
        }
      }
    }

    // Required kwargs. Missing kwargs are NOT an error here — many
    // kwargs have defaults the engine would supply at lowering time
    // (e.g. distribution parameter defaults). Unknown kwargs are
    // also not flagged; we may not model every accepted parameter.
    for (const k in sig.kwargs) {
      if (!(k in kwargs)) continue;
      const at = inferExpr(kwargs[k]);
      const next = T.unify(sig.kwargs[k], at, s);
      if (next == null) return kwargError(displayOp, k, sig.kwargs[k], at, kwargs[k].loc);
      s = next;
    }

    return T.substitute(sig.result, s);
  }

  // ---- Special-case op handlers --------------------------------------

  function inferElementof(positional, loc) {
    if (positional.length !== 1) return arityError('elementof', 1, positional.length, loc);
    const t = setValueType(positional[0]);
    if (t == null) {
      diagnostics.push({
        severity: 'error',
        message: 'elementof expects a set or set-constructor expression; got '
          + T.show(inferExpr(positional[0])),
        loc: positional[0].loc,
      });
      return T.failed('elementof bad arg');
    }
    return t;
  }

  /**
   * Resolve a set expression to the type of values that "live in" it,
   * for elementof. Returns null if the expression isn't a recognised
   * set form. Sets aren't first-class types (per spec §sec:flatpir
   * "Sets and types are distinct"), so we only walk far enough to get
   * a value-type answer for the elementof use-case.
   *
   * Supported shapes:
   *   reals / posreals / nonnegreals / unitinterval / …  → scalar real
   *   integers / posintegers / nonnegintegers           → scalar integer
   *   booleans                                          → scalar boolean
   *   complexes                                         → scalar complex
   *   anything / rngstates                              → %any
   *   interval(lo, hi)                                  → real
   *   stdsimplex(n)                                     → array<1, [n], real>
   *   cartpow(S, n)                                     → array<1, [n], elem(S)>
   *   cartpow(S, n, m, …)                               → array<rank, [n,m,…], elem(S)>
   *   cartprod(S1, S2, …)                               → tuple<elem(S1), …>  (positional)
   *   cartprod(a=S1, b=S2, …)                           → record{a: elem(S1), …}  (kwargs)
   */
  function setValueType(node) {
    if (!node) return null;
    if (node.type === 'SetRef')           return SET_VALUE_TYPES[node.name] || T.any();
    if (node.type === 'Identifier' && builtins.isSet(node.name))
                                          return SET_VALUE_TYPES[node.name] || T.any();
    if (node.type !== 'CallExpr')         return null;
    if (!node.callee || node.callee.type !== 'Identifier') return null;
    const op = node.callee.name;
    switch (op) {
      case 'interval':   return T.REAL;
      case 'stdsimplex': {
        const n = literalIntFrom(node.args[0]);
        return T.array(1, [n != null ? n : '%dynamic'], T.REAL);
      }
      case 'cartpow': {
        const inner = setValueType(node.args[0]);
        if (inner == null) return null;
        const dims = node.args.slice(1).map(literalIntFrom)
          .map(n => n != null ? n : '%dynamic');
        return T.array(dims.length, dims, inner);
      }
      case 'cartprod': {
        // Split positional vs keyword. The two forms produce different
        // shapes per spec §sec:valuetypes Sets.
        const pos = [], kw = {};
        for (const a of node.args) {
          if (a && a.type === 'KeywordArg') kw[a.name] = a.value;
          else pos.push(a);
        }
        if (Object.keys(kw).length > 0) {
          const fields = {};
          for (const k in kw) {
            const t = setValueType(kw[k]);
            if (t == null) return null;
            fields[k] = t;
          }
          return T.record(fields);
        }
        const elems = pos.map(setValueType);
        if (elems.some(e => e == null)) return null;
        return elems.length === 1 ? elems[0] : T.tuple(elems);
      }
    }
    return null;
  }

  /** Pull an integer literal out of an AST node, or null if not literal. */
  function literalIntFrom(node) {
    if (!node) return null;
    if (node.type === 'NumberLiteral' && isIntegerLiteral(node.raw)) return node.value;
    return null;
  }

  function inferLawof(positional, loc) {
    if (positional.length !== 1) return arityError('lawof', 1, positional.length, loc);
    const at = inferExpr(positional[0]);
    // lawof of a measure is permitted (the spec's identity law makes
    // it redundant but valid) — return as-is.
    if (T.isMeasure(at)) return at;
    if (T.isValue(at))   return T.measure(at);
    diagnostics.push({
      severity: 'error',
      message: 'lawof expects a value-typed argument, got ' + T.show(at),
      loc: positional[0].loc,
    });
    return T.failed('lawof bad arg');
  }

  function inferRecord(kwargs) {
    const fields = {};
    for (const k in kwargs) fields[k] = inferExpr(kwargs[k]);
    return T.record(fields);
  }

  function inferJoint(kwargs, callLoc) {
    // joint(name1=M1, name2=M2, ...) → measure<record{name1: T1, name2: T2, ...}>
    // Each kwarg must be measure-typed; we extract its domain.
    const fields = {};
    for (const k in kwargs) {
      const at = inferExpr(kwargs[k]);
      if (T.isMeasure(at)) {
        fields[k] = at.domain;
      } else if (at.kind === 'deferred' || at.kind === 'any') {
        fields[k] = T.deferred();
      } else {
        diagnostics.push({
          severity: 'error',
          message: 'joint kwarg "' + k + '" expects a measure, got ' + T.show(at),
          loc: kwargs[k].loc || callLoc,
        });
        return T.failed('joint bad kwarg');
      }
    }
    return T.measure(T.record(fields));
  }

  // ---- Field access --------------------------------------------------

  function inferFieldAccess(node) {
    const objT = inferExpr(node.object);
    if (objT.kind === 'record' && objT.fields[node.field]) return objT.fields[node.field];
    if (objT.kind === 'record') {
      diagnostics.push({
        severity: 'error',
        message: 'unknown field "' + node.field + '" on record',
        loc: node.loc,
      });
      return T.failed('unknown field');
    }
    return T.deferred();   // tables, modules, deferred objects
  }

  // ---- Diagnostics helpers -------------------------------------------

  function arityError(op, expected, got, loc) {
    diagnostics.push({
      severity: 'error',
      message: op + ' expects ' + expected + ' positional argument(s), got ' + got,
      loc,
    });
    return T.failed(op + ' arity');
  }
  function argError(op, i, expected, got, loc) {
    diagnostics.push({
      severity: 'error',
      message: op + ': arg ' + (i + 1) + ' expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' arg type');
  }
  function kwargError(op, k, expected, got, loc) {
    diagnostics.push({
      severity: 'error',
      message: op + ': kwarg "' + k + '" expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' kwarg type');
  }

  function setMarker(name) {
    // Internal-only "set" marker, not a user-visible type. elementof
    // recognises it by `kind === 'set'`. Anywhere else gets a deferred
    // back from isValue() which falls back to deferred handling.
    return { kind: 'set', name };
  }
}

module.exports = { inferTypes };
