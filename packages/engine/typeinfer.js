'use strict';

// FlatPIR structural type inference.
//
// Operates on the LoweredModule produced by `pir.lowerToModule(...)`.
// Walks each binding's lowered RHS, infers a type for every call,
// writes per-call meta annotations (FlatPIR `(%meta type phase)`,
// type slot only — phase is in phaseinfer.js), and sets
// `binding.inferredType` to the type of the binding's outermost
// expression for fast lookup.
//
// Diagnostics are collected into a flat array compatible with the
// analyzer's existing diagnostic stream (same {severity, message,
// loc} shape) so they merge cleanly into the editor.
//
// Why on lowered IR
// =================
// The source AST has many node kinds (BinaryExpr, UnaryExpr,
// ArrayLiteral, TupleLiteral, FieldAccess, etc.). FlatPIR collapses
// them all to calls — `add`, `mul`, `vector`, `tuple`, `get_field`,
// etc. The inference pass over the IR is therefore one switch on
// {lit, const, ref, hole, call}, with the call case dispatching by
// op name. Cleaner; fewer special cases.
//
// Polymorphism
// ============
// Built-in signatures use type variables (`weighted: (real,
// measure<T>) → measure<T>`); types.js's unify handles them.
//
// User-defined function/kernel signatures carry their result type
// directly (computed at definition time by inferring the body in the
// scope where parameters take their declared types). For now we
// don't recompute the body's type per call site — that polymorphic
// flow is in the FlatPIR spec but unused in practice for the
// visualizer's current scope. Added when needed.
//
// Scopes
// ======
// `functionof(body, kw=...)` and `kernelof(body, kw=...)` introduce
// an inner `%local` scope. Inside their bodies, parameter refs are
// `(%ref %local <name>)`. The inference pass tracks an active scope
// stack: a Map<paramName, type> for each enclosing reified callable.
// %local refs resolve against this stack; %self refs against the
// module's binding map.

const T = require('./types');
const builtins = require('./builtins');

// =====================================================================
// Constant maps (carried over from the AST-based version)
// =====================================================================

const CONST_TYPES = {
  pi:    T.REAL,
  inf:   T.REAL,
  im:    T.COMPLEX,
  true:  T.BOOLEAN,
  false: T.BOOLEAN,
};

const SET_VALUE_TYPES = {
  reals: T.REAL, posreals: T.REAL, nonnegreals: T.REAL, unitinterval: T.REAL,
  integers: T.INTEGER, posintegers: T.INTEGER, nonnegintegers: T.INTEGER,
  booleans: T.BOOLEAN,
  complexes: T.COMPLEX,
  rngstates: T.any(),
  anything: T.any(),
};

// =====================================================================
// Public entry
// =====================================================================

/**
 * Run type inference over a LoweredModule. Mutates each binding to
 * set `binding.inferredType` and writes per-call `meta.type`
 * annotations. Returns diagnostics for type mismatches; loc fields
 * point at the source AST positions captured during lowering.
 */
function inferTypes(loweredModule) {
  const diagnostics = [];
  const visiting = new Set();
  const visited  = new Set();

  for (const [name] of loweredModule.bindings) inferBinding(name);
  return diagnostics;

  function inferBinding(name) {
    const b = loweredModule.bindings.get(name);
    if (!b)                   return T.failed('unknown binding "' + name + '"');
    if (visited.has(name))    return b.inferredType || T.deferred();
    if (visiting.has(name)) {
      const t = T.failed('cyclic type inference at "' + name + '"');
      b.inferredType = t;
      return t;
    }
    visiting.add(name);
    const t = inferExpr(b.rhs, []);   // [] = no enclosing scopes
    visiting.delete(name);
    visited.add(name);
    b.inferredType = t;
    return t;
  }

  // -------------------------------------------------------------------
  // Expression-level inference
  // -------------------------------------------------------------------
  // `scopes` is a stack of Map<paramName, type> for each enclosing
  // functionof/kernelof. Top of stack is the innermost scope.

  function inferExpr(expr, scopes) {
    if (!expr) return T.failed('null expression');
    switch (expr.kind) {
      case 'lit':   return inferLit(expr);
      case 'const': return inferConst(expr);
      case 'ref':   return inferRef(expr, scopes);
      case 'hole':  return T.any();   // bound positionally inside fn(...)
      case 'call':  return inferCall(expr, scopes);
    }
    return T.deferred();
  }

  function inferLit(expr) {
    const v = expr.value;
    if (typeof v === 'number') {
      // Lower.js preserves the lexical-form distinction in `numType`
      // (integer literals have no decimal/exponent in source).
      // Fall back to runtime check for synthesized lits without it.
      if (expr.numType === 'integer') return T.INTEGER;
      if (expr.numType === 'real')    return T.REAL;
      return Number.isInteger(v) ? T.INTEGER : T.REAL;
    }
    if (typeof v === 'boolean') return T.BOOLEAN;
    if (typeof v === 'string')  return T.STRING;
    return T.deferred();
  }

  function inferConst(expr) {
    if (CONST_TYPES[expr.name])  return CONST_TYPES[expr.name];
    if (builtins.isSet(expr.name)) return setMarker(expr.name);
    return T.any();
  }

  function inferRef(expr, scopes) {
    if (expr.ns === '%local') {
      // Look up in the scope stack from innermost outward.
      for (let i = scopes.length - 1; i >= 0; i--) {
        if (scopes[i].has(expr.name)) return scopes[i].get(expr.name);
      }
      return T.failed('unbound %local "' + expr.name + '"');
    }
    if (expr.ns === 'self') {
      if (loweredModule.bindings.has(expr.name)) return inferBinding(expr.name);
      // Some surface idents (constants, set names) lower as refs
      // rather than const — handle that gracefully here too.
      if (CONST_TYPES[expr.name])    return CONST_TYPES[expr.name];
      if (builtins.isSet(expr.name)) return setMarker(expr.name);
      return T.failed('undefined name "' + expr.name + '"');
    }
    // Cross-module ref — not yet implemented.
    return T.deferred();
  }

  function inferCall(expr, scopes) {
    // User-defined call: lower.js puts the callee on `target`.
    if (expr.target) return inferUserCall(expr, scopes);

    // Special-cased ops whose result type depends on actuals or
    // structural shape in ways that don't fit the static signature
    // table.
    switch (expr.op) {
      case 'elementof': return write(inferElementof(expr, scopes), expr);
      case 'lawof':     return write(inferLawof(expr, scopes), expr);
      case 'record':    return write(inferRecord(expr, scopes), expr);
      case 'joint':     return write(inferJoint(expr, scopes), expr);
      case 'tuple':     return write(inferTuple(expr, scopes), expr);
      case 'vector':    return write(inferVector(expr, scopes), expr);
      // kernelof and fn are lowered to functionof by lower.js (per
      // spec §sec:kernelof line 421-422 and §sec:fn line 618-628),
      // so we only see functionof here.
      case 'functionof': return write(inferReification(expr, scopes), expr);
    }

    return write(inferGenericCall(expr, scopes), expr);
  }

  // Helper to attach inferred type to the call's meta slot AND
  // return the type. setMeta is from pir.js but we don't import to
  // keep this module standalone — direct write is fine.
  function write(t, expr) {
    if (!expr.meta) expr.meta = {};
    expr.meta.type = t;
    return t;
  }

  // -------------------------------------------------------------------
  // Generic call inference: signature lookup + arg unify
  // -------------------------------------------------------------------

  function inferGenericCall(expr, scopes) {
    const op = expr.op;
    const sig = T.signatureOf(op);
    if (!sig) return T.deferred();

    let s = new Map();
    const args   = expr.args   || [];
    const kwargs = expr.kwargs || {};

    if (sig.args !== null) {
      const rawN = sig.args.length;
      const got  = args.length;
      const variadic = sig.variadic === 'positional';
      const fixedN = variadic ? rawN - 1 : rawN;
      if (variadic) {
        if (got < fixedN) return arityError(op, '≥' + fixedN, got, expr.loc);
      } else if (got !== rawN) {
        return arityError(op, rawN, got, expr.loc);
      }
      for (let i = 0; i < fixedN; i++) {
        const at = inferExpr(args[i], scopes);
        const next = T.unify(sig.args[i], at, s);
        if (next == null) return argError(op, i, sig.args[i], at, args[i].loc);
        s = next;
      }
      if (variadic) {
        const tail = sig.args[rawN - 1];
        for (let i = fixedN; i < got; i++) {
          const at = inferExpr(args[i], scopes);
          const next = T.unify(tail, at, s);
          if (next == null) return argError(op, i, tail, at, args[i].loc);
          s = next;
        }
      }
    }

    for (const k in sig.kwargs) {
      if (!(k in kwargs)) continue;   // optional/defaulted kwargs allowed missing
      const at = inferExpr(kwargs[k], scopes);
      const next = T.unify(sig.kwargs[k], at, s);
      if (next == null) return kwargError(op, k, sig.kwargs[k], at, kwargs[k].loc);
      s = next;
    }
    return T.substitute(sig.result, s);
  }

  // -------------------------------------------------------------------
  // Special-case op handlers
  // -------------------------------------------------------------------

  function inferElementof(expr, scopes) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError('elementof', 1, args.length, expr.loc);
    const t = setValueType(args[0], scopes);
    if (t == null) {
      const argT = inferExpr(args[0], scopes);
      if (argT && argT.kind === 'failed') return T.failed('elementof cascade');
      diagnostics.push({
        severity: 'error',
        message: 'elementof expects a set or set-constructor expression; got ' + T.show(argT),
        loc: args[0].loc,
      });
      return T.failed('elementof bad arg');
    }
    return t;
  }

  function inferLawof(expr, scopes) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError('lawof', 1, args.length, expr.loc);
    const at = inferExpr(args[0], scopes);
    if (at && at.kind === 'failed') return T.failed('lawof cascade');
    if (T.isMeasure(at)) return at;             // identity law: lawof(measure) = measure
    if (T.isValue(at))   return T.measure(at);
    diagnostics.push({
      severity: 'error',
      message: 'lawof expects a value-typed argument, got ' + T.show(at),
      loc: args[0].loc,
    });
    return T.failed('lawof bad arg');
  }

  function inferRecord(expr, scopes) {
    // record uses `fields` (ordered), not `kwargs`.
    const fields = expr.fields || [];
    const out = {};
    for (const f of fields) out[f.name] = inferExpr(f.value, scopes);
    return T.record(out);
  }

  function inferJoint(expr, scopes) {
    const fields = expr.fields || [];
    const out = {};
    for (const f of fields) {
      const at = inferExpr(f.value, scopes);
      if (T.isMeasure(at)) out[f.name] = at.domain;
      else if (at.kind === 'deferred' || at.kind === 'any') out[f.name] = T.deferred();
      else if (at.kind === 'failed') return T.failed('joint cascade');
      else {
        diagnostics.push({
          severity: 'error',
          message: 'joint kwarg "' + f.name + '" expects a measure, got ' + T.show(at),
          loc: f.value.loc || expr.loc,
        });
        return T.failed('joint bad kwarg');
      }
    }
    return T.measure(T.record(out));
  }

  function inferTuple(expr, scopes) {
    const args = expr.args || [];
    return T.tuple(args.map(a => inferExpr(a, scopes)));
  }

  function inferVector(expr, scopes) {
    // `(call vector e1 e2 …)` — the array's length is the number of
    // arguments (statically known); the element type is the unifying
    // type of the elements. Empty vectors get an %any element type.
    const args = expr.args || [];
    if (args.length === 0) return T.array(1, [0], T.any());
    const elemTypes = args.map(a => inferExpr(a, scopes));
    let s = new Map();
    let elem = elemTypes[0];
    for (let i = 1; i < elemTypes.length; i++) {
      const next = T.unify(elem, elemTypes[i], s);
      if (next == null) {
        diagnostics.push({
          severity: 'error',
          message: 'array element type mismatch: '
            + T.show(elem) + ' vs ' + T.show(elemTypes[i]),
          loc: args[i].loc || expr.loc,
        });
        return T.failed('array element mismatch');
      }
      s = next;
      elem = T.substitute(elem, s);
    }
    return T.array(1, [args.length], T.substitute(elem, s));
  }

  // -------------------------------------------------------------------
  // Reification: functionof / kernelof / fn
  // -------------------------------------------------------------------
  //
  // Per spec §sec:functionof and §sec:kernelof:
  //   * functionof(body, kw=...) reifies body into a callable. If body
  //     is value-typed, the result is a function; if measure-typed, a
  //     kernel.
  //   * kernelof(body, kw=...) ≡ functionof(lawof(body), kw=...) —
  //     always produces a kernel; the body must be value-typed.
  //   * fn(body) lowers to functionof with placeholder parameters
  //     extracted from the body's holes.
  //
  // The function's parameters carry the type of their boundary. For a
  // placeholder boundary (`par = _par_`), the parameter's type is %any
  // (the placeholder is `elementof(anything)` per spec). For an
  // elementof-bound boundary (`par = _some_elementof`), it's the value
  // type of that elementof's set. For a stochastic-bound boundary
  // (`theta1 = theta1`), the parameter type is the boundary expression's
  // structural type — the spec says boundaries are substituted with
  // `elementof(valueset(boundary))` whose value type follows the
  // boundary's domain.

  function inferReification(expr, scopes) {
    // Only `functionof` reaches here — kernelof and fn are lowered
    // to functionof by lower.js. The kernelof spec rule "x must not
    // be a measure" emerges naturally from the lawof inside the
    // lowered form (lawof requires a value-typed argument); we don't
    // need a special case here.
    const params      = expr.params      || [];   // scope-local names
    const paramKwargs = expr.paramKwargs || [];   // surface kwarg names
    const newScope = new Map();
    for (let i = 0; i < params.length; i++) {
      let paramType = T.any();
      const kwName = paramKwargs[i];
      if (kwName && expr.kwargs && expr.kwargs[kwName]) {
        const boundaryT = inferExpr(expr.kwargs[kwName], scopes);
        if (T.isMeasure(boundaryT)) {
          diagnostics.push({
            severity: 'error',
            message: 'functionof boundary "' + kwName
              + '" must be a value, got ' + T.show(boundaryT),
            loc: expr.kwargs[kwName].loc || expr.loc,
          });
          paramType = T.failed('functionof boundary type');
        } else if (T.isValue(boundaryT)) {
          paramType = boundaryT;
        }
      }
      newScope.set(params[i], paramType);
    }

    const innerScopes = scopes.concat([newScope]);
    const bodyT = expr.body ? inferExpr(expr.body, innerScopes) : T.deferred();
    // Inputs use the *surface* keyword name from paramKwargs — that's
    // what call-site kwargs bind to. Types come from the scope.
    const inputs = params.map((p, i) => ({
      name: paramKwargs[i] || p,
      type: newScope.get(p),
    }));

    // Per spec §sec:functionof-measure: a functionof with a measure
    // body produces a kernel; with a value body, a function.
    if (T.isMeasure(bodyT))      return T.kernelType(inputs, bodyT);
    if (T.isValue(bodyT))        return T.funcType(inputs,   bodyT);
    if (bodyT.kind === 'failed') return T.failed('functionof cascade');
    return T.deferred();
  }

  // -------------------------------------------------------------------
  // User-defined call: callee is a (%ref self <fn-name>)
  // -------------------------------------------------------------------

  function inferUserCall(expr, scopes) {
    const head = expr.target;
    if (!head || head.ns !== 'self') {
      // Cross-module user calls — not yet implemented.
      return write(T.deferred(), expr);
    }
    const calleeType = inferBinding(head.name);
    if (!T.isCallable(calleeType)) {
      // Cascade silently when the callee already failed or is still
      // deferred (couldn't infer its type — e.g. unknown built-in,
      // standard module function not yet typed). Only error when we
      // positively know it's a non-callable (scalar / measure / etc.).
      if (calleeType && (calleeType.kind === 'failed' || calleeType.kind === 'deferred'
                         || calleeType.kind === 'any')) {
        return write(T.deferred(), expr);
      }
      diagnostics.push({
        severity: 'error',
        message: '"' + head.name + '" is not callable (got ' + T.show(calleeType) + ')',
        loc: expr.loc,
      });
      return write(T.failed('not callable'), expr);
    }

    // For now: take the callee's `result` directly. This is the
    // "monomorphic-at-definition" simplification. Once we add full
    // polymorphism, we'd traverse the callee's body with the call
    // site's actual argument types.
    //
    // We DO type-check the call args against the callee's input
    // types — that catches passing wrong-typed values to functions.
    const inputs = calleeType.inputs;
    const args   = expr.args   || [];
    const kwargs = expr.kwargs || {};

    // Positional first, then keyword. Spec allows both calling
    // conventions for user-defined callables with explicit boundaries.
    for (let i = 0; i < inputs.length; i++) {
      const inp = inputs[i];
      let actual = null, actualLoc = expr.loc;
      if (i < args.length) {
        actual = inferExpr(args[i], scopes);
        actualLoc = args[i].loc;
      } else if (inp.name in kwargs) {
        actual = inferExpr(kwargs[inp.name], scopes);
        actualLoc = kwargs[inp.name].loc;
      } else {
        // Missing argument. Diagnostic but don't bail — the result
        // type doesn't depend on which inputs were supplied (we use
        // the function's declared result).
        diagnostics.push({
          severity: 'error',
          message: 'call to "' + head.name + '" missing argument "' + inp.name + '"',
          loc: expr.loc,
        });
        continue;
      }
      if (actual && actual.kind !== 'failed') {
        const s = T.unify(inp.type, actual, new Map());
        if (s == null) {
          diagnostics.push({
            severity: 'error',
            message: head.name + ': arg "' + inp.name + '" expects ' + T.show(inp.type)
              + ', got ' + T.show(actual),
            loc: actualLoc,
          });
        }
      }
    }
    return write(calleeType.result, expr);
  }

  // -------------------------------------------------------------------
  // Set-expression value-type resolution (used by elementof)
  // -------------------------------------------------------------------

  function setValueType(expr, scopes) {
    if (!expr) return null;
    if (expr.kind === 'const' && SET_VALUE_TYPES[expr.name] !== undefined) {
      return SET_VALUE_TYPES[expr.name];
    }
    if (expr.kind === 'ref' && expr.ns === 'self' && SET_VALUE_TYPES[expr.name] !== undefined) {
      return SET_VALUE_TYPES[expr.name];
    }
    if (expr.kind !== 'call') return null;
    switch (expr.op) {
      case 'interval':   return T.REAL;
      case 'stdsimplex': {
        const n = expr.args && expr.args[0] && expr.args[0].kind === 'lit'
          && Number.isInteger(expr.args[0].value) ? expr.args[0].value : '%dynamic';
        return T.array(1, [n], T.REAL);
      }
      case 'cartpow': {
        const inner = setValueType(expr.args[0], scopes);
        if (inner == null) return null;
        const dims = (expr.args || []).slice(1).map(a =>
          (a.kind === 'lit' && Number.isInteger(a.value)) ? a.value : '%dynamic');
        return T.array(dims.length, dims, inner);
      }
      case 'cartprod': {
        const fields = expr.fields || null;
        if (fields && fields.length > 0) {
          const out = {};
          for (const f of fields) {
            const t = setValueType(f.value, scopes);
            if (t == null) return null;
            out[f.name] = t;
          }
          return T.record(out);
        }
        const elems = (expr.args || []).map(a => setValueType(a, scopes));
        if (elems.some(e => e == null)) return null;
        return elems.length === 1 ? elems[0] : T.tuple(elems);
      }
    }
    return null;
  }

  // -------------------------------------------------------------------
  // Diagnostics helpers (suppress cascades when inputs already failed)
  // -------------------------------------------------------------------

  function arityError(op, expected, got, loc) {
    diagnostics.push({
      severity: 'error',
      message: op + ' expects ' + expected + ' positional argument(s), got ' + got,
      loc,
    });
    return T.failed(op + ' arity');
  }
  function argError(op, i, expected, got, loc) {
    if (got && got.kind === 'failed') return T.failed(op + ' arg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': arg ' + (i + 1) + ' expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' arg type');
  }
  function kwargError(op, k, expected, got, loc) {
    if (got && got.kind === 'failed') return T.failed(op + ' kwarg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': kwarg "' + k + '" expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' kwarg type');
  }
}

// Internal "set" marker — not a user-facing type. elementof handles it.
function setMarker(name) { return { kind: 'set', name }; }

module.exports = { inferTypes };
