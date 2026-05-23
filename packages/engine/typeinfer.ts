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

import type { IRNode } from './engine-types';

const T = require('./types.ts');
const builtins = require('./builtins.ts');

// =====================================================================
// Constant maps (carried over from the AST-based version)
// =====================================================================

const CONST_TYPES: Record<string, any> = {
  pi:    T.REAL,
  inf:   T.REAL,
  im:    T.COMPLEX,
  true:  T.BOOLEAN,
  false: T.BOOLEAN,
};

const SET_VALUE_TYPES: Record<string, any> = {
  reals: T.REAL, posreals: T.REAL, nonnegreals: T.REAL, unitinterval: T.REAL,
  integers: T.INTEGER, posintegers: T.INTEGER, nonnegintegers: T.INTEGER,
  booleans: T.BOOLEAN,
  complexes: T.COMPLEX,
  rngstates: T.any(),
  anything: T.any(),
};

// Op classifications for shape-polymorphic inference. Binary ops
// take two numeric operands and return a numeric of the broadcast
// shape (scalar/scalar → scalar; scalar+array → array; array+array
// of matching shape → array). Unary ops take one numeric and
// return the same shape. Comparisons take two numerics and return
// boolean of the broadcast shape (so vec_a < vec_b yields
// array<bool>; scalar_a < vec_b yields array<bool> too).
const BINARY_ARITH_OPS = new Set(['add', 'sub', 'mul', 'div', 'mod', 'pow']);
const UNARY_ARITH_OPS  = new Set([
  'neg', 'pos', 'abs', 'abs2', 'exp', 'log', 'log10', 'sqrt',
  'sin', 'cos', 'floor', 'ceil', 'round',
]);
const COMPARISON_OPS = new Set(['lt', 'le', 'gt', 'ge', 'equal', 'unequal']);

// =====================================================================
// Public entry
// =====================================================================

/**
 * Run type inference over a LoweredModule. Mutates each binding to
 * set `binding.inferredType` and writes per-call `meta.type`
 * annotations. Returns diagnostics for type mismatches; loc fields
 * point at the source AST positions captured during lowering.
 */
function inferTypes(loweredModule: any) {
  const ctx = createInferenceContext(loweredModule);
  for (const [name] of loweredModule.bindings) ctx.inferBinding(name);
  return ctx.diagnostics;
}

/**
 * On-demand inference at a synthetic call site: given a body
 * expression and a scope binding param names to concrete input
 * types, return the inferred result type. Used for plot-time
 * specialization of polymorphic functions — the module-level
 * inference produces a best-effort type with `any` inputs, but
 * when we plot we have specific input types and can specialize
 * exactly, the same way a real call site would.
 *
 * Assumes the module has already been processed by inferTypes
 * (each binding has its inferredType set). Lazy inferBinding
 * still works if not — it'll walk on demand.
 *
 * @param loweredModule  the module produced by `lower(ast)`
 * @param expr           an IR expression (typically the body of a
 *                        functionof / kernelof)
 * @param paramTypes     Map<paramName, type>; each %local-ref
 *                        inside `expr` resolves through this.
 * @returns inferred type of `expr` in that scope
 */
function inferExprInScope(loweredModule: any, expr: IRNode, paramTypes: any) {
  const ctx = createInferenceContext(loweredModule);
  const scopes = paramTypes ? [paramTypes] : [];
  return ctx.inferExpr(expr, scopes);
}

/**
 * Build a fresh inference context bound to a LoweredModule. Returns
 * the `diagnostics` accumulator and the same `inferBinding` /
 * `inferExpr` helpers that drive the module-level pass — exposing
 * them lets external callers run inference on synthetic
 * expressions (call sites, sub-bodies) reusing exactly the same
 * rules. Cycle detection (visiting/visited) is per-context, so
 * separate contexts don't interfere.
 */
function createInferenceContext(loweredModule: any) {
  const diagnostics: any[] = [];
  const visiting = new Set();
  const visited  = new Set();

  function inferBinding(name: any): any {
    const b = loweredModule.bindings.get(name);
    if (!b)                   return T.failed('unknown binding "' + name + '"');
    if (visited.has(name))    return b.inferredType || T.deferred();
    if (visiting.has(name)) {
      const t = T.failed('cyclic type inference at "' + name + '"');
      b.inferredType = t;
      return t;
    }
    visiting.add(name);
    const t: any = inferExpr(b.rhs, []);   // [] = no enclosing scopes
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

  function inferExpr(expr: any, scopes: any): any {
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

  function inferLit(expr: any) {
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

  function inferConst(expr: any) {
    if (CONST_TYPES[expr.name])  return CONST_TYPES[expr.name];
    if (builtins.isSet(expr.name)) return setMarker(expr.name);
    return T.any();
  }

  function inferRef(expr: any, scopes: any): any {
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

  function inferCall(expr: any, scopes: any): any {
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
      // tuple_get(<tuple-expr>, <slot lit>) — internal IR op emitted by
      // pir.lowerToModule for multi-LHS bindings (`a, b = rand(...)`).
      // Result type is the tuple's i-th element. Special-cased because
      // the result depends on the literal slot value, which the generic
      // signature table can't express.
      case 'tuple_get': return write(inferTupleGet(expr, scopes), expr);
      // get_field(<record-expr>, <name lit>) — record field access
      // lowered from surface `obj.field`. Same kind of special case as
      // tuple_get: the result type depends on the literal field-name
      // argument.
      case 'get_field': return write(inferGetField(expr, scopes), expr);
      // Lebesgue / Counting parametrise on a support set (spec §06).
      // The measure's domain mirrors the support's value-type, so
      // Lebesgue(support = cartpow(reals, n)) is measure(array<real,n>).
      case 'Lebesgue':  return write(inferReferenceMeasure(expr, scopes, T.REAL), expr);
      case 'Counting':  return write(inferReferenceMeasure(expr, scopes, T.INTEGER), expr);
      case 'vector':    return write(inferVector(expr, scopes), expr);
      case 'iid':       return write(inferIid(expr, scopes), expr);
      // kernelof and fn are lowered to functionof by lower.js (per
      // spec §sec:kernelof line 421-422 and §sec:fn line 618-628),
      // so we only see functionof here.
      case 'functionof': return write(inferReification(expr, scopes), expr);
      // transpose / adjoint: spec §07 — apply to vectors and matrices.
      // For vectors, return type is transposed_vector (the new spec
      // type from flatppl-design 244b0e5); for transposed_vector,
      // return type is vector (involution); for matrices, swap dims.
      case 'transpose': return write(inferTransposeAdjoint(expr, scopes), expr);
      case 'adjoint':   return write(inferTransposeAdjoint(expr, scopes), expr);
    }
    // Numeric arithmetic with shape polymorphism: both scalars,
    // both arrays of matching shape, or scalar/array broadcast.
    if (BINARY_ARITH_OPS.has(expr.op)) return write(inferArith2(expr, scopes), expr);
    if (UNARY_ARITH_OPS.has(expr.op))  return write(inferArith1(expr, scopes), expr);
    if (COMPARISON_OPS.has(expr.op))   return write(inferComparison(expr, scopes), expr);

    return write(inferGenericCall(expr, scopes), expr);
  }

  // Helper to attach inferred type to the call's meta slot AND
  // return the type. setMeta is from pir.js but we don't import to
  // keep this module standalone — direct write is fine.
  function write(t: any, expr: any) {
    if (!expr.meta) expr.meta = {};
    expr.meta.type = t;
    return t;
  }

  // -------------------------------------------------------------------
  // Generic call inference: signature lookup + arg unify
  // -------------------------------------------------------------------

  function inferGenericCall(expr: any, scopes: any): any {
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
        const at: any = inferExpr(args[i], scopes);
        const next = T.unify(sig.args[i], at, s);
        if (next == null) return argError(op, i, sig.args[i], at, args[i].loc);
        s = next;
      }
      if (variadic) {
        const tail = sig.args[rawN - 1];
        for (let i = fixedN; i < got; i++) {
          const at: any = inferExpr(args[i], scopes);
          const next = T.unify(tail, at, s);
          if (next == null) return argError(op, i, tail, at, args[i].loc);
          s = next;
        }
      }
    }

    for (const k in sig.kwargs) {
      if (!(k in kwargs)) continue;   // optional/defaulted kwargs allowed missing
      const at: any = inferExpr(kwargs[k], scopes);
      const next = T.unify(sig.kwargs[k], at, s);
      if (next == null) return kwargError(op, k, sig.kwargs[k], at, kwargs[k].loc);
      s = next;
    }
    return T.substitute(sig.result, s);
  }

  // -------------------------------------------------------------------
  // Special-case op handlers
  // -------------------------------------------------------------------

  // transpose / adjoint dispatch on input shape per spec §07:
  //   vector(n)            → transposed_vector(n)
  //   transposed_vector(n) → vector(n)             (involution)
  //   matrix(m, n)         → matrix(n, m)          (dim swap)
  //   matrix(N, m, n)      → matrix(N, n, m)       (atom-batched per-slice;
  //                          rank-3 not currently expressible in array type
  //                          system but documented here for future extension)
  //   scalar               → error (transpose undefined on scalars)
  //   other                → deferred (unknown shape — let runtime handle)
  function inferTransposeAdjoint(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 1) return arityError(expr.op, 1, args.length, expr.loc);
    const at: any = inferExpr(args[0], scopes);
    if (!at) return T.deferred();
    switch (at.kind) {
      case 'array':
        if (at.rank === 1) {
          // vector(n) → transposed_vector(n)
          return T.tvector(at.shape[0], at.elem);
        }
        if (at.rank === 2) {
          // matrix(m, n) → matrix(n, m). Swap shape entries; elem unchanged.
          return T.array(2, [at.shape[1], at.shape[0]], at.elem);
        }
        // rank ≥ 3 — spec is silent; runtime supports swap-last-two-axes.
        // Pass through unchanged at the type level.
        return at;
      case 'tvector':
        // transposed_vector(n) → vector(n)
        return T.array(1, [at.length], at.elem);
      case 'scalar':
        return T.failed(expr.op + ': not defined on scalars');
      case 'deferred':
      case 'any':
      case 'var':
        return T.deferred();
    }
    return T.deferred();
  }

  function inferElementof(expr: any, scopes: any) {
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

  function inferLawof(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 1) return arityError('lawof', 1, args.length, expr.loc);
    const at: any = inferExpr(args[0], scopes);
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

  function inferRecord(expr: any, scopes: any) {
    // record uses `fields` (ordered), not `kwargs`.
    const fields = expr.fields || [];
    const out: Record<string, any> = {};
    for (const f of fields) out[f.name] = inferExpr(f.value, scopes);
    return T.record(out);
  }

  function inferJoint(expr: any, scopes: any) {
    const fields = expr.fields || [];
    const out: Record<string, any> = {};
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

  function inferTuple(expr: any, scopes: any) {
    const args = expr.args || [];
    return T.tuple(args.map((a: any) => inferExpr(a, scopes)));
  }

  // Lebesgue(support = S) / Counting(support = S). The support kwarg is
  // optional; missing → use the default scalar. When present, the support
  // is a set expression whose value-type drives the result's measure-domain
  // (cartpow → array, cartprod → record/tuple, stdsimplex → array<real,n>).
  // Falls back to the default scalar when the support shape can't be
  // statically resolved (e.g. the support is itself a binding ref).
  function inferReferenceMeasure(expr: any, scopes: any, defaultElem: any) {
    const args   = expr.args || [];
    const kwargs = expr.kwargs || {};
    let support = null;
    // Spec §06 form is `Lebesgue(support = S)`, but keep accepting a
    // single positional support per the calling-convention rule.
    if ('support' in kwargs)        support = kwargs.support;
    else if (args.length === 1)     support = args[0];
    if (!support) return T.measure(defaultElem);
    const t = setValueType(support, scopes);
    return T.measure(t || defaultElem);
  }

  function inferGetField(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) {
      return arityError('get_field', 2, args.length, expr.loc);
    }
    const recT: any = inferExpr(args[0], scopes);
    if (recT && recT.kind === 'failed') return T.failed('get_field cascade');
    if (!recT || recT.kind !== 'record' || !recT.fields) {
      diagnostics.push({
        severity: 'error',
        message: 'get_field expects a record-typed expression; got ' + T.show(recT),
        loc: args[0].loc || expr.loc,
      });
      return T.failed('get_field bad arg');
    }
    const nameIR = args[1];
    if (!nameIR || nameIR.kind !== 'lit' || typeof nameIR.value !== 'string') {
      return T.failed('get_field name must be a literal string');
    }
    if (!(nameIR.value in recT.fields)) {
      diagnostics.push({
        severity: 'error',
        message: `get_field: '${nameIR.value}' is not a field of ${T.show(recT)}`,
        loc: expr.loc,
      });
      return T.failed('get_field unknown field');
    }
    return recT.fields[nameIR.value];
  }

  function inferTupleGet(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) {
      return arityError('tuple_get', 2, args.length, expr.loc);
    }
    const tupleT: any = inferExpr(args[0], scopes);
    if (tupleT && tupleT.kind === 'failed') return T.failed('tuple_get cascade');
    if (!tupleT || tupleT.kind !== 'tuple') {
      diagnostics.push({
        severity: 'error',
        message: 'tuple_get expects a tuple-typed expression; got ' + T.show(tupleT),
        loc: args[0].loc || expr.loc,
      });
      return T.failed('tuple_get bad arg');
    }
    // Slot must be a literal int — the lowering pass always emits one.
    const slotIR = args[1];
    if (!slotIR || slotIR.kind !== 'lit' || typeof slotIR.value !== 'number') {
      return T.failed('tuple_get slot must be a literal index');
    }
    const i = slotIR.value | 0;
    if (i < 0 || i >= tupleT.elems.length) {
      diagnostics.push({
        severity: 'error',
        message: `tuple_get index ${i} out of range for ${T.show(tupleT)}`,
        loc: expr.loc,
      });
      return T.failed('tuple_get out of range');
    }
    return tupleT.elems[i];
  }

  function inferVector(expr: any, scopes: any) {
    // `(call vector e1 e2 …)` — the array's length is the number of
    // arguments (statically known); the element type is the unifying
    // type of the elements. Empty vectors get an %any element type.
    const args = expr.args || [];
    if (args.length === 0) return T.array(1, [0], T.any());
    const elemTypes = args.map((a: any) => inferExpr(a, scopes));
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

  // ---- Numeric arithmetic with shape polymorphism -------------------
  //
  // Binary: scalar+scalar → scalar; matching-shape arrays elementwise;
  // scalar/array broadcast. Unary: shape-preserving. Comparisons
  // produce boolean of the broadcast shape.

  function inferArith2(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) return arityError(expr.op, 2, args.length, expr.loc);
    const aT: any = inferExpr(args[0], scopes);
    const bT: any = inferExpr(args[1], scopes);
    if (aT.kind === 'failed' || bT.kind === 'failed') return T.failed(expr.op + ' cascade');
    const r: any = T.unifyArith(aT, bT, new Map());
    if (r == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand types ' + T.show(aT) + ' and '
          + T.show(bT) + ' are not numerically compatible',
        loc: expr.loc,
      });
      return T.failed(expr.op + ' shape mismatch');
    }
    return r.result;
  }

  function inferArith1(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError(expr.op, 1, args.length, expr.loc);
    const aT = inferExpr(args[0], scopes);
    if (aT.kind === 'failed') return T.failed(expr.op + ' cascade');
    // Scalar in → scalar out; array in → array out (shape preserved).
    // floor/ceil/round produce integer scalars from real input;
    // otherwise the result type matches the input. Other unary maths
    // are real → real for now (spec doesn't require richer overloads).
    const isIntCast = (expr.op === 'floor' || expr.op === 'ceil' || expr.op === 'round');
    function liftElemwise(t: any): any {
      if (t.kind === 'scalar')   return isIntCast ? T.INTEGER : T.REAL;
      if (t.kind === 'array')    return T.array(t.rank, t.shape.slice(),
                                                 liftElemwise(t.elem));
      if (t.kind === 'deferred' || t.kind === 'any') return t;
      return null;
    }
    const out = liftElemwise(aT);
    if (out == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand must be numeric (scalar or array of scalars), got ' + T.show(aT),
        loc: args[0].loc,
      });
      return T.failed(expr.op + ' bad operand');
    }
    return out;
  }

  function inferComparison(expr: any, scopes: any): any {
    // Comparisons unify operand shapes via unifyArith and return
    // boolean of that shape. `equal(scalar, array)` would broadcast;
    // `equal(scalar, scalar)` → boolean.
    const args = expr.args || [];
    if (args.length !== 2) return arityError(expr.op, 2, args.length, expr.loc);
    const aT: any = inferExpr(args[0], scopes);
    const bT: any = inferExpr(args[1], scopes);
    if (aT.kind === 'failed' || bT.kind === 'failed') return T.failed(expr.op + ' cascade');
    const r: any = T.unifyArith(aT, bT, new Map());
    if (r == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand types ' + T.show(aT) + ' and '
          + T.show(bT) + ' are not comparable',
        loc: expr.loc,
      });
      return T.failed(expr.op + ' shape mismatch');
    }
    // Result is boolean with the same shape as r.result.
    if (r.result.kind === 'array') {
      return T.array(r.result.rank, r.result.shape.slice(), T.BOOLEAN);
    }
    return T.BOOLEAN;
  }

  // ---- iid: shape-aware `iid(M, n)` ---------------------------------
  //
  // Per spec §sec:iid: produces measure<array<1, [n], M.domain>> when
  // n is statically known; %dynamic otherwise. We resolve n via
  // literal and binding-ref folding so common cases (`n = 10;
  // iid(M, n)`) yield concrete shapes for downstream shape checks.
  function inferIid(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length < 2) return arityError('iid', '≥2', args.length, expr.loc);
    const measureT: any = inferExpr(args[0], scopes);
    if (!T.isMeasure(measureT)) {
      if (measureT && measureT.kind === 'failed') return T.failed('iid cascade');
      diagnostics.push({
        severity: 'error',
        message: 'iid: arg 1 expects a measure, got ' + T.show(measureT),
        loc: args[0].loc,
      });
      return T.failed('iid bad measure');
    }
    // Resolve dimensions: walk each n arg and try to fold to an
    // integer literal. Refs to integer-typed bindings remain
    // %dynamic for now (full constant folding via resolveConstant
    // could be promoted from orchestrator if we want).
    //
    // New spec: arg 1 may be a single positive int OR a vector of
    // positive ints. The vector-literal form (lowered as `vector(...)`)
    // is unpacked into per-axis dims here so downstream shape checks
    // see concrete dims. A non-literal vector falls through to a
    // single dynamic-rank axis (best we can do statically).
    const dims: any[] = [];
    let dimArgs = args.slice(1);
    if (dimArgs.length === 1
      && dimArgs[0].kind === 'call'
      && dimArgs[0].op === 'vector'
      && Array.isArray(dimArgs[0].args)) {
      dimArgs = dimArgs[0].args;
    }
    for (let i = 0; i < dimArgs.length; i++) {
      const arg = dimArgs[i];
      const dT = inferExpr(arg, scopes);
      // Type-check: each dim must be integer-promotable. Allow an
      // integer-array second arg too (the non-literal vector case).
      const s = T.unify(T.INTEGER, dT, new Map());
      if (s == null) {
        // Permit `iid(M, sizes)` where `sizes` is an integer array.
        if (dimArgs.length === 1 && dT && dT.kind === 'array') {
          dims.push('%dynamic');
          continue;
        }
        diagnostics.push({
          severity: 'error',
          message: 'iid: dim ' + (i + 1) + ' expects integer, got ' + T.show(dT),
          loc: arg.loc,
        });
        return T.failed('iid bad dim');
      }
      dims.push(literalIntFromIR(arg) != null ? literalIntFromIR(arg) : '%dynamic');
    }
    const rank = dims.length;
    return T.measure(T.array(rank, dims, measureT.domain));
  }

  function literalIntFromIR(ir: any) {
    if (!ir) return null;
    if (ir.kind === 'lit' && ir.numType === 'integer') return ir.value;
    if (ir.kind === 'lit' && Number.isInteger(ir.value)) return ir.value;
    return null;
  }

  function inferReification(expr: any, scopes: any): any {
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
    const bodyT: any = expr.body ? inferExpr(expr.body, innerScopes) : T.deferred();
    // Inputs use the *surface* keyword name from paramKwargs — that's
    // what call-site kwargs bind to. Types come from the scope.
    const inputs = params.map((p: any, i: any) => ({
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

  function inferUserCall(expr: any, scopes: any): any {
    const head = expr.target;
    if (!head || head.ns !== 'self') {
      // Cross-module user calls — not yet implemented.
      return write(T.deferred(), expr);
    }
    const calleeType: any = inferBinding(head.name);
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
    let args   = expr.args   || [];
    let kwargs = expr.kwargs || {};

    // Auto-splatting (spec §sec:calling-convention lines 99-102):
    // a single positional record argument is equivalent to passing
    // each field as a kwarg. We detect this when the call has
    // exactly one positional arg of record type whose field names
    // are a subset of the callee's input names; the splat replaces
    // the positional list with a kwarg map for the type-check below.
    // This lets the spec's `f(record(a=x, b=y))` and `f(some_record_
    // value)` typecheck against `f` declared with a/b kwargs.
    if (args.length === 1 && Object.keys(kwargs).length === 0 && inputs.length > 0) {
      const splatType = inferExpr(args[0], scopes);
      if (splatType && splatType.kind === 'record' && splatType.fields) {
        const inputNames = new Set(inputs.map((i: any) => i.name));
        let allMatch = true;
        for (const k in splatType.fields) {
          if (!inputNames.has(k)) { allMatch = false; break; }
        }
        if (allMatch) {
          // Synthesize per-field exprs by typing through `splatType`
          // — no AST rewrite here, just record the per-field types
          // for the unify loop. We re-key the call by field name.
          const splatKwargs: Record<string, any> = {};
          for (const k in splatType.fields) splatKwargs[k] = { __splatType: splatType.fields[k] };
          args = [];
          kwargs = splatKwargs;
        }
      }
    }

    // Positional first, then keyword. Spec allows both calling
    // conventions for user-defined callables with explicit boundaries.
    for (let i = 0; i < inputs.length; i++) {
      const inp = inputs[i];
      let actual: any = null, actualLoc = expr.loc;
      if (i < args.length) {
        actual = inferExpr(args[i], scopes);
        actualLoc = args[i].loc;
      } else if (inp.name in kwargs) {
        const kw = kwargs[inp.name];
        if (kw && kw.__splatType) {
          // Came from auto-splat — type already resolved against the
          // record's field. No AST node to re-infer / locate.
          actual = kw.__splatType;
        } else {
          actual = inferExpr(kw, scopes);
          actualLoc = kw.loc;
        }
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

  function setValueType(expr: any, scopes: any): any {
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
        const inner: any = setValueType(expr.args[0], scopes);
        if (inner == null) return null;
        const dims = (expr.args || []).slice(1).map((a: any) =>
          (a.kind === 'lit' && Number.isInteger(a.value)) ? a.value : '%dynamic');
        return T.array(dims.length, dims, inner);
      }
      case 'cartprod': {
        const fields = expr.fields || null;
        if (fields && fields.length > 0) {
          const out: Record<string, any> = {};
          for (const f of fields) {
            const t = setValueType(f.value, scopes);
            if (t == null) return null;
            out[f.name] = t;
          }
          return T.record(out);
        }
        const elems = (expr.args || []).map((a: any) => setValueType(a, scopes));
        if (elems.some((e: any) => e == null)) return null;
        return elems.length === 1 ? elems[0] : T.tuple(elems);
      }
    }
    return null;
  }

  // -------------------------------------------------------------------
  // Diagnostics helpers (suppress cascades when inputs already failed)
  // -------------------------------------------------------------------

  function arityError(op: any, expected: any, got: any, loc: any) {
    diagnostics.push({
      severity: 'error',
      message: op + ' expects ' + expected + ' positional argument(s), got ' + got,
      loc,
    });
    return T.failed(op + ' arity');
  }
  function argError(op: any, i: any, expected: any, got: any, loc: any) {
    if (got && got.kind === 'failed') return T.failed(op + ' arg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': arg ' + (i + 1) + ' expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' arg type');
  }
  function kwargError(op: any, k: any, expected: any, got: any, loc: any) {
    if (got && got.kind === 'failed') return T.failed(op + ' kwarg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': kwarg "' + k + '" expects ' + T.show(expected)
        + ', got ' + T.show(got),
      loc,
    });
    return T.failed(op + ' kwarg type');
  }

  return { diagnostics, inferBinding, inferExpr };
}

// Internal "set" marker — not a user-facing type. elementof handles it.
function setMarker(name: any) { return { kind: 'set', name }; }

module.exports = { inferTypes, inferExprInScope };
