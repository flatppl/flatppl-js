'use strict';

// =====================================================================
// sampler-eval-compile.ts — fused-loop compiler for evaluateExprN
// =====================================================================
//
// Compiles a scalar-elementwise IR expression to a single JS loop that
// computes one Float64Array(N) in one pass, with NO intermediate
// arrays. Bit-identical to the node-by-node interpreter in
// sampler-eval-batched.ts: each op is the scalar arm of its ARITH_OPS
// entry, in the same nesting/order, on the same scalars
// (`test/fused-loop-scalar-ops.test.ts` checks every op against ARITH_OPS).
//
// Dependencies (evaluateExpr, resolveConst) are injected via
// initCompiler() from sampler-eval-batched.ts's initARITHOPSN — avoids a
// require cycle and keeps those internals unexported.

let _evaluateExpr: any = null;   // single-point evaluator (for folds)
let _resolveConst: any = null;   // const-name resolver

function initCompiler(deps: any): void {
  _evaluateExpr = deps.evaluateExpr;
  _resolveConst = deps.resolveConst;
}

const _valueOps = require('./value-ops.ts');
const _gamma = require('@stdlib/math-base-special-gamma');
const _gammaln = require('@stdlib/math-base-special-gammaln');
const _erfc = require('@stdlib/math-base-special-erfc');
const _erfcinv = require('@stdlib/math-base-special-erfcinv');

// The loop's own scalar ops. Every value in the loop is a real number or a
// boolean: per-atom inputs are real Float64Arrays (`_atomArray`), folds are
// `+c` of a number or boolean (`runPlan`), and each op maps reals to reals.
// So the loop needs only the scalar arm of each ARITH_OPS entry. These are
// separate closures that only ever see scalars, so V8 can inline them;
// ARITH_OPS entries also take Values, arrays, complex and records, and their
// call sites in the loop stayed megamorphic.
const SCALAR_OPS: Record<string, (...a: any[]) => any> = {
  add: (a: any, b: any) => a + b,
  sub: (a: any, b: any) => a - b,
  mul: (a: any, b: any) => a * b,
  div: (a: any, b: any) => Math.floor(a / b),
  divide: (a: any, b: any) => a / b,
  mod: (a: any, b: any) => a - b * Math.floor(a / b),
  neg: (a: any) => -a,
  pos: (a: any) => +a,
  pow: (a: any, b: any) => Math.pow(a, b),
  abs: (a: any) => Math.abs(a),
  abs2: (a: any) => a * a,
  exp: (a: any) => Math.exp(a),
  log: (a: any) => Math.log(a),
  log2: (a: any) => Math.log2(a),
  log10: (a: any) => Math.log10(a),
  log1p: (a: any) => Math.log1p(a),
  expm1: (a: any) => Math.expm1(a),
  sqrt: (a: any) => Math.sqrt(a),
  sin: (a: any) => Math.sin(a),
  cos: (a: any) => Math.cos(a),
  tan: (a: any) => Math.tan(a),
  asin: (a: any) => Math.asin(a),
  acos: (a: any) => Math.acos(a),
  atan: (a: any) => Math.atan(a),
  atan2: (y: any, x: any) => Math.atan2(y, x),
  sinh: (a: any) => Math.sinh(a),
  cosh: (a: any) => Math.cosh(a),
  tanh: (a: any) => Math.tanh(a),
  asinh: (a: any) => Math.asinh(a),
  acosh: (a: any) => Math.acosh(a),
  atanh: (a: any) => Math.atanh(a),
  floor: (a: any) => Math.floor(a),
  ceil: (a: any) => Math.ceil(a),
  round: (a: any) => _valueOps._roundEven(a),
  min: (a: any, b: any) => Math.min(a, b),
  max: (a: any, b: any) => Math.max(a, b),
  gamma: (a: any) => _gamma(a),
  loggamma: (a: any) => _gammaln(a),
  logit: (p: any) => Math.log(p / (1 - p)),
  invlogit: (x: any) => 1 / (1 + Math.exp(-x)),
  probit: (p: any) => -Math.SQRT2 * _erfcinv(2 * p),
  invprobit: (x: any) => 0.5 * _erfc(-x / Math.SQRT2),
  lt: (a: any, b: any) => a < b,
  le: (a: any, b: any) => a <= b,
  gt: (a: any, b: any) => a > b,
  ge: (a: any, b: any) => a >= b,
  equal: (a: any, b: any) => _valueOps._discreteEqual(a, b),
  unequal: (a: any, b: any) => !_valueOps._discreteEqual(a, b),
  isfinite: (a: any) => Number.isFinite(a),
  isinf: (a: any) => !Number.isNaN(a) && !Number.isFinite(a),
  isnan: (a: any) => Number.isNaN(a),
  iszero: (a: any) => a === 0,
  land: (a: any, b: any) => a && b,
  lor: (a: any, b: any) => a || b,
  lxor: (a: any, b: any) => !a !== !b,
  lnot: (a: any) => !a,
  ifelse: (c: any, a: any, b: any) => (c ? a : b),
  boolean: (a: any) => _valueOps._booleanScalar(a),
  integer: (a: any) => _valueOps._integerScalar(a),
  _signed_pow: (y: any, e: any) => _valueOps._signedPowScalar(y, e),
  _image_gate: (y: any, lo: any, hi: any) => _valueOps._imageGateScalar(y, lo, hi),
};

// Compilable scalar ops + arity — the REAL scalar-primitive set, ONE
// source of truth in `ops.ts` (engine-concepts §18). Each has an
// ARITH_OPS scalar entry. Structural (tuple/get_field), aggregate,
// approximation (polynomial/...), and COMPLEX ops are deliberately
// absent (complex isn't in the REAL subset): such a node either folds
// (when atom-independent) or forces a bail.
const _COMPILE_ARITY: Record<string, number> =
  require('./ops.ts').REAL_SCALAR_PRIM_ARITY;

// True if any `ref` node anywhere under `ir` names a per-atom value.
function _hasPerAtomRef(ir: any, perAtomNames: Set<string>): boolean {
  if (!ir || typeof ir !== 'object') return false;
  if (ir.kind === 'ref') return perAtomNames.has(ir.name);
  if (ir.kind !== 'call') return false;
  const args = ir.args || [];
  for (let i = 0; i < args.length; i++) {
    if (_hasPerAtomRef(args[i], perAtomNames)) return true;
  }
  if (ir.kwargs) for (const k in ir.kwargs) {
    if (_hasPerAtomRef(ir.kwargs[k], perAtomNames)) return true;
  }
  if (Array.isArray(ir.fields)) {
    for (let i = 0; i < ir.fields.length; i++) {
      if (_hasPerAtomRef(ir.fields[i] && ir.fields[i].value, perAtomNames)) return true;
    }
  }
  return false;
}

// A compiled plan: a loop fn plus the metadata to bind its inputs.
//   loopFn(OPS, A, C, out, N): OPS = SCALAR_OPS; A = per-atom Float64Array[]; C = folded
//   scalar[]; writes out[i] for i in [0,N).
//   atomRefs[k] is the ref name feeding A[k]; foldIRs[j] is the
//   atom-independent sub-IR feeding C[j] (evaluated once per call).
type Plan = { loopFn: any; atomRefs: string[]; foldIRs: any[] };

const _BAIL = Symbol('bail');

// Recursively emit a JS expression string for `ir`, registering
// per-atom refs into `atomRefs` and folded sub-IRs into `foldIRs`.
// Throws _BAIL when the node is not compilable.
function _emit(ir: any, perAtomNames: Set<string>,
               atomRefs: string[], atomIdx: Map<string, number>,
               foldIRs: any[]): string {
  // Atom-independent subtree → fold to a single constant slot.
  if (!_hasPerAtomRef(ir, perAtomNames)) {
    const j = foldIRs.length;
    foldIRs.push(ir);
    return 'C[' + j + ']';
  }
  if (!ir || typeof ir !== 'object') throw _BAIL;
  if (ir.kind === 'ref') {
    // A per-atom ref (atom-independent refs were folded above).
    if (!perAtomNames.has(ir.name)) throw _BAIL;
    let k = atomIdx.get(ir.name);
    if (k === undefined) { k = atomRefs.length; atomRefs.push(ir.name); atomIdx.set(ir.name, k); }
    return 'A[' + k + '][i]';
  }
  if (ir.kind === 'lit') {
    if (typeof ir.value !== 'number') throw _BAIL;
    return '(' + _numLit(ir.value) + ')';
  }
  if (ir.kind !== 'call') throw _BAIL;
  const op = ir.op;
  const arity = (_COMPILE_ARITY as any)[op];
  if (arity === undefined) throw _BAIL;            // structural / unknown op
  if (ir.kwargs && Object.keys(ir.kwargs).length > 0) throw _BAIL;
  const args = ir.args || [];
  if (args.length !== arity) throw _BAIL;
  const parts: string[] = [];
  for (let i = 0; i < args.length; i++) {
    parts.push(_emit(args[i], perAtomNames, atomRefs, atomIdx, foldIRs));
  }
  return 'OPS.' + op + '(' + parts.join(',') + ')';
}

// Numeric literal → a safe JS source token. Rejects non-finite (NaN /
// ±Infinity have no literal form we want to splice) → bail.
function _numLit(v: number): string {
  if (!Number.isFinite(v)) throw _BAIL;
  // Use a parenthesised numeric form; negative numbers stay valid.
  return JSON.stringify(v);
}

// Build a Plan from an IR, or null if not compilable.
function compilePlan(ir: any, perAtomNames: Set<string>): Plan | null {
  // Only compile expressions that actually depend on per-atom data.
  // A fully atom-independent root would fold to a single constant and
  // return Float64Array(N), but the interpreter returns a scalar there
  // (broadcastN short-circuit) — bail so the interpreter owns that case
  // and the contract (and bit-exact type) is preserved.
  if (!_hasPerAtomRef(ir, perAtomNames)) return null;

  const atomRefs: string[] = [];
  const atomIdx = new Map<string, number>();
  const foldIRs: any[] = [];
  let body: string;
  try {
    const expr = _emit(ir, perAtomNames, atomRefs, atomIdx, foldIRs);
    body = 'for(let i=0;i<N;i++){out[i]=+(' + expr + ');}return out;';
  } catch (e) {
    if (e === _BAIL) return null;
    throw e;
  }
  // eslint-disable-next-line no-new-func
  const loopFn = new Function('OPS', 'A', 'C', 'out', 'N', body);
  return { loopFn, atomRefs, foldIRs };
}

// Resolve a per-atom ref's backing Float64Array(N). Accepts a bare
// Float64Array or a Value with shape [N] (scalar atoms). Returns null
// (→ caller bails) for vector atoms (shape length > 1) or anything else.
function _atomArray(v: any, N: number): Float64Array | null {
  if (v instanceof Float64Array && v.length === N) return v;
  if (v && Array.isArray(v.shape) && v.data instanceof Float64Array) {
    // Bail on complex values — the compiler handles real scalar atoms only.
    // A complex Value carries v.im (the imaginary part); passing only v.data
    // (the real part) to the loop would give wrong results for ops like abs2.
    if (v.dtype === 'complex' || v.im instanceof Float64Array) return null;
    if (v.shape.length === 1 && v.shape[0] === N) return v.data;
  }
  return null;
}

// Execute a Plan for one call. Returns Float64Array(N), or null if a
// runtime precondition fails (vector-atom ref, non-numeric fold) — the
// caller then falls back to the interpreter.
function runPlan(plan: Plan, refArrays: any, baseEnv: any, overlay: any, N: number): Float64Array | null {
  const A = new Array(plan.atomRefs.length);
  for (let k = 0; k < plan.atomRefs.length; k++) {
    const arr = _atomArray(refArrays ? refArrays[plan.atomRefs[k]] : undefined, N);
    if (arr === null) return null;
    A[k] = arr;
  }
  // Fold env: baseEnv with overlay overriding (matches _perAtomFallback).
  let foldEnv = baseEnv || {};
  if (overlay && Object.keys(overlay).length > 0) foldEnv = Object.assign({}, foldEnv, overlay);
  const Cc = new Array(plan.foldIRs.length);
  for (let j = 0; j < plan.foldIRs.length; j++) {
    const c = _evaluateExpr(plan.foldIRs[j], foldEnv);
    if (typeof c !== 'number' && typeof c !== 'boolean') return null;  // non-scalar fold → bail
    Cc[j] = +c;
  }
  const out = new Float64Array(N);
  return plan.loopFn(SCALAR_OPS, A, Cc, out, N);
}

module.exports = { initCompiler, _hasPerAtomRef, _COMPILE_ARITY, SCALAR_OPS, compilePlan, runPlan };
