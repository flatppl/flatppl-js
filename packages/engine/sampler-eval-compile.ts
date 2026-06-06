'use strict';

// =====================================================================
// sampler-eval-compile.ts — fused-loop compiler for evaluateExprN
// =====================================================================
//
// Compiles a scalar-elementwise IR expression to a single JS loop that
// computes one Float64Array(N) in one pass, with NO intermediate
// arrays. Bit-identical to the node-by-node interpreter in
// sampler-eval-batched.ts because it calls the SAME ARITH_OPS scalar
// primitives, in the same nesting/order, on the same scalars.
//
// Dependencies (ARITH_OPS, evaluateExpr, resolveConst) are injected via
// initCompiler() from sampler-eval-batched.ts's initARITHOPSN — avoids a
// require cycle and keeps those internals unexported.

let _OPS: any = null;            // ARITH_OPS scalar table
let _evaluateExpr: any = null;   // single-point evaluator (for folds)
let _resolveConst: any = null;   // const-name resolver

function initCompiler(deps: any): void {
  _OPS = deps.ARITH_OPS;
  _evaluateExpr = deps.evaluateExpr;
  _resolveConst = deps.resolveConst;
}

// Compilable scalar ops + arity. Exactly the _SCALAR_PRIM_ARITY set
// (each has an ARITH_OPS scalar entry). Structural (tuple/get_field),
// aggregate, approximation (polynomial/...), and complex ops are
// deliberately ABSENT: such a node either folds (when atom-independent)
// or forces a bail.
const _COMPILE_ARITY: Record<string, number> = {
  add: 2, sub: 2, mul: 2, div: 2, divide: 2, mod: 2, neg: 1, pos: 1, pow: 2,
  abs: 1, abs2: 1, exp: 1, log: 1, log10: 1, log1p: 1, expm1: 1, sqrt: 1,
  sin: 1, cos: 1, tan: 1, asin: 1, acos: 1, atan: 1, atan2: 2,
  sinh: 1, cosh: 1, tanh: 1, asinh: 1, acosh: 1, atanh: 1,
  floor: 1, ceil: 1, round: 1, min: 2, max: 2,
  gamma: 1, loggamma: 1, logit: 1, invlogit: 1, probit: 1, invprobit: 1,
  lt: 2, le: 2, gt: 2, ge: 2, equal: 2, unequal: 2,
  isfinite: 1, isinf: 1, isnan: 1, iszero: 1,
  land: 2, lor: 2, lxor: 2, lnot: 1, ifelse: 3,
  boolean: 1, integer: 1,
};

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
//   loopFn(OPS, A, C, out, N): A = per-atom Float64Array[]; C = folded
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
  // Emit a call to the SAME ARITH_OPS scalar primitive (bit-exact).
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
  return plan.loopFn(_OPS, A, Cc, out, N);
}

module.exports = { initCompiler, _hasPerAtomRef, _COMPILE_ARITY, compilePlan, runPlan };
