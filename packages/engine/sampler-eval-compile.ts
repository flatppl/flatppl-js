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

module.exports = { initCompiler, _hasPerAtomRef, _COMPILE_ARITY };
