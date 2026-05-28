'use strict';

// =====================================================================
// sampler-aggregate.ts — aggregate(f_reduction, output_axes, expr)
// (spec §04 §sec:aggregate; engine-concepts §16)
// =====================================================================
//
// Extracted from sampler.ts as part of the §17.5 sampler split
// (engine-concepts §11). Holds the AGGREGATE_PATTERNS specialiser
// table + the canonical broadcast-reduce default lowering.
//
// Multi-axis tensor contraction. Phase = join of input phases (this
// runs per-atom via the standard per-atom-fallback path; no batching
// at this layer in v0.1).
//
// Architecture (engine-concepts §16):
//   AGGREGATE_PATTERNS — list of pattern specialisers (matmul, …)
//     tried in order; first match wins. Gated by the `aggregate`
//     optimisation toggle.
//   _evalAggregateBroadcastReduce — the canonical permute-reshape-
//     broadcast-reduce einsum lowering. Default for everything the
//     specialisers don't catch; ALSO the correctness oracle the
//     dual-mode test runner compares specialisers against (with
//     `aggregate=off` forcing the broadcast-reduce path).
//
// CJS module — uses require() for the lazy cycle on sampler.ts, so
// the file must NOT use ES export syntax (mirrors sampler-registry.ts).

const valueLib = require('./value.ts');
const valueOps = require('./value-ops.ts');
const perfConfig = require('./perf-config.ts');
const { _matmul } = require('./sampler-linalg.ts');

// Lazy access to sampler.ts (cycle: aggregate calls `evaluateExpr` /
// `ARITH_OPS`, which the evaluator re-enters via `_evalAggregate`).
// We do NOT memoise the require — sampler.ts re-requires aggregate
// during load, so the first snapshot would be the partial-exports
// view (ARITH_OPS not yet set). Node caches the module object itself,
// so each `require('./sampler.ts')` returns the same live module.exports
// — cheap, and we always see the post-load state at call time.
function _sampler(): any {
  return require('./sampler.ts');
}
function evaluateExpr(ir: any, env: any): any {
  return _sampler().evaluateExpr(ir, env);
}
// Proxy stand-in for `ARITH_OPS` — every property read forwards to
// the live table on sampler.ts (which lives at `_internal.ARITH_OPS`,
// not at the top level). Lets the specialisers below reference
// `ARITH_OPS.transpose` / `ARITH_OPS.sum` etc. as if it were the
// in-module const.
const ARITH_OPS: any = new Proxy({}, {
  get(_target, prop) { return _sampler()._internal.ARITH_OPS[prop]; },
});

// Axis length inference walks the IR looking for the FIRST get(...) or
// get0(...) whose selector slot k is an axis ref; the length of that
// axis is then the length of the container's dim-k. Walk stops at
// nested aggregate boundaries — those have their own axis scope.
function _inferAggregateAxisLengths(exprIR: any, axisNames: string[], env: any) {
  const lengths: Record<string, number> = {};
  function _shapeOf(val: any): number[] | null {
    if (val == null) return null;
    if (typeof val === 'number' || typeof val === 'boolean') return [];
    if (Array.isArray(val)) {
      // Nested-array form: shape from outer length + recurse for inner.
      const inner: number[] = [];
      if (val.length > 0 && Array.isArray(val[0])) {
        const tail = _shapeOf(val[0]);
        if (tail) for (const x of tail) inner.push(x);
      }
      return [val.length, ...inner];
    }
    if (val && (val as any).BYTES_PER_ELEMENT) return [(val as any).length];
    if (val && Array.isArray((val as any).shape)) return (val as any).shape;
    return null;
  }
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.kind === 'call' && n.op === 'aggregate') return;   // inner scope
    if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
      const args = n.args || [];
      const container = args[0];
      const sels = args.slice(1);
      let containerVal: any = undefined;
      for (let k = 0; k < sels.length; k++) {
        const s = sels[k];
        if (s && s.kind === 'axis' && !(s.name in lengths)) {
          if (containerVal === undefined) {
            try { containerVal = evaluateExpr(container, env); }
            catch (_) { containerVal = null; }
          }
          if (containerVal == null) continue;
          const shape = _shapeOf(containerVal);
          if (shape && typeof shape[k] === 'number') {
            lengths[s.name] = shape[k];
          }
        }
      }
    }
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op' || k === 'name' || k === 'ns') continue;
      walk(n[k]);
    }
  }
  walk(exprIR);
  // Verify every needed axis got a length; raise a clear error if not.
  const missing = axisNames.filter((a: string) => !(a in lengths));
  if (missing.length > 0) {
    throw new Error(
      `aggregate: could not infer length of axis ${missing.map((a: string) => '.' + a).join(', ')} ` +
      `— each axis must index a known array at least once in expr`);
  }
  return lengths;
}

// Collect axis names that appear in expr, NOT descending into nested
// aggregate(...) bodies (each aggregate has its own closed axis scope
// per spec §05).
function _collectInScopeAxisNames(exprIR: any): Set<string> {
  const names = new Set<string>();
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.kind === 'axis') { names.add(n.name); return; }
    if (n.kind === 'call' && n.op === 'aggregate') return;
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op' || k === 'name' || k === 'ns') continue;
      walk(n[k]);
    }
  }
  walk(exprIR);
  return names;
}

// Pattern table — each entry recognises a specific aggregate shape
// and dispatches to an accelerated implementation. Tried in order;
// first match wins. The broadcast-reduce default
// (`_evalAggregateBroadcastReduce`) is the fallback AND the
// correctness oracle that every specialiser must match. Run tests
// with `perf-config.setOptimization('aggregate', false)` to force the
// broadcast-reduce path and verify equivalence.
//
// Each entry has the shape:
//   match(ir, env): match-state | null
//   execute(ir, env, match): result
// where `match` may carry pre-extracted IR refs (e.g. the two factor
// arrays for matmul) so `execute` doesn't have to re-walk the IR.
const AGGREGATE_PATTERNS: Array<{
  name: string;
  match: (ir: any, env: any) => any;
  execute: (ir: any, env: any, match: any) => any;
}> = [];

function _evalAggregate(ir: any, env: any): any {
  // Pattern dispatch is gated by the `aggregate` optimisation toggle.
  // When disabled, every aggregate flows through the broadcast-reduce
  // default (engine-concepts §16) so equivalence tests (`inBothModes`)
  // can verify each specialiser agrees with the canonical lowering.
  if (perfConfig.getOptimization('aggregate')) {
    for (const p of AGGREGATE_PATTERNS) {
      const m = p.match(ir, env);
      if (m) return p.execute(ir, env, m);
    }
  }
  return _evalAggregateBroadcastReduce(ir, env);
}

// ---------------------------------------------------------------------
// Helper for the matmul-family specialisers below: dispatch the
// nested-array or shape-rich-Value matrix product. Optionally
// transposes either operand before the multiplication. For shape-
// rich Values the transpose is a free Klein-4 tag flip; for nested
// arrays it's `ARITH_OPS.transpose`.
// ---------------------------------------------------------------------
function _matmulDispatch(A: any, B: any, transA: boolean, transB: boolean): any {
  if (valueLib.isValue(A) || valueLib.isValue(B)) {
    let aV = valueLib.asValue(A);
    let bV = valueLib.asValue(B);
    if (transA) aV = valueLib.transpose(aV);
    if (transB) bV = valueLib.transpose(bV);
    return valueOps.mul(aV, bV);
  }
  const aN = transA ? (ARITH_OPS as any).transpose(A) : A;
  const bN = transB ? (ARITH_OPS as any).transpose(B) : B;
  return _matmul(aN, bN);
}

// ---------------------------------------------------------------------
// Specialiser: matmul family (4 transpose variants)
//
//   aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])   ≡  A · B
//   aggregate(sum, [.i, .k], A[.j, .i] * B[.j, .k])   ≡  Aᵀ · B
//   aggregate(sum, [.i, .k], A[.i, .j] * B[.k, .j])   ≡  A · Bᵀ
//   aggregate(sum, [.i, .k], A[.j, .i] * B[.k, .j])   ≡  Aᵀ · Bᵀ
//
// Each operand's axes can be in either order; the matcher determines
// which dim each axis name occupies and sets a transpose flag.
// Scalar multiplication is commutative, so either factor order in
// the `mul` body matches the same logical product.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'matmul-family',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 2) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    const kName = outAxes[1].name;
    if (iName === kName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;

    // A's standard matmul layout is `A[.i, .j]` — output dim first,
    // reduce dim second; transposed is `A[.j, .i]`. B's standard is
    // `B[.j, .k]` — reduce dim first, output dim second; transposed
    // is `B[.k, .j]`. The two operands have ASYMMETRIC "normal"
    // conventions, so they get separate classifiers.
    function classifyA(fac: any):
      { src: any; trans: boolean; jName: string } | null
    {
      if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
      if (!fac.args || fac.args.length !== 3) return null;
      const s0 = fac.args[1], s1 = fac.args[2];
      if (!s0 || s0.kind !== 'axis') return null;
      if (!s1 || s1.kind !== 'axis') return null;
      if (s0.name === iName) return { src: fac.args[0], trans: false, jName: s1.name };
      if (s1.name === iName) return { src: fac.args[0], trans: true,  jName: s0.name };
      return null;
    }
    function classifyB(fac: any):
      { src: any; trans: boolean; jName: string } | null
    {
      if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
      if (!fac.args || fac.args.length !== 3) return null;
      const s0 = fac.args[1], s1 = fac.args[2];
      if (!s0 || s0.kind !== 'axis') return null;
      if (!s1 || s1.kind !== 'axis') return null;
      if (s1.name === kName) return { src: fac.args[0], trans: false, jName: s0.name };
      if (s0.name === kName) return { src: fac.args[0], trans: true,  jName: s1.name };
      return null;
    }

    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // Try assigning (f1=A, f2=B), then (f1=B, f2=A).
    for (const [fA, fB] of [[f1, f2], [f2, f1]]) {
      const ca = classifyA(fA);
      const cb = classifyB(fB);
      if (!ca || !cb) continue;
      // Same reduction axis from both sides.
      if (ca.jName !== cb.jName) continue;
      if (ca.jName === iName || ca.jName === kName) continue;
      return { aIR: ca.src, bIR: cb.src, transA: ca.trans, transB: cb.trans };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    return _matmulDispatch(A, B, match.transA, match.transB);
  },
});

// ---------------------------------------------------------------------
// Specialiser: matrix-vector multiplication (incl. transposed-A variant)
//
//   aggregate(sum, [.i], A[.i, .j] * v[.j])   ≡  A · v
//   aggregate(sum, [.i], A[.j, .i] * v[.j])   ≡  Aᵀ · v
//
// The single output axis owns A's first or second dim depending on
// which side `.i` appears; the shared axis (.j) iterates over A's
// other dim and over v.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'matvec',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 1) return null;
    if (outAxes[0].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // The matrix factor has rank-2 indexing (3 args to `get`); the
    // vector factor has rank-1 indexing (2 args). Try both orderings.
    for (const [matGet, vecGet] of [[f1, f2], [f2, f1]]) {
      if (!matGet || matGet.kind !== 'call' || matGet.op !== 'get') continue;
      if (!vecGet || vecGet.kind !== 'call' || vecGet.op !== 'get') continue;
      if (!matGet.args || matGet.args.length !== 3) continue;
      if (!vecGet.args || vecGet.args.length !== 2) continue;
      const mS = [matGet.args[1], matGet.args[2]];
      const vS = vecGet.args[1];
      if (mS[0].kind !== 'axis' || mS[1].kind !== 'axis') continue;
      if (!vS || vS.kind !== 'axis') continue;
      const jName = vS.name;
      if (jName === iName) continue;
      // Determine A's orientation: which dim is .i, which is .j?
      let transA = false;
      if (mS[0].name === iName && mS[1].name === jName) transA = false;
      else if (mS[0].name === jName && mS[1].name === iName) transA = true;
      else continue;
      return { aIR: matGet.args[0], vIR: vecGet.args[0], transA };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const v = evaluateExpr(match.vIR, env);
    if (valueLib.isValue(A) || valueLib.isValue(v)) {
      let aV = valueLib.asValue(A);
      if (match.transA) aV = valueLib.transpose(aV);
      return valueOps.mul(aV, valueLib.asValue(v));
    }
    // Nested-array matvec.
    const Ax = match.transA ? (ARITH_OPS as any).transpose(A) : A;
    const m = Ax.length;
    const n = Ax[0].length;
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = 0;
      const row = Ax[i];
      for (let j = 0; j < n; j++) s += row[j] * v[j];
      out[i] = s;
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: outer product
//
//   aggregate(<any>, [.i, .j], u[.i] * v[.j])   ≡  u · vᵀ (outer)
//
// No reduce axes → the reduction function is irrelevant (every
// "reduction" is over a single value). The broadcast-reduce default
// produces the right answer but materialises a `_broadcastTo` copy;
// the specialiser computes the outer product directly into one pass.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'outer-product',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [_fIR, axesIR, bodyIR] = args;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 2) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    const jName = outAxes[1].name;
    if (iName === jName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // Two rank-1 indexings, one on `.i` one on `.j`.
    for (const [uGet, vGet] of [[f1, f2], [f2, f1]]) {
      if (!uGet || uGet.kind !== 'call' || uGet.op !== 'get') continue;
      if (!vGet || vGet.kind !== 'call' || vGet.op !== 'get') continue;
      if (!uGet.args || uGet.args.length !== 2) continue;
      if (!vGet.args || vGet.args.length !== 2) continue;
      const us = uGet.args[1], vs = vGet.args[1];
      if (!us || us.kind !== 'axis' || us.name !== iName) continue;
      if (!vs || vs.kind !== 'axis' || vs.name !== jName) continue;
      return { uIR: uGet.args[0], vIR: vGet.args[0] };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const u = evaluateExpr(match.uIR, env);
    const v = evaluateExpr(match.vIR, env);
    // Direct outer product into a nested array. The Value-typed path
    // (atom-batched) routes through valueOps.mul which handles outer
    // via the tvector tag (vector × transposed-vector → matrix per
    // spec §07 linear-algebra rules).
    if (valueLib.isValue(u) || valueLib.isValue(v)) {
      const uV = valueLib.asValue(u);
      const vV = valueLib.asValue(v);
      // Outer = uV (col) · transpose(vV) (row). transpose() flips the
      // Klein-4 tag; valueOps.mul then dispatches to outer.
      return valueOps.mul(uV, valueLib.transpose(vV));
    }
    const m = u.length, n = v.length;
    const out: any[] = new Array(m);
    for (let i = 0; i < m; i++) {
      const row = new Float64Array(n);
      const ui = u[i];
      for (let j = 0; j < n; j++) row[j] = ui * v[j];
      out[i] = row;
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: batched matmul
//
//   aggregate(sum, [.b, .i, .k],
//             A[.b, .i, .j] * B[.b, .j, .k])  ≡ per-batch A·B
//
// Common in atom-batched contexts (sampler runs aggregate per
// atom; the leading axis broadcasts the per-atom data). Avoids the
// full O(Nb · Ni · Nj · Nk) broadcast intermediate the default
// would build — at most O(Nb · max(Ni·Nj, Nj·Nk, Ni·Nk)) per
// stage.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'batched-matmul',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 3) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis'
        || outAxes[2].kind !== 'axis') return null;
    const bName = outAxes[0].name;
    const iName = outAxes[1].name;
    const kName = outAxes[2].name;
    if (bName === iName || iName === kName || bName === kName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    for (const [aGet, bGet] of [[f1, f2], [f2, f1]]) {
      if (!aGet || aGet.kind !== 'call' || aGet.op !== 'get') continue;
      if (!bGet || bGet.kind !== 'call' || bGet.op !== 'get') continue;
      if (!aGet.args || aGet.args.length !== 4) continue;
      if (!bGet.args || bGet.args.length !== 4) continue;
      const aSels = [aGet.args[1], aGet.args[2], aGet.args[3]];
      const bSels = [bGet.args[1], bGet.args[2], bGet.args[3]];
      if (aSels.some((s: any) => !s || s.kind !== 'axis')) continue;
      if (bSels.some((s: any) => !s || s.kind !== 'axis')) continue;
      // Canonical layout: A[.b, .i, .j], B[.b, .j, .k]. (Other
      // permutations are an extension; this matcher targets the
      // typical case.)
      if (aSels[0].name !== bName) continue;
      if (aSels[1].name !== iName) continue;
      if (bSels[0].name !== bName) continue;
      if (bSels[2].name !== kName) continue;
      const jName = aSels[2].name;
      if (jName === bName || jName === iName || jName === kName) continue;
      if (bSels[1].name !== jName) continue;
      return { aIR: aGet.args[0], bIR: bGet.args[0] };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    // Per-batch nested-array matmul. (Value-typed batched matmul
    // would go through valueOps with the batch axis as leading dim;
    // for v0.1 the nested-array path covers the typical user case.)
    const Nb = A.length;
    const out: any[] = new Array(Nb);
    for (let b = 0; b < Nb; b++) {
      out[b] = _matmul(A[b], B[b]);
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: pure axis reduction (single source, no arithmetic)
//
//   aggregate(f, [out_axes…], get(arr, sels…))
//
// Common pattern: column sums, row means, axis-wise max/min, etc.
// All seven reductions are supported via _AGGREGATE_REDUCTIONS. The
// fast path applies the reduction along each non-output dim directly
// on the source — no broadcast intermediate, no per-axis lift.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'pure-axis-reduction',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    const fname = (fIR && fIR.kind === 'ref' && fIR.name) || null;
    if (!fname || !_AGGREGATE_REDUCTIONS[fname]) return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length < 1) return null;
    const outAxisNames: string[] = [];
    for (const a of outAxes) {
      if (!a || a.kind !== 'axis') return null;
      outAxisNames.push(a.name);
    }
    // Body must be a single get / get0 — no arithmetic.
    if (!bodyIR || bodyIR.kind !== 'call') return null;
    if (bodyIR.op !== 'get' && bodyIR.op !== 'get0') return null;
    if (!bodyIR.args || bodyIR.args.length < 2) return null;
    // Every selector must be an axis ref (no `only`/`all`/integer in
    // this fast path — those would require pre-applying selectors,
    // which the broadcast-reduce default does already).
    const sels = bodyIR.args.slice(1);
    const selAxisAt: Record<string, number> = {};
    for (let k = 0; k < sels.length; k++) {
      const s = sels[k];
      if (!s || s.kind !== 'axis') return null;
      if (s.name in selAxisAt) return null;  // same axis twice = diagonal extraction (not this fast path)
      selAxisAt[s.name] = k;
    }
    // Every output axis must be present in the body's selectors.
    for (const a of outAxisNames) if (!(a in selAxisAt)) return null;
    return { srcIR: bodyIR.args[0], fname, outAxisNames, selAxisAt, numSourceDims: sels.length };
  },
  execute(_ir: any, env: any, match: any): any {
    const src = evaluateExpr(match.srcIR, env);
    const flat = _toFlat(src);
    const sourceShape = flat.shape;
    const sourceStrides = _rowMajorStrides(sourceShape);

    // Determine per-output-cell traversal: iterate output axes,
    // for each (output-coord) reduce across the remaining dims.
    const outDimSizes = match.outAxisNames.map(
      (a: string) => sourceShape[match.selAxisAt[a]]);
    const outSize = outDimSizes.reduce((p: number, n: number) => p * n, 1);
    const outStrides = match.outAxisNames.map(
      (a: string) => sourceStrides[match.selAxisAt[a]]);

    // Reduce axes = source dims not in output_axes.
    const reduceDims: { sourceDim: number; size: number }[] = [];
    const inOutput = new Set(match.outAxisNames);
    for (const a of Object.keys(match.selAxisAt)) {
      if (!inOutput.has(a)) {
        const d = match.selAxisAt[a];
        reduceDims.push({ sourceDim: d, size: sourceShape[d] });
      }
    }
    const reduceSize = reduceDims.reduce(
      (p: number, x: any) => p * x.size, 1);

    const outData = new Float64Array(outSize);
    const tmp = new Float64Array(reduceSize);
    const outCoord = new Array(match.outAxisNames.length).fill(0);
    const redCoord = new Array(reduceDims.length).fill(0);
    for (let i = 0; i < outSize; i++) {
      let baseOff = 0;
      for (let k = 0; k < outCoord.length; k++) baseOff += outCoord[k] * outStrides[k];
      // Reduce over the reduce-dims at this output cell.
      for (let r = 0; r < reduceDims.length; r++) redCoord[r] = 0;
      for (let r = 0; r < reduceSize; r++) {
        let off = baseOff;
        for (let k = 0; k < reduceDims.length; k++) {
          off += redCoord[k] * sourceStrides[reduceDims[k].sourceDim];
        }
        tmp[r] = flat.data[off];
        // Increment redCoord (last dim fastest).
        for (let k = reduceDims.length - 1; k >= 0; k--) {
          redCoord[k]++;
          if (redCoord[k] < reduceDims[k].size) break;
          redCoord[k] = 0;
        }
      }
      outData[i] = _applyAggregateReduction(match.fname, tmp, reduceSize);
      // Increment outCoord.
      for (let k = outCoord.length - 1; k >= 0; k--) {
        outCoord[k]++;
        if (outCoord[k] < outDimSizes[k]) break;
        outCoord[k] = 0;
      }
    }
    return _flatToNested(outData, outDimSizes);
  },
});

// Map reduction name → ARITH_OPS implementation.
//
// IMPORTANT: do not call `_AGGREGATE_REDUCTIONS[fname](data)` from
// new aggregate paths. Use `_applyAggregateReduction(fname, data,
// expectedCount)` (below). The wrapper enforces the materialisation
// invariant the non-linear reductions (mean / var / std) rely on:
// `data.length` must equal the count of distinct values being
// reduced over — singleton-expanded / strided views give silently
// wrong results for mean (wrong denominator) and var / std (wrong
// Bessel correction). The two existing call sites
// (`_evalAggregateBroadcastReduce`, the pure-axis-reduction
// specialiser) both materialise their `tmp` buffer to `reduceSize`
// genuine elements before calling, and route through the wrapper
// to make that contract explicit and check-able.
const _AGGREGATE_REDUCTIONS: Record<string, (a: any) => number> = {
  sum:     (a: any) => (ARITH_OPS as any).sum(a),
  prod:    (a: any) => (ARITH_OPS as any).prod(a),
  mean:    (a: any) => (ARITH_OPS as any).mean(a),
  var:     (a: any) => (ARITH_OPS as any).var(a),
  std:     (a: any) => (ARITH_OPS as any).std(a),
  maximum: (a: any) => (ARITH_OPS as any).maximum(a),
  minimum: (a: any) => (ARITH_OPS as any).minimum(a),
};

// Reductions whose result depends on the count (n) of input
// values, not just their multiset content. These are silently
// wrong if `data` is a singleton-broadcast view rather than a
// fully-materialised buffer.
const _COUNT_DEPENDENT_REDUCTIONS = new Set(['mean', 'var', 'std']);

// Sanctioned entry point for applying a reduction to a buffer in an
// aggregate context. `expectedCount` should be the genuine count of
// elements being reduced (i.e. the product of reduce-axis sizes for
// this output cell). If the caller passes a Float64Array whose
// length disagrees with `expectedCount`, we throw — this catches
// the future-specialiser footgun of feeding a non-linear reduction
// a singleton-expanded view.
//
// Pass `expectedCount` as the integer count, or `null` to skip the
// check (use this only when the caller has already validated
// materialisation upstream, e.g. via `_broadcastTo(lifted,
// fullShape)`).
function _applyAggregateReduction(
  fname: string,
  data: Float64Array,
  expectedCount: number | null,
): number {
  const reduce = _AGGREGATE_REDUCTIONS[fname];
  if (!reduce) {
    throw new Error(`aggregate: unknown reduction '${fname}'`);
  }
  if (expectedCount !== null && _COUNT_DEPENDENT_REDUCTIONS.has(fname)) {
    if (data.length !== expectedCount) {
      throw new Error(`aggregate: reduction '${fname}' expects a fully `
        + `materialised buffer of length ${expectedCount}, got ${data.length}; `
        + 'singleton-expanded views give wrong mean / var / std (count '
        + 'mismatch). Materialise via _broadcastTo before reducing.');
    }
  }
  return +reduce(data);
}

// =====================================================================
// Broadcast-reduce default implementation
// =====================================================================
//
// See engine-concepts §16 for the rationale. The pipeline:
//
//   1. Determine canonical axis order = [output_axes…, reduce_axes…].
//   2. For each `get(arr, ...)` in expr, build an "aligned tensor": a
//      Float64Array + shape, where the shape has one dim per canonical
//      axis. Dims the source actually uses carry the axis's length;
//      dims it doesn't are singletons (broadcast-stretchable).
//      Non-axis selectors (integer / `only`) are pre-applied at this
//      stage so the aligned tensor only carries the surviving dims.
//   3. Lift the rest of `expr` to operate on aligned tensors. Each
//      arithmetic op is a broadcast-elementwise sweep over the
//      canonical-order coordinate space — one flat loop per op.
//      Subtrees with no axis refs evaluate to scalars and broadcast
//      as singletons.
//   4. Tail-reduce: with canonical order putting reduce axes last,
//      the reduction is a contiguous-block sweep over the trailing
//      `reduceSize` cells per output slot.
//
// All steps are vectorised tensor primitives — the same ops every
// accelerated backend offers. No per-axis-coord JS function-call
// overhead.

function _shapeProd(shape: number[]): number {
  let p = 1; for (let i = 0; i < shape.length; i++) p *= shape[i]; return p;
}

// Row-major strides for a shape: stride[i] = product of shape[i+1..].
function _rowMajorStrides(shape: number[]): number[] {
  const N = shape.length;
  const out = new Array(N);
  let s = 1;
  for (let i = N - 1; i >= 0; i--) { out[i] = s; s *= shape[i]; }
  return out;
}

// Broadcasting strides: along any dim of size 1, the stride is 0
// (don't advance in that dim — repeat the singleton value).
function _broadcastStrides(srcShape: number[], outShape: number[]): number[] {
  const N = srcShape.length;
  const out = new Array(N);
  let s = 1;
  for (let i = N - 1; i >= 0; i--) {
    out[i] = (srcShape[i] === 1) ? 0 : s;
    s *= srcShape[i];
  }
  return out;
}

// Convert a nested-array (or 1-D typed array) value to {data, shape}.
// Nested → flat: walk in row-major order. Typed-array → wraps as
// shape=[length]. Scalar → shape=[].
function _toFlat(val: any): { data: Float64Array; shape: number[] } {
  if (val == null) return { data: new Float64Array([0]), shape: [] };
  if (typeof val === 'number' || typeof val === 'boolean') {
    return { data: new Float64Array([+val]), shape: [] };
  }
  if (val && (val as any).BYTES_PER_ELEMENT !== undefined) {
    // Typed array — 1-D.
    return { data: val as Float64Array, shape: [val.length] };
  }
  if (Array.isArray(val)) {
    // Determine shape by walking the first element down.
    const shape: number[] = [val.length];
    let probe: any = val[0];
    while (Array.isArray(probe)
           || (probe && probe.BYTES_PER_ELEMENT !== undefined)) {
      shape.push(probe.length);
      probe = probe[0];
    }
    const size = _shapeProd(shape);
    const data = new Float64Array(size);
    let w = 0;
    function fill(x: any) {
      if (Array.isArray(x) || (x && x.BYTES_PER_ELEMENT !== undefined)) {
        for (let i = 0; i < x.length; i++) fill(x[i]);
      } else {
        data[w++] = +x;
      }
    }
    fill(val);
    return { data, shape };
  }
  if (valueLib.isValue(val)) {
    return { data: val.data as Float64Array, shape: val.shape.slice() };
  }
  throw new Error(`aggregate: cannot interpret value of type ${typeof val} as a tensor`);
}

// Flat data + shape → nested-JS-array (rank > 1) or Float64Array (rank
// 1) or scalar (rank 0). Matches the existing output convention from
// `_matmul` and the linspace family.
function _flatToNested(data: Float64Array, shape: number[]): any {
  if (shape.length === 0) return data[0];
  if (shape.length === 1) return data;
  function build(start: number, sh: number[]): any {
    if (sh.length === 1) {
      const out = new Float64Array(sh[0]);
      for (let i = 0; i < sh[0]; i++) out[i] = data[start + i];
      return out;
    }
    const stride = _shapeProd(sh.slice(1));
    const out: any[] = new Array(sh[0]);
    for (let i = 0; i < sh[0]; i++) {
      out[i] = build(start + i * stride, sh.slice(1));
    }
    return out;
  }
  return build(0, shape);
}

// True if `node` (an IR subtree) contains any `kind: 'axis'` node.
// Used for constant-hoisting: axis-free subtrees evaluate once and
// broadcast as singletons. Stops at nested aggregate boundaries
// (those have a closed axis scope per spec §05).
function _containsAxisRef(node: any): boolean {
  if (!node || typeof node !== 'object') return false;
  if (Array.isArray(node)) {
    for (const c of node) if (_containsAxisRef(c)) return true;
    return false;
  }
  if (node.kind === 'axis') return true;
  if (node.kind === 'call' && node.op === 'aggregate') return false;
  for (const k of Object.keys(node)) {
    if (k === 'loc' || k === 'kind' || k === 'op'
        || k === 'name' || k === 'ns') continue;
    if (_containsAxisRef(node[k])) return true;
  }
  return false;
}

// Build an aligned tensor for a `get(arr, sel0, sel1, …)` IR call.
// The resulting shape has `canonicalAxes.length` dims; each is either
// the axis's length (if this source uses that axis) or 1 (singleton,
// broadcast-stretchable). Non-axis selectors (integer indices, `only`)
// are pre-applied and collapse their source dim before the alignment.
function _alignedTensorFromGet(
  getIR: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
): { data: Float64Array; shape: number[] } {
  const args = getIR.args || [];
  const oneBased = getIR.op === 'get';
  const arr = evaluateExpr(args[0], env);
  const sels = args.slice(1);
  const src = _toFlat(arr);
  const sourceStrides = _rowMajorStrides(src.shape);

  // Walk selectors. For each, decide:
  //   - axis ref: remember this dim corresponds to a canonical axis.
  //   - integer / 'only': collapse this dim (advance baseOffset).
  //   - 'all': not supported in aggregate body (per §04 spec the body
  //     uses axis names; `all`/':' would mean "keep an unnamed dim"
  //     which has no defined semantics for the contraction).
  const axisAt: Record<string, number> = {};   // axisName → source dim
  let baseOffset = 0;
  for (let k = 0; k < sels.length; k++) {
    const s = sels[k];
    if (s && s.kind === 'axis') {
      axisAt[s.name] = k;
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'only') {
      if (src.shape[k] !== 1) {
        throw new Error(`aggregate: 'only' selector requires the indexed `
          + `axis to have length 1, got length ${src.shape[k]}`);
      }
      // Index 0 — contributes 0 to baseOffset.
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'all') {
      throw new Error(`aggregate: 'all' / ':' is not supported in an `
        + `aggregate body; use an axis name (.name) instead`);
    }
    // Integer index — evaluate (1-based for `get`, 0-based for `get0`).
    const idx = +evaluateExpr(s, env);
    const idx0 = oneBased ? (idx | 0) - 1 : (idx | 0);
    if (idx0 < 0 || idx0 >= src.shape[k]) {
      throw new Error(`aggregate: index ${oneBased ? idx : idx0} out of bounds `
        + `for axis of length ${src.shape[k]}`);
    }
    baseOffset += idx0 * sourceStrides[k];
  }

  // Build the aligned shape + a per-canonical-position source stride.
  // If the canonical axis isn't in this source, the dim is singleton
  // and contributes 0 to the source offset (broadcast).
  const N = canonicalAxes.length;
  const alignedShape = new Array(N);
  const sourceStrideAt: number[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const axisName = canonicalAxes[i];
    if (axisName in axisAt) {
      alignedShape[i] = axisLengths[axisName];
      sourceStrideAt[i] = sourceStrides[axisAt[axisName]];
    } else {
      alignedShape[i] = 1;
      sourceStrideAt[i] = 0;
    }
  }

  // Materialise the aligned tensor. Iterate output cells in row-major
  // order with a coordinate vector; for each, compute the source
  // offset = baseOffset + Σ coord[i] · sourceStrideAt[i].
  const alignedSize = _shapeProd(alignedShape);
  const out = new Float64Array(alignedSize);
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < alignedSize; linear++) {
    let srcOff = baseOffset;
    for (let i = 0; i < N; i++) srcOff += coord[i] * sourceStrideAt[i];
    out[linear] = src.data[srcOff];
    // Increment coord, last dim fastest.
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < alignedShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: alignedShape };
}

// Inline scalar implementations for the most common binary ops in
// aggregate bodies — bypassing the ARITH_OPS dispatch saves a JS
// function call per element in the broadcast inner loop.
const _AGG_BIN: Record<string, (x: number, y: number) => number> = {
  add:    (x, y) => x + y,
  sub:    (x, y) => x - y,
  mul:    (x, y) => x * y,
  div:    (x, y) => x / y,
  divide: (x, y) => x / y,
  pow:    (x, y) => {
    // Math.pow is ~100 ns per call; for the common integer-exponent
    // cases (polyeval / indexof0 patterns produce exponents 0, 1, 2,
    // 3) repeated mul is 10–20× faster. Falls through to Math.pow
    // for non-small / non-integer exponents.
    if (y === (y | 0)) {
      if (y === 0) return 1;
      if (y === 1) return x;
      if (y === 2) return x * x;
      if (y === 3) return x * x * x;
      if (y === 4) { const x2 = x * x; return x2 * x2; }
    }
    return Math.pow(x, y);
  },
  mod:    (x, y) => x % y,
};

const _AGG_UN: Record<string, (x: number) => number> = {
  neg: (x) => -x,    pos: (x) => +x,
  exp:   Math.exp,    log:   Math.log,    log10: Math.log10,
  log1p: Math.log1p,  expm1: Math.expm1,
  sqrt:  Math.sqrt,   abs:   Math.abs,    abs2: (x) => x * x,
  sin:   Math.sin,    cos:   Math.cos,    tan:   Math.tan,
  asin:  Math.asin,   acos:  Math.acos,   atan:  Math.atan,
  sinh:  Math.sinh,   cosh:  Math.cosh,   tanh:  Math.tanh,
  asinh: Math.asinh,  acosh: Math.acosh,  atanh: Math.atanh,
  floor: Math.floor,  ceil:  Math.ceil,   round: Math.round,
};

// Broadcast-elementwise binary op over two aligned tensors. Each
// already has shape canonicalAxes.length; broadcasting comes from
// per-dim singleton vs full-length stride choice.
function _broadcastBinary(
  a: { data: Float64Array; shape: number[] },
  b: { data: Float64Array; shape: number[] },
  fn: (x: number, y: number) => number,
): { data: Float64Array; shape: number[] } {
  const N = a.shape.length;
  const outShape = new Array(N);
  for (let i = 0; i < N; i++) outShape[i] = Math.max(a.shape[i], b.shape[i]);
  const outSize = _shapeProd(outShape);
  const aStrides = _broadcastStrides(a.shape, outShape);
  const bStrides = _broadcastStrides(b.shape, outShape);
  const out = new Float64Array(outSize);
  // Fast paths: rank-specific nested loops — much friendlier to V8's
  // JIT than the generic coord-walker. The aggregate runtime hits
  // rank-3 most often (one atom dim + one output axis + one reduce
  // axis), and rank-2 for value-domain broadcasts. The generic
  // fallback handles ranks 0, 1, and ≥4.
  if (N === 3) {
    const D0 = outShape[0], D1 = outShape[1], D2 = outShape[2];
    const aS0 = aStrides[0], aS1 = aStrides[1], aS2 = aStrides[2];
    const bS0 = bStrides[0], bS1 = bStrides[1], bS2 = bStrides[2];
    const aData = a.data, bData = b.data;
    let outIdx = 0;
    for (let i0 = 0; i0 < D0; i0++) {
      const aOff0 = i0 * aS0, bOff0 = i0 * bS0;
      for (let i1 = 0; i1 < D1; i1++) {
        const aOff1 = aOff0 + i1 * aS1, bOff1 = bOff0 + i1 * bS1;
        for (let i2 = 0; i2 < D2; i2++) {
          out[outIdx++] = fn(aData[aOff1 + i2 * aS2], bData[bOff1 + i2 * bS2]);
        }
      }
    }
    return { data: out, shape: outShape };
  }
  if (N === 2) {
    const D0 = outShape[0], D1 = outShape[1];
    const aS0 = aStrides[0], aS1 = aStrides[1];
    const bS0 = bStrides[0], bS1 = bStrides[1];
    const aData = a.data, bData = b.data;
    let outIdx = 0;
    for (let i0 = 0; i0 < D0; i0++) {
      const aOff0 = i0 * aS0, bOff0 = i0 * bS0;
      for (let i1 = 0; i1 < D1; i1++) {
        out[outIdx++] = fn(aData[aOff0 + i1 * aS1], bData[bOff0 + i1 * bS1]);
      }
    }
    return { data: out, shape: outShape };
  }
  if (N === 1) {
    const D0 = outShape[0];
    const aS0 = aStrides[0], bS0 = bStrides[0];
    const aData = a.data, bData = b.data;
    for (let i0 = 0; i0 < D0; i0++) {
      out[i0] = fn(aData[i0 * aS0], bData[i0 * bS0]);
    }
    return { data: out, shape: outShape };
  }
  if (N === 0) {
    out[0] = fn(a.data[0], b.data[0]);
    return { data: out, shape: outShape };
  }
  // Generic coord-walker for rank ≥4.
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < outSize; linear++) {
    let aOff = 0, bOff = 0;
    for (let i = 0; i < N; i++) { aOff += coord[i] * aStrides[i]; bOff += coord[i] * bStrides[i]; }
    out[linear] = fn(a.data[aOff], b.data[bOff]);
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < outShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: outShape };
}

function _broadcastUnary(
  a: { data: Float64Array; shape: number[] },
  fn: (x: number) => number,
): { data: Float64Array; shape: number[] } {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]);
  return { data: out, shape: a.shape.slice() };
}

// Lift an expression IR subtree to an aligned tensor of canonical
// shape. Each leaf becomes either a scalar (axis-free subtree) or an
// aligned tensor (a `get` call); each interior arithmetic op becomes
// a broadcast-elementwise sweep.
function _liftAggregateExpr(
  node: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
): { data: Float64Array; shape: number[] } {
  // Constant-hoist: subtree with no axis refs evaluates once and
  // broadcasts as all-singleton.
  if (!_containsAxisRef(node)) {
    const v = +evaluateExpr(node, env);
    const shape = canonicalAxes.map(() => 1);
    return { data: new Float64Array([v]), shape };
  }
  // `get` / `get0`: produces an aligned tensor at this leaf.
  if (node.kind === 'call' && (node.op === 'get' || node.op === 'get0')) {
    return _alignedTensorFromGet(node, canonicalAxes, axisLengths, env);
  }
  // Binary arithmetic: lift both operands, broadcast-apply.
  if (node.kind === 'call' && node.args && node.args.length === 2) {
    const fn = _AGG_BIN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env);
      const b = _liftAggregateExpr(node.args[1], canonicalAxes, axisLengths, env);
      return _broadcastBinary(a, b, fn);
    }
  }
  // Unary scalar math.
  if (node.kind === 'call' && node.args && node.args.length === 1) {
    const fn = _AGG_UN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env);
      return _broadcastUnary(a, fn);
    }
  }
  throw new Error(`aggregate: unsupported op '${node.op || node.kind}' `
    + `in body — broadcast-reduce default supports arithmetic and `
    + `unary math on indexed arrays. Hoist non-broadcasting subterms `
    + `to bindings outside the aggregate.`);
}

function _evalAggregateBroadcastReduce(ir: any, env: any): any {
  const args = ir.args || [];
  if (args.length !== 3) {
    throw new Error(`aggregate: expected 3 args, got ${args.length}`);
  }
  const [fIR, axesIR, exprIR] = args;

  const fname = (fIR.kind === 'ref' && fIR.name)
              || (fIR.kind === 'const' && fIR.name);
  const reduce = fname && _AGGREGATE_REDUCTIONS[fname];
  if (!reduce) throw new Error(`aggregate: unknown reduction '${fname}'`);

  if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') {
    throw new Error('aggregate: output_axes must be an array literal of axis names');
  }
  const outAxes = (axesIR.args || []).map((a: any) => {
    if (a.kind !== 'axis') throw new Error('aggregate: output_axes entries must be axis names (.name)');
    return a.name;
  });

  const usedAxes = _collectInScopeAxisNames(exprIR);
  const reduceAxes: string[] = [];
  for (const a of usedAxes) if (!outAxes.includes(a)) reduceAxes.push(a);

  // Canonical axis order: [output…, reduce…]. With reduce axes at the
  // end the final reduction is a tail-reduce over a contiguous block.
  const canonicalAxes = [...outAxes, ...reduceAxes];
  const lengths = _inferAggregateAxisLengths(exprIR, canonicalAxes, env);

  // Lift the body to an aligned tensor over the full canonical shape.
  const lifted = _liftAggregateExpr(exprIR, canonicalAxes, lengths, env);
  // Make sure every dim is materialised (no remaining singletons that
  // need broadcasting; the result tensor has every canonical-axis
  // length as its dim). If the body only depended on output axes and
  // some reduce axes were untouched, that dim is still 1 — reduce-by-
  // singleton is a no-op for sum, multiplication by 1 for prod, etc.
  // — we explicitly handle this below.
  const outShape = outAxes.map((a: string) => lengths[a]);
  const reduceShape = reduceAxes.map((a: string) => lengths[a]);
  const outSize = _shapeProd(outShape);

  // Some lifted dims may still be 1 (singleton) where the canonical
  // dim should be `lengths[a]`. Materialise by stretching to the full
  // shape before reducing — strictly necessary only when the
  // reduction is non-linear in the count (e.g. mean / var / std), but
  // doing it uniformly keeps the reduction code simple.
  if (reduceAxes.length === 0) {
    // No reduction needed. The output IS the lifted tensor (with any
    // singleton dims broadcast to their full length).
    const fullShape = outShape.slice();
    const full = _broadcastTo(lifted, fullShape);
    return _flatToNested(full.data, fullShape);
  }

  const fullShape = canonicalAxes.map((a) => lengths[a]);
  const full = _broadcastTo(lifted, fullShape);

  // Tail-reduce: with `full.shape` = [output…, reduce…], each output
  // cell owns a contiguous run of `reduceSize` elements in the flat
  // buffer. Slice each run and apply the reduction.
  const reduceSize = _shapeProd(reduceShape);
  const outData = new Float64Array(outSize);
  const tmp = new Float64Array(reduceSize);
  for (let i = 0; i < outSize; i++) {
    const base = i * reduceSize;
    for (let j = 0; j < reduceSize; j++) tmp[j] = full.data[base + j];
    outData[i] = _applyAggregateReduction(fname!, tmp, reduceSize);
  }
  return _flatToNested(outData, outShape);
}

// Materialise a singleton-broadcast tensor to its full shape — copy
// data so every dim equals its target length. After this the data
// buffer is contiguous in row-major order over `targetShape`.
function _broadcastTo(
  t: { data: Float64Array; shape: number[] },
  targetShape: number[],
): { data: Float64Array; shape: number[] } {
  // Fast path: already at target shape.
  let same = true;
  for (let i = 0; i < targetShape.length; i++) {
    if (t.shape[i] !== targetShape[i]) { same = false; break; }
  }
  if (same) return t;
  const N = targetShape.length;
  const outSize = _shapeProd(targetShape);
  const srcStrides = _broadcastStrides(t.shape, targetShape);
  const out = new Float64Array(outSize);
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < outSize; linear++) {
    let srcOff = 0;
    for (let i = 0; i < N; i++) srcOff += coord[i] * srcStrides[i];
    out[linear] = t.data[srcOff];
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < targetShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: targetShape.slice() };
}

// =====================================================================
// Atom-batched aggregate evaluator (engine-concepts §20.10.10)
// =====================================================================
//
// `_evalAggregateBroadcastReduceN(ir, refArrays, N, baseEnv, overlay)`
// is the atom-batched analogue of `_evalAggregateBroadcastReduce`.
// Used by `evaluateExprN` when an `aggregate(...)` IR is encountered
// in a per-atom-batched context (typically: fusion (a) Step 2 output
// where one or more body refs are stochastic / atom-batched). The
// alternative per-atom fallback evaluates the aggregate once per
// engine atom, incurring ~40 µs/atom of axis-inference + body-lift
// overhead; the batched version prepends an implicit atom axis to
// the canonical-axis ordering and lifts the body tensor to shape
// `[N, ...outAxes, ...reduceAxes]` in ONE pass, then tail-reduces
// per (atom, output_tuple).
//
// **Refs in the body:**
//   - If a ref's value is atom-batched (a Value with shape =
//     [N, ...rest]), the leading N is treated as the atom axis;
//     remaining dims provide the selector axes from `get(...)`.
//   - Otherwise (atom-indep — bare scalar, bare typed-array, or
//     Value without leading N), broadcast across the atom axis
//     (singleton stride for the atom dim).
//
// **Output:** a Value with shape=[N, ...outAxes] (atom-major).
//
// **Coverage:** matches the non-batched `_evalAggregateBroadcastReduce`
// for atom-indep refs (the atom dim becomes singleton and the result
// is broadcast back). For atom-batched refs, the per-atom slicing
// loop in `_perAtomFallback` collapses to one batched-lift +
// tail-reduce.

function _evalAggregateBroadcastReduceN(
  ir: any, refArrays: any, N: number, baseEnv: any, overlay: any,
): any {
  const args = ir.args || [];
  if (args.length !== 3) {
    throw new Error(`aggregateN: expected 3 args, got ${args.length}`);
  }
  const [fIR, axesIR, exprIR] = args;
  const fname = (fIR.kind === 'ref' && fIR.name)
              || (fIR.kind === 'const' && fIR.name);
  const reduce = fname && _AGGREGATE_REDUCTIONS[fname];
  if (!reduce) throw new Error(`aggregateN: unknown reduction '${fname}'`);
  if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') {
    throw new Error('aggregateN: output_axes must be an array literal of axis names');
  }
  const outAxes = (axesIR.args || []).map((a: any) => {
    if (a.kind !== 'axis')
      throw new Error('aggregateN: output_axes entries must be axis names (.name)');
    return a.name;
  });

  // Atom axis prepends to the canonical ordering. Its name uses a
  // double-underscore prefix to avoid colliding with any user-axis
  // name (spec §05 axis labels start with a letter).
  const ATOM_AXIS = '__atom_N';
  const usedAxes = _collectInScopeAxisNames(exprIR);
  const reduceAxes: string[] = [];
  for (const a of usedAxes) if (!outAxes.includes(a)) reduceAxes.push(a);
  const canonicalAxes = [ATOM_AXIS, ...outAxes, ...reduceAxes];

  // Build an env for evaluating subtree containers. Containers may be
  // bound in refArrays (per-atom values) OR baseEnv (atom-indep
  // values). Overlay (highest priority) is applied last. Track WHICH
  // names are atom-batched (membership in refArrays — not shape-
  // sniffed, since an atom-indep array might coincidentally have
  // shape[0] === N) so the lifter knows where the atom dim lives.
  const env: Record<string, any> = Object.assign({}, baseEnv || {});
  const atomBatchedNames: Set<string> = new Set();
  if (refArrays) {
    for (const k of Object.keys(refArrays)) {
      env[k] = refArrays[k];
      atomBatchedNames.add(k);
    }
  }
  if (overlay) {
    for (const k of Object.keys(overlay)) {
      env[k] = overlay[k];
      // Overlay wins; if a name was atom-batched but overlay
      // overrides, treat the new value at face value (no atom
      // unless leading dim matches N — overlay is a refinement
      // pattern usually used to inject atom-indep substitutes).
      atomBatchedNames.delete(k);
    }
  }

  // Infer axis lengths. Atom axis is N (caller-given). Other axes
  // come from get(arr, .ax) shape inspection — for atom-batched
  // refs the shape's first dim is N (stripped during shape-of); for
  // atom-indep refs the shape is taken verbatim.
  const axisLengths: Record<string, number> = { [ATOM_AXIS]: N };
  _inferAxisLengthsN(exprIR, outAxes.concat(reduceAxes), axisLengths,
                     env, atomBatchedNames, N);

  // Lift the body to a [N, ...outAxes, ...reduceAxes] tensor.
  const lifted = _liftAggregateExprN(exprIR, canonicalAxes, axisLengths,
                                     env, atomBatchedNames, N);

  // Materialise singleton dims to their full length (matches the
  // non-batched path).
  const fullShape = canonicalAxes.map((a) => axisLengths[a]);
  const full = _broadcastTo(lifted, fullShape);

  const outShape = [N, ...outAxes.map((a: string) => axisLengths[a])];
  const outSize = _shapeProd(outShape);
  if (reduceAxes.length === 0) {
    // No reduction — already shape [N, ...outAxes].
    return { shape: outShape, data: full.data };
  }
  const reduceSize = _shapeProd(reduceAxes.map((a: string) => axisLengths[a]));
  const outData = new Float64Array(outSize);
  // Fast paths for sum / mean / prod: inline tight reduce loop
  // (no tmp-copy + no per-cell function call). The generic path
  // (tmp + _applyAggregateReduction) handles var / std / min / max
  // which need either two passes or a per-cell function call anyway.
  const fdata = full.data;
  if (fname === 'sum') {
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      let acc = 0;
      for (let j = 0; j < reduceSize; j++) acc += fdata[base + j];
      outData[i] = acc;
    }
  } else if (fname === 'mean') {
    const invN = 1 / reduceSize;
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      let acc = 0;
      for (let j = 0; j < reduceSize; j++) acc += fdata[base + j];
      outData[i] = acc * invN;
    }
  } else if (fname === 'prod') {
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      let acc = 1;
      for (let j = 0; j < reduceSize; j++) acc *= fdata[base + j];
      outData[i] = acc;
    }
  } else {
    const tmp = new Float64Array(reduceSize);
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      for (let j = 0; j < reduceSize; j++) tmp[j] = fdata[base + j];
      outData[i] = _applyAggregateReduction(fname!, tmp, reduceSize);
    }
  }
  return { shape: outShape, data: outData };
}

// Walk the body and infer axis lengths for atom-batched context.
// Same logic as `_inferAggregateAxisLengths` but with awareness that
// containers may be atom-batched. A container is atom-batched ONLY
// when its top-level ref is in `atomBatchedNames` (the refArrays
// keys); shape coincidence (e.g. an atom-indep length-N vector when
// N matches the atom count) is NOT treated as atom-batched.
function _inferAxisLengthsN(
  exprIR: any, axisNames: string[], lengths: Record<string, number>,
  env: any, atomBatchedNames: Set<string>, N: number,
) {
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (n.kind === 'call' && n.op === 'aggregate') return;
    if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
      const args = n.args || [];
      const container = args[0];
      const sels = args.slice(1);
      let containerVal: any = undefined;
      let containerShape: number[] | null = null;
      for (let k = 0; k < sels.length; k++) {
        const s = sels[k];
        if (s && s.kind === 'axis' && !(s.name in lengths)) {
          if (containerVal === undefined) {
            try { containerVal = evaluateExpr(container, env); }
            catch (_) { containerVal = null; }
            if (containerVal != null) {
              containerShape = _shapeOfContainer(
                containerVal, container, atomBatchedNames, N);
            }
          }
          if (containerShape && typeof containerShape[k] === 'number') {
            lengths[s.name] = containerShape[k];
          }
        }
      }
    }
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op' || k === 'name' || k === 'ns') continue;
      walk((n as any)[k]);
    }
  }
  walk(exprIR);
  const missing = axisNames.filter((a) => !(a in lengths));
  if (missing.length > 0) {
    throw new Error(
      `aggregateN: could not infer length of axis ${missing.map((a) => '.' + a).join(', ')} ` +
      `— each axis must index a known array at least once in expr`);
  }
}

// Inspect a container value and return its PER-ATOM shape.
// "Atom-batched" is determined by ref-name membership in
// `atomBatchedNames`, NOT by shape sniffing (an atom-indep array of
// shape [N] would otherwise be mis-classified when its length
// coincides with the atom count).
//
// - If the container is a self-ref whose name is in
//   `atomBatchedNames` AND the value's shape[0] === N → return
//   shape.slice(1) (the per-atom shape).
// - Otherwise → return shape verbatim.
function _shapeOfContainer(
  val: any, containerIR: any, atomBatchedNames: Set<string>, N: number,
): number[] | null {
  if (val == null) return null;
  if (typeof val === 'number' || typeof val === 'boolean') return [];
  let shape: number[] | null = null;
  if (val.BYTES_PER_ELEMENT !== undefined) {
    shape = [val.length];
  } else if (Array.isArray(val)) {
    shape = [val.length];
    let probe = val[0];
    while (Array.isArray(probe)
           || (probe && probe.BYTES_PER_ELEMENT !== undefined)) {
      shape.push(probe.length);
      probe = probe[0];
    }
  } else if (val && Array.isArray(val.shape)) {
    shape = val.shape.slice();
  }
  if (!shape) return null;
  // Atom-batched only if the container is a direct ref to an atom-
  // batched name (i.e. an entry in refArrays). Strip the leading N.
  const isAtomBatched = containerIR
    && containerIR.kind === 'ref'
    && containerIR.ns === 'self'
    && atomBatchedNames.has(containerIR.name)
    && shape.length > 0 && shape[0] === N;
  if (isAtomBatched) return shape.slice(1);
  return shape;
}

function _liftAggregateExprN(
  node: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
  atomBatchedNames: Set<string>, N: number,
): { data: Float64Array; shape: number[] } {
  // Constant-hoist (no axis refs): evaluate once. The result may be:
  //  - A scalar (atom-indep) → broadcast singleton everywhere.
  //  - A direct atom-batched ref to a rank-1 [N] scalar → broadcast
  //    across non-atom axes.
  // We detect atom-batched only by NAME (membership in
  // atomBatchedNames), not shape coincidence.
  if (!_containsAxisRef(node)) {
    const v = evaluateExpr(node, env);
    return _alignedConstTensorN(v, node, canonicalAxes, axisLengths,
                                atomBatchedNames, N);
  }
  if (node.kind === 'call' && (node.op === 'get' || node.op === 'get0')) {
    return _alignedTensorFromGetN(node, canonicalAxes, axisLengths, env,
                                  atomBatchedNames, N);
  }
  if (node.kind === 'call' && node.args && node.args.length === 2) {
    const fn = _AGG_BIN[node.op];
    if (fn) {
      const a = _liftAggregateExprN(node.args[0], canonicalAxes,
                                     axisLengths, env, atomBatchedNames, N);
      const b = _liftAggregateExprN(node.args[1], canonicalAxes,
                                     axisLengths, env, atomBatchedNames, N);
      return _broadcastBinary(a, b, fn);
    }
  }
  if (node.kind === 'call' && node.args && node.args.length === 1) {
    const fn = _AGG_UN[node.op];
    if (fn) {
      const a = _liftAggregateExprN(node.args[0], canonicalAxes,
                                     axisLengths, env, atomBatchedNames, N);
      return _broadcastUnary(a, fn);
    }
  }
  throw new Error(`aggregateN: unsupported op '${node.op || node.kind}' `
    + `in body — broadcast-reduce default supports arithmetic and `
    + `unary math on indexed arrays.`);
}

// Constant subtree (no axis refs). The evaluated value may be:
//   - Scalar (atom-indep) → broadcast singleton everywhere.
//   - Atom-batched scalar (direct ref to a per-atom rank-1 Value
//     with shape=[N]) → broadcast across non-atom axes.
function _alignedConstTensorN(
  v: any, node: any, canonicalAxes: string[],
  axisLengths: Record<string, number>,
  atomBatchedNames: Set<string>, N: number,
): { data: Float64Array; shape: number[] } {
  const shape = canonicalAxes.map(() => 1);
  // Direct atom-batched ref to a per-atom scalar value: name in
  // atomBatchedNames AND the value's shape is [N].
  const isDirectAtomRef = node && node.kind === 'ref'
    && node.ns === 'self' && atomBatchedNames.has(node.name);
  if (isDirectAtomRef) {
    let perAtom: Float64Array | null = null;
    if (v && v.BYTES_PER_ELEMENT !== undefined && v.length === N) {
      perAtom = v as Float64Array;
    } else if (v && Array.isArray(v.shape) && v.shape.length === 1
               && v.shape[0] === N) {
      perAtom = v.data as Float64Array;
    }
    if (perAtom) {
      shape[0] = N;
      return { data: perAtom, shape };
    }
  }
  return { data: new Float64Array([+v]), shape };
}

// Atom-batched-aware `get` lifter. Container value may be:
//   - atom-batched (direct ref to a name in `atomBatchedNames`,
//     value shape[0] === N) → leading dim is atom axis; the
//     remaining dims correspond to the get's selector positions.
//   - atom-indep → broadcasts across atom axis (singleton stride).
//
// Critical: "atom-batched" is determined by NAME, not shape. An
// atom-indep array of shape [N] (coincidental) would be wrongly
// classified by shape alone — driving the lifter to strip an axis
// that doesn't exist, and trip subsequent axis-length inference.
function _alignedTensorFromGetN(
  getIR: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
  atomBatchedNames: Set<string>, N: number,
): { data: Float64Array; shape: number[] } {
  const args = getIR.args || [];
  const oneBased = getIR.op === 'get';
  const containerIR = args[0];
  const arr = evaluateExpr(containerIR, env);
  const sels = args.slice(1);
  const src = _toFlat(arr);
  // Detect atom-batched container by REF NAME (not shape).
  const atomBatched = containerIR
    && containerIR.kind === 'ref'
    && containerIR.ns === 'self'
    && atomBatchedNames.has(containerIR.name)
    && src.shape.length > 0 && src.shape[0] === N
    && src.shape.length === sels.length + 1;
  // Source dims that correspond to the get's selector positions.
  // For atom-indep: dims = src.shape (length === sels.length).
  // For atom-batched: dims = src.shape.slice(1).
  const selDims = atomBatched ? src.shape.slice(1) : src.shape;
  const sourceStrides = _rowMajorStrides(src.shape);
  // Stride for the atom dim (offset between atom-i and atom-(i+1)):
  //   - atom-batched: product of remaining dims (the per-atom slice).
  //   - atom-indep: 0 (the same atom-indep value broadcasts to every
  //     atom — singleton stride).
  const atomStride = atomBatched ? sourceStrides[0] : 0;
  // Per-sel-dim stride into src.shape (after the atom dim if any).
  const selStrides = atomBatched ? sourceStrides.slice(1) : sourceStrides;

  // Resolve selectors.
  const axisAt: Record<string, number> = {};
  let baseOffset = 0;
  for (let k = 0; k < sels.length; k++) {
    const s = sels[k];
    if (s && s.kind === 'axis') {
      axisAt[s.name] = k;
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'only') {
      if (selDims[k] !== 1) {
        throw new Error(`aggregateN: 'only' selector requires the indexed `
          + `axis to have length 1, got length ${selDims[k]}`);
      }
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'all') {
      throw new Error(`aggregateN: 'all' / ':' is not supported in an `
        + `aggregate body; use an axis name (.name) instead`);
    }
    const idx = +evaluateExpr(s, env);
    const idx0 = oneBased ? (idx | 0) - 1 : (idx | 0);
    if (idx0 < 0 || idx0 >= selDims[k]) {
      throw new Error(`aggregateN: index ${oneBased ? idx : idx0} out of bounds `
        + `for axis of length ${selDims[k]}`);
    }
    baseOffset += idx0 * selStrides[k];
  }

  // Build aligned shape + per-canonical-position source stride.
  // The atom axis at position 0 → atomStride; other canonical axes →
  // selStrides if the get binds them, else singleton (stride 0).
  const M = canonicalAxes.length;
  const alignedShape = new Array(M);
  const sourceStrideAt: number[] = new Array(M);
  const ATOM_AXIS = canonicalAxes[0];
  for (let i = 0; i < M; i++) {
    const axisName = canonicalAxes[i];
    if (axisName === ATOM_AXIS) {
      alignedShape[i] = N;
      sourceStrideAt[i] = atomStride;
      continue;
    }
    if (axisName in axisAt) {
      alignedShape[i] = axisLengths[axisName];
      sourceStrideAt[i] = selStrides[axisAt[axisName]];
    } else {
      alignedShape[i] = 1;
      sourceStrideAt[i] = 0;
    }
  }

  // Materialise the aligned tensor. Rank-specific fast paths (1-3)
  // beat the generic coord-walker by ~10× because V8 JITs the
  // nested-loop form much better than the per-element coord-stride
  // recomputation. Rank 3 is the hot case for fusion (a) Step 2's
  // canonical output (atom × outer × reduce axes).
  const alignedSize = _shapeProd(alignedShape);
  const out = new Float64Array(alignedSize);
  const srcData = src.data;
  if (M === 3) {
    const D0 = alignedShape[0], D1 = alignedShape[1], D2 = alignedShape[2];
    const S0 = sourceStrideAt[0], S1 = sourceStrideAt[1], S2 = sourceStrideAt[2];
    let idx = 0;
    for (let i0 = 0; i0 < D0; i0++) {
      const off0 = baseOffset + i0 * S0;
      for (let i1 = 0; i1 < D1; i1++) {
        const off1 = off0 + i1 * S1;
        for (let i2 = 0; i2 < D2; i2++) {
          out[idx++] = srcData[off1 + i2 * S2];
        }
      }
    }
    return { data: out, shape: alignedShape };
  }
  if (M === 2) {
    const D0 = alignedShape[0], D1 = alignedShape[1];
    const S0 = sourceStrideAt[0], S1 = sourceStrideAt[1];
    let idx = 0;
    for (let i0 = 0; i0 < D0; i0++) {
      const off0 = baseOffset + i0 * S0;
      for (let i1 = 0; i1 < D1; i1++) {
        out[idx++] = srcData[off0 + i1 * S1];
      }
    }
    return { data: out, shape: alignedShape };
  }
  if (M === 1) {
    const D0 = alignedShape[0], S0 = sourceStrideAt[0];
    for (let i0 = 0; i0 < D0; i0++) out[i0] = srcData[baseOffset + i0 * S0];
    return { data: out, shape: alignedShape };
  }
  if (M === 0) {
    out[0] = srcData[baseOffset];
    return { data: out, shape: alignedShape };
  }
  // Generic fallback (rank ≥4).
  const coord = new Array(M).fill(0);
  for (let linear = 0; linear < alignedSize; linear++) {
    let srcOff = baseOffset;
    for (let i = 0; i < M; i++) srcOff += coord[i] * sourceStrideAt[i];
    out[linear] = srcData[srcOff];
    for (let i = M - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < alignedShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: alignedShape };
}

module.exports = {
  _inferAggregateAxisLengths,
  _collectInScopeAxisNames,
  AGGREGATE_PATTERNS,
  _evalAggregate,
  _evalAggregateBroadcastReduceN,
  _matmulDispatch,
};
