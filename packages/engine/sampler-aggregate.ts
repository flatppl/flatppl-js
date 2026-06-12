'use strict';

// =====================================================================
// sampler-aggregate.ts — aggregate(f_reduction, output_axes, expr)
// (spec §04 §sec:aggregate; engine-concepts §16)
// =====================================================================
//
// Extracted from sampler.ts as part of the sampler split
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
const aggregateShape = require('./aggregate-shape.ts');

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

// Engine-internal helper: extract the runtime shape of a value an
// aggregate body references. Returns null when the value is unknown
// or scalar-only (no shape to read). Used by the runtime axis-length
// resolver, ONLY for axes typeinfer left as `%dynamic`.
function _runtimeShapeOf(val: any): number[] | null {
  if (val == null) return null;
  if (typeof val === 'number' || typeof val === 'boolean') return [];
  if (Array.isArray(val)) {
    const inner: number[] = [];
    if (val.length > 0 && Array.isArray(val[0])) {
      const tail = _runtimeShapeOf(val[0]);
      if (tail) for (const x of tail) inner.push(x);
    }
    return [val.length, ...inner];
  }
  if (val && (val as any).BYTES_PER_ELEMENT) return [(val as any).length];
  if (val && Array.isArray((val as any).shape)) return (val as any).shape;
  return null;
}

// Build a per-axis length resolver: walks `get(arr, .axis, ...)`
// calls in the body to read the container's dim-k length whenever
// the resolver is asked for an axis not already resolved by
// typeinfer's annotation.
//
// In atom-batched contexts (`atomCfg != null`), containers may be
// atom-batched (the container's leading dim is the atom axis = N,
// stripped before reading the selector-dim length); detection is
// by REF NAME (membership in atomCfg.atomBatchedNames), not shape
// coincidence — see `_alignedTensorFromGet` for the rationale.
function _makeRuntimeAxisLengthResolver(
  exprIR: any, env: any, atomCfg: AtomConfig,
) {
  // Memoise so repeated `.has(name)` queries don't re-walk the body.
  const resolved: Record<string, number | null> = {};
  return function lengthOf(axisName: string): number | null {
    if (axisName in resolved) return resolved[axisName];
    let found: number | null = null;
    function walk(n: any) {
      if (found != null) return;
      if (!n || typeof n !== 'object') return;
      if (Array.isArray(n)) { for (const c of n) walk(c); return; }
      if (n.kind === 'call' && n.op === 'aggregate') return;   // inner scope
      if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
        const args = n.args || [];
        const container = args[0];
        const sels = args.slice(1);
        let selDims: number[] | null = null;
        let resolvedOnce = false;
        for (let k = 0; k < sels.length; k++) {
          const s = sels[k];
          if (s && s.kind === 'axis' && s.name === axisName) {
            if (!resolvedOnce) {
              resolvedOnce = true;
              let containerVal: any = null;
              try { containerVal = evaluateExpr(container, env); }
              catch (_) { containerVal = null; }
              const shape = _runtimeShapeOf(containerVal);
              if (shape) {
                // In atom-batched mode, strip the leading atom dim
                // ONLY when the container is an atom-batched ref by
                // NAME (atom-indep coincidental shape[0] === N is
                // NOT atom-batched).
                const atomBatched = !!(atomCfg && container
                  && container.kind === 'ref'
                  && container.ns === 'self'
                  && atomCfg.atomBatchedNames.has(container.name)
                  && shape.length > 0 && shape[0] === atomCfg.N
                  && shape.length === sels.length + 1);
                selDims = atomBatched ? shape.slice(1) : shape;
              }
            }
            if (selDims && typeof selDims[k] === 'number') {
              found = selDims[k];
              return;
            }
          }
        }
      }
      for (const k of Object.keys(n)) {
        if (k === 'loc' || k === 'kind' || k === 'op'
            || k === 'name' || k === 'ns') continue;
        walk(n[k]);
      }
    }
    walk(exprIR);
    resolved[axisName] = found;
    return found;
  };
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
//
// When `atomN` is set (atom-batched aggregate path), the mul dispatch
// passes `{ atomN }` so the rank-2 × rank-2 atom-aware variant
// (`_matBatchedMatMul`, registered in ops-declarations.ts) fires for
// per-atom matmul — shape `[N, m, n] × [N, n, p] → [N, m, p]`. Without
// `atomN`, the dispatcher treats those as rank-3 × rank-3 (no
// matching variant) and throws, which `_tryBatchedAggregatePatterns`
// catches as a fall-through signal.
// ---------------------------------------------------------------------
function _matmulDispatch(
  A: any, B: any, transA: boolean, transB: boolean, atomN?: number,
): any {
  if (valueLib.isValue(A) || valueLib.isValue(B)) {
    let aV = valueLib.asValue(A);
    let bV = valueLib.asValue(B);
    if (transA) aV = valueLib.transpose(aV);
    if (transB) bV = valueLib.transpose(bV);
    if (typeof atomN === 'number') {
      const ops = require('./ops.ts');
      return ops.dispatch('mul', [aV, bV], { atomN, wrappingOp: 'direct' });
    }
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
// P5: classification delegates to the shared `aggregate-patterns.ts`
// module — same source of truth as the dissolver's matmul matchers.
// The `execute` half stays here (runtime evaluation specific).
const aggregatePatterns = require('./aggregate-patterns.ts');

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
    const cls = aggregatePatterns.classifyMatmulBody(bodyIR, outAxes);
    if (!cls || cls.kind !== 'matmul') return null;
    return { aIR: cls.aIR, bIR: cls.bIR, transA: cls.transA, transB: cls.transB };
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    return _matmulDispatch(A, B, match.transA, match.transB,
      env && env.__atomN);
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
    const cls = aggregatePatterns.classifyMatmulBody(bodyIR, outAxes);
    if (!cls || cls.kind !== 'matvec') return null;
    return { aIR: cls.aIR, vIR: cls.vIR, transA: cls.transA };
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const v = evaluateExpr(match.vIR, env);
    if (valueLib.isValue(A) || valueLib.isValue(v)) {
      // _matmulDispatch threads env.__atomN (atom-batched harness) so
      // an atom-batched v (shape=[N, n]) fires the registered
      // mul(rank-2, rank-1) atom-aware variant (_matBatchedVecMul)
      // instead of mis-matching the direct mat×mat variant — which
      // throws on n ≠ N, and at n === N would have computed a WRONG
      // plain matrix product. All-atom-indep operands keep the direct
      // valueOps.mul path (the atom-aware matcher requires ≥1
      // atom-batched arg).
      return _matmulDispatch(A, v, match.transA, false, env && env.__atomN);
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
    const cls = aggregatePatterns.classifyMatmulBody(bodyIR, outAxes);
    if (!cls || cls.kind !== 'outer') return null;
    return { uIR: cls.uIR, vIR: cls.vIR };
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
// Specialiser: dot product (full reduction, zero output axes)
//
//   aggregate(sum, [], u[.j] * v[.j])   ≡  Σⱼ uⱼ·vⱼ
//
// — the spec's `s[] := u[.i] * v[.i]` shorthand. One fused
// multiply-accumulate loop instead of the broadcast-reduce default's
// elementwise-product intermediate (the alloc save). Returns a plain
// JS number, exactly like the default's scalar full-reduction output.
// Anything beyond flat real rank-1 operands (complex, diag-stored,
// transposed views, higher rank) falls back to the default, which
// owns those semantics and diagnostics.
//
// SINGLE-POINT ONLY: the matcher refuses the atom-batched harness
// (`env.__atomN`). In that context a per-atom scalar ref is a Value
// of shape=[N] — indistinguishable by shape from an atom-indep
// length-N vector — so a fused flat-buffer dot would silently
// multiply a per-atom column as if it were vector elements. The
// generic `_evalAggregateBroadcastReduceN` owns the batched case.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'dot-product',
  match(ir: any, env: any): any {
    if (env && typeof env.__atomN === 'number') return null;  // batched: generic only
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 0) return null;
    const cls = aggregatePatterns.classifyMatmulBody(bodyIR, outAxes);
    if (!cls || cls.kind !== 'dot') return null;
    return { uIR: cls.uIR, vIR: cls.vIR };
  },
  execute(ir: any, env: any, match: any): any {
    const u = evaluateExpr(match.uIR, env);
    const v = evaluateExpr(match.vIR, env);
    // Resolve each operand to a flat numeric buffer, or bail.
    const buf = (x: any): Float64Array | number[] | null => {
      if (Array.isArray(x)) {
        for (let i = 0; i < x.length; i++) {
          if (typeof x[i] !== 'number') return null;
        }
        return x;
      }
      if (x instanceof Float64Array) return x;
      if (valueLib.isValue(x)) {
        if (x.shape.length !== 1) return null;        // rank-1 only
        if (x.im || x.dtype === 'complex') return null;
        if (x.t && x.t !== 'N') return null;          // no transposed views
        if (valueLib.isDiagStored && valueLib.isDiagStored(x)) return null;
        return x.data;
      }
      return null;
    };
    const ub = buf(u), vb = buf(v);
    if (!ub || !vb || ub.length !== vb.length) {
      // Unsupported shape (or a length mismatch — the default raises
      // the canonical aggregate diagnostic for it).
      return _evalAggregateBroadcastReduce(ir, env);
    }
    let s = 0;
    for (let i = 0; i < ub.length; i++) s += ub[i] * vb[i];
    return s;
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
  execute(ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    // Nested-array operands only. A shape-explicit Value has no
    // useful `.length` (the per-batch loop below would read
    // `A.length === undefined` and return garbage), so Values route
    // to the generic broadcast-reduce, which owns rank-3 Value
    // semantics. In the atom-batched harness (`env.__atomN`) we must
    // not return the SINGLE-POINT generic — throw instead, so
    // `_tryBatchedAggregatePatterns` catches and falls through to
    // `_evalAggregateBroadcastReduceN`.
    if (valueLib.isValue(A) || valueLib.isValue(B)) {
      if (env && typeof env.__atomN === 'number') {
        throw new Error('batched-matmul specialiser: Value operands in '
          + 'atom-batched context — fall through to the generic lowering');
      }
      return _evalAggregateBroadcastReduce(ir, env);
    }
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
    return _flatToValueSP(outData, outDimSizes);
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

// Single-point aggregate result packer. Rank-0/1 keep their existing
// scalar / Float64Array form (unchanged downstream + materialiser
// behaviour); rank >= 2 returns a shape-explicit Value {shape, data},
// MATCHING the atom-batched path (`return { shape: outShape, data }`)
// and the matmul specialiser. A nested JS array here would otherwise
// materialise as an {elems} tuple-of-row-measures rather than a matrix
// (materialiser-shared.fixedValueToMeasure). `_toFlat` already accepts
// both Values and nested arrays, so downstream consumers are unaffected.
function _flatToValueSP(data: Float64Array, shape: number[]): any {
  if (shape.length >= 2) return { shape: shape.slice(), data };
  return _flatToNested(data, shape);
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

// =====================================================================
// Unified aligned-tensor lifter (P2 — single-point AND atom-batched)
// =====================================================================
//
// `atomCfg` is `null` for the single-point aggregate evaluator
// (canonicalAxes contains only user axes) and `{N, atomBatchedNames,
// atomAxis}` for the atom-batched evaluator (canonicalAxes prepends
// `atomAxis` to the user-axis ordering). The merge keeps the
// single-point and atom-batched semantics in lockstep:
//   - atom-batched detection is by REF NAME (membership in
//     atomBatchedNames), never by shape coincidence — spec §04 line
//     853's length-uniqueness requirement makes shape-sniffing wrong
//     when an atom-indep array's length coincidentally equals N.
//   - When atomCfg is null, every atom-axis check short-circuits.
//
// Before P2, this function existed twice (`_alignedTensorFromGet` +
// `_alignedTensorFromGetN`) with drift: the N version had rank-1/2/3
// JIT-friendly nested loops that the non-N version lacked; the
// repeated-axis (einsum 'ii') sum-of-strides fix had to land in both.
type AtomConfig = {
  N: number;
  atomBatchedNames: Set<string>;
  atomAxis: string;     // typically '__atom_N'
} | null;

function _alignedTensorFromGet(
  getIR: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
  atomCfg: AtomConfig,
): { data: Float64Array; shape: number[] } {
  const args = getIR.args || [];
  const oneBased = getIR.op === 'get';
  const containerIR = args[0];
  const arr = evaluateExpr(containerIR, env);
  const sels = args.slice(1);
  const src = _toFlat(arr);
  const sourceStrides = _rowMajorStrides(src.shape);

  // Atom-batched detection by REF NAME. The container's leading dim
  // becomes the atom axis; remaining dims correspond to the get's
  // selector positions. When atomCfg is null, this short-circuits to
  // false and the function behaves as the single-point lifter.
  const atomBatched: boolean = !!(atomCfg && containerIR
    && containerIR.kind === 'ref'
    && containerIR.ns === 'self'
    && atomCfg.atomBatchedNames.has(containerIR.name)
    && src.shape.length > 0 && src.shape[0] === atomCfg.N
    && src.shape.length === sels.length + 1);
  const selDims = atomBatched ? src.shape.slice(1) : src.shape;
  // Stride for the atom dim (offset between atom-i and atom-(i+1)):
  //   - atom-batched: product of remaining dims (the per-atom slice).
  //   - atom-indep:   0 (the same atom-indep value broadcasts across
  //     every atom — singleton stride).
  const atomStride = atomBatched ? sourceStrides[0] : 0;
  // Per-sel-dim stride into src.shape (after the atom dim if any).
  const selStrides = atomBatched ? sourceStrides.slice(1) : sourceStrides;

  // Walk selectors. For each, decide:
  //   - axis ref: this dim corresponds to a canonical axis. The
  //     SAME axis name may appear in multiple selector positions
  //     (e.g. `A[.i, .i]` for trace) — spec §04 §sec:aggregate:
  //     "All array dimensions indexed with the same axis name must
  //     have the same length." When axis .i indexes both dim 0 and
  //     dim 1, the lifter must advance ONE coordinate across BOTH
  //     source dims simultaneously, so we accumulate strides (sum of
  //     all per-position source strides for that axis). The repeated
  //     axis is the standard einsum repeated-index pattern
  //     (np.einsum('ii->...', A)).
  //   - integer / 'only': collapse this dim (advance baseOffset).
  //   - 'all': not supported in aggregate body (per §04 spec the body
  //     uses axis names; `all`/':' would mean "keep an unnamed dim"
  //     which has no defined semantics for the contraction).
  const axisStrideSum: Record<string, number> = {};
  let baseOffset = 0;
  for (let k = 0; k < sels.length; k++) {
    const s = sels[k];
    if (s && s.kind === 'axis') {
      axisStrideSum[s.name] = (axisStrideSum[s.name] || 0) + selStrides[k];
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'only') {
      if (selDims[k] !== 1) {
        throw new Error(`aggregate: 'only' selector requires the indexed `
          + `axis to have length 1, got length ${selDims[k]}`);
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
    if (idx0 < 0 || idx0 >= selDims[k]) {
      throw new Error(`aggregate: index ${oneBased ? idx : idx0} out of bounds `
        + `for axis of length ${selDims[k]}`);
    }
    baseOffset += idx0 * selStrides[k];
  }

  // Build the aligned shape + a per-canonical-position source stride.
  // - atom axis (when atomCfg != null) → atomStride (or 0 atom-indep).
  // - user axis present in this get → axisStrideSum (sum-of-strides
  //   handles repeated occurrences / trace pattern).
  // - user axis not present here → singleton (stride 0, broadcast).
  const M = canonicalAxes.length;
  const alignedShape = new Array(M);
  const sourceStrideAt: number[] = new Array(M);
  for (let i = 0; i < M; i++) {
    const axisName = canonicalAxes[i];
    if (atomCfg && axisName === atomCfg.atomAxis) {
      alignedShape[i] = atomCfg.N;
      sourceStrideAt[i] = atomStride;
      continue;
    }
    if (axisName in axisStrideSum) {
      alignedShape[i] = axisLengths[axisName];
      sourceStrideAt[i] = axisStrideSum[axisName];
    } else {
      alignedShape[i] = 1;
      sourceStrideAt[i] = 0;
    }
  }

  // Materialise the aligned tensor. Rank-specific fast paths (0-3)
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
  // Generic coord-walker for rank ≥4.
  const coord = new Array(M).fill(0);
  for (let linear = 0; linear < alignedSize; linear++) {
    let srcOff = baseOffset;
    for (let i = 0; i < M; i++) srcOff += coord[i] * sourceStrideAt[i];
    out[linear] = srcData[srcOff];
    // Increment coord, last dim fastest.
    for (let i = M - 1; i >= 0; i--) {
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
//
// Unified P2 lifter — `atomCfg` is null for single-point evaluation
// and `{N, atomBatchedNames, atomAxis}` for the atom-batched
// evaluator. The constant-subtree branch handles BOTH:
//   - true constants (atom-indep, no axis refs) → singleton tensor;
//   - direct atom-batched refs to per-atom rank-1 [N] values (only
//     reachable when atomCfg != null) → atom-axis-stretched tensor.
function _liftAggregateExpr(
  node: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
  atomCfg: AtomConfig,
): { data: Float64Array; shape: number[] } {
  // Constant-hoist: subtree with no axis refs evaluates once.
  if (!_containsAxisRef(node)) {
    const v = evaluateExpr(node, env);
    return _alignedConstTensor(v, node, canonicalAxes, atomCfg);
  }
  // `get` / `get0`: produces an aligned tensor at this leaf.
  if (node.kind === 'call' && (node.op === 'get' || node.op === 'get0')) {
    return _alignedTensorFromGet(node, canonicalAxes, axisLengths, env, atomCfg);
  }
  // Binary arithmetic: lift both operands, broadcast-apply.
  if (node.kind === 'call' && node.args && node.args.length === 2) {
    const fn = _AGG_BIN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env, atomCfg);
      const b = _liftAggregateExpr(node.args[1], canonicalAxes, axisLengths, env, atomCfg);
      return _broadcastBinary(a, b, fn);
    }
  }
  // Unary scalar math.
  if (node.kind === 'call' && node.args && node.args.length === 1) {
    const fn = _AGG_UN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env, atomCfg);
      return _broadcastUnary(a, fn);
    }
  }
  throw new Error(`aggregate: unsupported op '${node.op || node.kind}' `
    + `in body — broadcast-reduce default supports arithmetic and `
    + `unary math on indexed arrays. Hoist non-broadcasting subterms `
    + `to bindings outside the aggregate.`);
}

// Constant subtree (no axis refs). The evaluated value may be:
//   - Scalar (atom-indep) → broadcast singleton everywhere.
//   - When atomCfg != null and `node` is a direct ref to an atom-
//     batched name, AND v is a per-atom rank-1 [N] value → broadcast
//     across non-atom axes (atom axis takes the per-atom data).
function _alignedConstTensor(
  v: any, node: any, canonicalAxes: string[], atomCfg: AtomConfig,
): { data: Float64Array; shape: number[] } {
  const shape = canonicalAxes.map(() => 1);
  if (atomCfg) {
    const isDirectAtomRef = node && node.kind === 'ref'
      && node.ns === 'self' && atomCfg.atomBatchedNames.has(node.name);
    if (isDirectAtomRef) {
      let perAtom: Float64Array | null = null;
      if (v && v.BYTES_PER_ELEMENT !== undefined && v.length === atomCfg.N) {
        perAtom = v as Float64Array;
      } else if (v && Array.isArray(v.shape) && v.shape.length === 1
                 && v.shape[0] === atomCfg.N) {
        perAtom = v.data as Float64Array;
      }
      if (perAtom) {
        // Atom axis is always at position 0 in atom-batched canonical
        // ordering (the evaluator prepends atomAxis to user axes).
        shape[0] = atomCfg.N;
        return { data: perAtom, shape };
      }
    }
  }
  return { data: new Float64Array([+v]), shape };
}

// =====================================================================
// Unified aggregate broadcast-reduce evaluator (P2)
// =====================================================================
//
// `_evalAggregateGeneric(ir, env, atomCfg)` runs the spec §04 §16
// permute-reshape-broadcast-reduce lowering for ANY aggregate IR
// (single-point when atomCfg=null, atom-batched when atomCfg
// supplies {N, refArrays, baseEnv, overlay}). Before P2 this lived
// in TWO functions (`_evalAggregateBroadcastReduce` +
// `_evalAggregateBroadcastReduceN`) that re-implemented the same
// pipeline; the atom-batched version had JIT fast paths and inline
// sum/mean/prod reductions the single-point one lacked. Now there
// is ONE pipeline; the single-point path inherits the fast paths.
//
// `atomCfg` shape:
//   null              → single-point: canonicalAxes from typeinfer's
//                        annotation; lengths via runtime resolver; no
//                        atom axis.
//   { N, refArrays, baseEnv, overlay } → atom-batched: prepend an
//                        atom-axis entry to canonicalAxes; build an
//                        env from baseEnv + refArrays + overlay; the
//                        atomBatchedNames set is the refArrays keys.
//
// SOTA alignment: JAX vmap composition — the *same* primitive rule
// applies per axis level (no separate add_batched vs add). Our atom
// axis is just one more axis in canonicalAxes; treating it specially
// at the evaluator level was the smell P2 fixes.

type AggregateAtomConfig = {
  N: number;
  refArrays: Record<string, any>;
  baseEnv: any;
  overlay: any;
} | null;

function _evalAggregateGeneric(
  ir: any, env: any, atomCfg: AggregateAtomConfig,
): any {
  const args = ir.args || [];
  const isBatched = atomCfg !== null;
  const tag = isBatched ? 'aggregateN' : 'aggregate';
  if (args.length !== 3) {
    throw new Error(`${tag}: expected 3 args, got ${args.length}`);
  }
  const [fIR, , exprIR] = args;

  const fname = (fIR.kind === 'ref' && fIR.name)
              || (fIR.kind === 'const' && fIR.name);
  const reduce = fname && _AGGREGATE_REDUCTIONS[fname];
  if (!reduce) throw new Error(`${tag}: unknown reduction '${fname}'`);

  // P1: read the canonical form (typeinfer populated it; runtime
  // resolves any %dynamic lengths via the body walker).
  const canonical = aggregateShape.getCanonical(ir);
  if (!canonical) {
    throw new Error(`${tag}: output_axes must be an array literal of axis names`);
  }
  const { outAxes, reduceAxes } = canonical;

  // Build the runtime env + atom-axis prepended canonical ordering.
  let runEnv: any;
  let liftCfg: AtomConfig;
  let atomAxis = '';
  if (isBatched) {
    // The atom axis name uses a double-underscore prefix to avoid
    // colliding with any user axis (spec §05 axis labels start with
    // a letter).
    atomAxis = '__atom_N';
    const { N, refArrays, baseEnv, overlay } = atomCfg!;
    runEnv = Object.assign({}, baseEnv || {});
    const atomBatchedNames: Set<string> = new Set();
    if (refArrays) {
      for (const k of Object.keys(refArrays)) {
        runEnv[k] = refArrays[k];
        atomBatchedNames.add(k);
      }
    }
    if (overlay) {
      for (const k of Object.keys(overlay)) {
        runEnv[k] = overlay[k];
        // Overlay overrides: if a name was atom-batched but overlay
        // overrides, treat the new value at face value.
        atomBatchedNames.delete(k);
      }
    }
    liftCfg = { N, atomBatchedNames, atomAxis };
  } else {
    runEnv = env;
    liftCfg = null;
  }
  const canonicalAxes = isBatched
    ? [atomAxis, ...outAxes, ...reduceAxes]
    : [...outAxes, ...reduceAxes];

  // Resolve axis lengths. The atom axis (when present) is N from
  // atomCfg; other axes use the canonical annotation's static
  // lengths, with %dynamic entries resolved against runEnv via the
  // runtime body walker.
  const lengths: Record<string, number> = isBatched ? { [atomAxis]: atomCfg!.N } : {};
  const resolved = aggregateShape.resolveAxisLengths(
    ir, canonical, _makeRuntimeAxisLengthResolver(exprIR, runEnv, liftCfg));
  for (const a of Object.keys(resolved)) lengths[a] = resolved[a];

  // Lift the body to an aligned tensor over the full canonical shape.
  const lifted = _liftAggregateExpr(exprIR, canonicalAxes, lengths, runEnv, liftCfg);

  // Output shape: drop reduce axes. Atom-batched output is a Value
  // {shape, data} with leading atom dim; single-point returns the
  // nested-array / scalar form via _flatToNested.
  const outAxesFull = isBatched ? [atomAxis, ...outAxes] : outAxes;
  const outShape = outAxesFull.map((a: string) => lengths[a]);
  const reduceShape = reduceAxes.map((a: string) => lengths[a]);
  const outSize = _shapeProd(outShape);

  // No-reduction short-circuit: the output IS the lifted tensor
  // (with any singleton dims broadcast to their full length).
  if (reduceAxes.length === 0) {
    const full = _broadcastTo(lifted, outShape);
    if (isBatched) return { shape: outShape, data: full.data };
    return _flatToValueSP(full.data, outShape);
  }

  // Stretch any remaining singletons to their full length, then
  // tail-reduce over the contiguous trailing reduce block.
  const fullShape = canonicalAxes.map((a) => lengths[a]);
  const full = _broadcastTo(lifted, fullShape);
  const reduceSize = _shapeProd(reduceShape);
  const outData = new Float64Array(outSize);
  const fdata = full.data;
  // Fast paths for sum / mean / prod: tight inline reduce loops
  // (no tmp-copy + no per-cell function call). The generic path
  // (tmp + _applyAggregateReduction) handles var / std / min / max
  // which need either two passes or a per-cell function call anyway.
  if (fname === 'sum') {
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      let acc = 0;
      for (let j = 0; j < reduceSize; j++) acc += fdata[base + j];
      outData[i] = acc;
    }
  } else if (fname === 'mean') {
    const invR = 1 / reduceSize;
    for (let i = 0; i < outSize; i++) {
      const base = i * reduceSize;
      let acc = 0;
      for (let j = 0; j < reduceSize; j++) acc += fdata[base + j];
      outData[i] = acc * invR;
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
  if (isBatched) return { shape: outShape, data: outData };
  return _flatToValueSP(outData, outShape);
}

// Thin shim — single-point callers use this entry; atom-batched
// callers go through `_evalAggregateBroadcastReduceN` below (same
// generic pipeline with `atomCfg != null`).
function _evalAggregateBroadcastReduce(ir: any, env: any): any {
  return _evalAggregateGeneric(ir, env, null);
}

// P6: try atom-batched AGGREGATE_PATTERNS specialisers before falling
// through to the generic broadcast-reduce. Returns the specialiser's
// result on hit; null on miss (caller falls through to
// `_evalAggregateBroadcastReduceN`).
//
// The env carries the atom count under `__atomN`; specialisers thread
// it into `ops.dispatch('mul', ..., { atomN })` (via `_matmulDispatch`
// for matmul-family and matvec) so the atom-aware mul variants fire:
//   - matvec: atom-indep A=[m, n] × atom-batched v=[N, n] →
//     `_matBatchedVecMul` (the linear-regression `X · betas` shape);
//   - matmul: `[N, m, n] × [N, n, p] → [N, m, p]` (and the shared-A /
//     shared-B mixes) → `_matBatchedMatMul` (Phase 2.2, 2026-05-31).
// When no atom-aware variant matches, the dispatcher throws ("no
// variant matched"), which we catch as a fall-through signal. The
// dot-product specialiser refuses `__atomN` envs outright (a per-atom
// scalar column is shape-indistinguishable from a length-N vector);
// the batched-matmul specialiser throws on Value operands here. Both
// fall through to the generic batched lowering.
//
// The wire lets the polyeval-style hot path AND linear-regression-
// style `X · betas` AND any model with `A · v_per_atom` AND batched
// matmul shapes (state-space transition × covariance, etc.) skip the
// generic broadcast-reduce intermediate when a specialiser would
// vectorise better.
function _tryBatchedAggregatePatterns(
  ir: any, refArrays: any, N: number, baseEnv: any, overlay: any,
): any | null {
  if (!perfConfig.getOptimization('aggregate')) return null;
  // Build the same evaluation env the generic pipeline would use —
  // overlay > refArrays > baseEnv — so specialiser execute() sees
  // the atom-batched refArray Values when present.
  const env: Record<string, any> = Object.assign({}, baseEnv || {});
  if (refArrays) {
    for (const k of Object.keys(refArrays)) env[k] = refArrays[k];
  }
  if (overlay) {
    for (const k of Object.keys(overlay)) env[k] = overlay[k];
  }
  // Stash the atom count under a private key so specialisers' execute
  // can propagate it to the variant dispatcher (`ops.dispatch(..., {
  // atomN })`). Specialisers that don't need it just ignore the key.
  // The underscore prefix puts it out of reach of any user binding
  // name (binding names can't start with `__` per spec §04).
  env.__atomN = N;
  for (const p of AGGREGATE_PATTERNS) {
    // Today's specialisers' execute() works on whatever operand
    // shape comes out of evaluateExpr; for atom-batched refs, that
    // returns the atom-batched Value verbatim. valueOps.mul +
    // variant dispatch handle the supported shape combos:
    //   - rank-2 × atom-batched rank-1 → vectorised matvec
    //   - rank-2 × rank-2 atom-batched → batched matmul (Phase 2.2)
    // For unsupported combos the dispatch throws — we catch and
    // return null to fall through to the generic broadcast-reduce.
    const m = p.match(ir, env);
    if (!m) continue;
    try {
      const r = p.execute(ir, env, m);
      if (r === undefined || r === null) continue;
      return r;
    } catch (_) {
      // Specialiser refused / threw — fall through to generic.
      return null;
    }
  }
  return null;
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
// Atom-batched aggregate evaluator entry (engine-concepts §20.10)
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
// **P2 (LANDED 2026-05-30):** this entry is a thin shim over
// `_evalAggregateGeneric(ir, env, atomCfg)`. The body walker, axis
// inference, alignment, and tail-reduce all live in ONE pipeline
// (single-point = the same generic with `atomCfg=null`). The
// previously-separate `_inferAxisLengthsN` / `_shapeOfContainer` /
// `_liftAggregateExprN` / `_alignedConstTensorN` /
// `_alignedTensorFromGetN` helpers retired into the shared lifter
// (`_alignedTensorFromGet` + `_liftAggregateExpr` + `_alignedConst-
// Tensor` now take an `AtomConfig` parameter).
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
// **Atom-batched detection is by REF NAME**, not shape. An atom-indep
// array of shape [N] (coincidental) would be wrongly classified by
// shape alone — driving the lifter to strip an axis that doesn't
// exist, and trip subsequent axis-length inference.

function _evalAggregateBroadcastReduceN(
  ir: any, refArrays: any, N: number, baseEnv: any, overlay: any,
): any {
  return _evalAggregateGeneric(ir, /* env */ null, {
    N, refArrays, baseEnv, overlay,
  });
}


module.exports = {
  AGGREGATE_PATTERNS,
  _evalAggregate,
  _evalAggregateBroadcastReduceN,
  _tryBatchedAggregatePatterns,
  _matmulDispatch,
};
