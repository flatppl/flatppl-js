'use strict';

// =====================================================================
// mat-poisson.ts — PoissonProcess core (spec §08; engine-concepts §2.3)
// =====================================================================
//
// The mathematically load-bearing core of `PoissonProcess(intensity)`,
// decoupled from worker / classifier orchestration so it is verifiable
// in isolation. The full `matPoissonProcess` / `walkPoissonProcess`
// handlers wrap the worker draws + the consume/rest density contract
// around these.
//
// Spec §08: variates are arrays of points (permutation-invariant). For
// the canonical construction `intensity = weighted(M, shape)` (M = the
// expected count = `totalmass(intensity)`, `shape = normalize(intensity)`
// a normalized distribution):
//
//   SAMPLING (per atom i):  k_i ~ Poisson(M_i);  k_i points iid from shape.
//   DENSITY  (w.r.t. iid(Lebesgue, k), per atom i):
//       logp_i = Σ_j log λ(t_ij) − M_i
//              = k_i·log(M_i) + Σ_j shape_logpdf(t_ij) − M_i
//   because λ(t) = intensity(t) = M·shape_pdf(t).
//
// The per-atom output has VARIABLE length k_i → the ragged value kind
// (ragged.ts): VectorOfVectors for scalar points.

const R = require('./ragged.ts');

// Lazily-required shared plumbing (avoid load-order cycles — sampler.ts
// pulls in a large graph; mat-poisson stays a near-leaf otherwise).
function _shared() { return require('./materialiser-shared.ts'); }
function _orchestrator() { return require('./orchestrator.ts'); }
function _valueLib() { return require('./value.ts'); }
function _empirical() { return require('./empirical.ts'); }
function _sampler() { return require('./sampler.ts'); }

// A resolved scalar/vector value → a literal-valued IR node (the same
// "bake params as literals" idiom matDirichlet / matBinnedPoissonProcess
// use, so the leaf samples with empty refArrays = atom-independent).
function _literalIRof(val: any): any {
  if (typeof val === 'number' || typeof val === 'boolean') {
    return { kind: 'lit', value: +val };
  }
  const av = _valueLib().asValue(val);
  if (av.shape.length === 0) return { kind: 'lit', value: av.data[0] };
  if (av.shape.length === 1) {
    return { kind: 'call', op: 'vector',
             args: Array.from(av.data, (v) => ({ kind: 'lit', value: v })) };
  }
  throw new Error('PoissonProcess: shape-leaf params of rank ≥ 2 are unsupported (MVP)');
}

// Rebuild a scalar-distribution leaf IR with every parameter resolved to a
// literal (atom-INDEPENDENT). Throws (→ "per-atom shape deferred") if any
// param cannot be resolved against the fixed-phase env.
function _resolveLeafToLiteralIR(leafIR: any, ctx: any): any {
  const orch = _orchestrator();
  const out: any = { kind: 'call', op: leafIR.op };
  const resolve1 = (node: any) => {
    // A draw-dependent param makes resolveIRToValue THROW ("draw not evaluable")
    // rather than return null; either way the param is not atom-independent.
    let v: any = null;
    try { v = orch.resolveIRToValue(node, ctx.bindings, ctx.fixedValues); }
    catch (_e) { v = null; }
    if (v == null) {
      throw new Error('PoissonProcess: cannot resolve a shape-leaf parameter to a '
        + 'constant — per-atom (draw-dependent) shape parameters are deferred for '
        + 'generative sampling; the density/likelihood path handles them (spec §08)');
    }
    return _literalIRof(v);
  };
  if (leafIR.kwargs) {
    out.kwargs = {};
    for (const k in leafIR.kwargs) {
      if (Object.prototype.hasOwnProperty.call(leafIR.kwargs, k)) {
        out.kwargs[k] = resolve1(leafIR.kwargs[k]);
      }
    }
  }
  if (Array.isArray(leafIR.args) && leafIR.args.length > 0) {
    out.args = leafIR.args.map(resolve1);
  }
  return out;
}

// Parse the MVP intensity into { M, shapeLeafIR }. `intensityIR` is first
// resolved via expandMeasure (so a ref to a separately-bound intensity, and
// the inner shape ref, both expand). The MVP form is a single weighted shape
// `weighted(M, shape)` — M = expected count = totalmass (the shape is a
// normalized probability distribution, so the weight IS the total mass).
// expandMeasure renders a CONSTANT-weight binding as `logweighted(log M, …)`
// and an inline call as `weighted(M, …)`; we accept both. Returns null for
// any other intensity shape (general measure / superpose = the follow-up).
function _parsePoissonIntensity(intensityIR: any, ctx: any): any {
  const exp = _orchestrator().expandMeasure(intensityIR, ctx);
  if (!exp || exp.kind !== 'call' || !Array.isArray(exp.args) || exp.args.length !== 2) {
    return null;
  }
  const orch = _orchestrator();
  if (exp.op === 'weighted') {
    const w = orch.resolveIRToValue(exp.args[0], ctx.bindings, ctx.fixedValues);
    if (typeof w !== 'number') return null;
    return { M: w, shapeLeafIR: exp.args[1] };
  }
  if (exp.op === 'logweighted') {
    const lw = orch.resolveIRToValue(exp.args[0], ctx.bindings, ctx.fixedValues);
    if (typeof lw !== 'number') return null;
    return { M: Math.exp(lw), shapeLeafIR: exp.args[1] };
  }
  return null;
}

// matPoissonProcess — atom-batched draw of variable-length point sets.
//
// MVP (spec §08; engine-concepts §2.3): intensity = weighted(M, shape) with
// an ATOM-INDEPENDENT expected count M = totalmass and an atom-independent
// scalar shape distribution. Per atom i: k_i ~ Poisson(M) points, each iid
// from `shape`. Because the points are iid and exchangeable, drawing one
// pooled sampleN of Σk_i points and partitioning it by the per-atom counts
// is exactly equivalent to per-atom draws — so this batches into two worker
// calls (counts, then the pooled points) + a main-thread ragged assembly,
// mirroring matBinnedPoissonProcess. Per-atom-varying M / shape params are a
// documented follow-up (the density path already handles per-θ shapes).
function matPoissonProcess(name: string, d: any, ctx: any): Promise<any> {
  const distIR = d.distIR || {};
  const intensityIR = (distIR.kwargs && distIR.kwargs.intensity)
    || (distIR.args && distIR.args[0]);
  if (!intensityIR) {
    return Promise.reject(new Error('PoissonProcess: requires an intensity argument'));
  }
  const parsed = _parsePoissonIntensity(intensityIR, ctx);
  if (!parsed) {
    return Promise.reject(new Error(
      'PoissonProcess: the MVP supports intensity = weighted(<expected count>, '
      + '<scalar distribution>) with an atom-independent expected count only; a '
      + 'general-measure / superpose / per-atom intensity is a follow-up (spec §08)'));
  }
  const shared = _shared();
  const Mval = parsed.M;
  if (!(Mval >= 0)) {
    return Promise.reject(new Error(
      'PoissonProcess: the intensity weight (expected count) must be ≥ 0, got ' + Mval));
  }
  if (!_sampler().isKnownDistribution(parsed.shapeLeafIR.op)) {
    return Promise.reject(new Error(
      "PoissonProcess: the MVP shape must be a scalar distribution, got '"
      + parsed.shapeLeafIR.op + "' (d-dim / composite point kernels are a follow-up)"));
  }
  let leafIR: any;
  try {
    leafIR = _resolveLeafToLiteralIR(parsed.shapeLeafIR, ctx);
  } catch (err) {
    return Promise.reject(err);
  }
  const N = ctx.sampleCount;
  const M = Mval;
  // M === 0 ⇒ Poisson(0) is degenerate (every atom draws 0; stdlib disallows
  // rate 0, cf. matBinnedPoissonProcess). All-empty ragged, no worker calls.
  if (M === 0) {
    const ragged = assemblePoissonRagged(new Float64Array(N), new Float64Array(0));
    return Promise.resolve(_empirical().raggedMeasure(ragged, null));
  }
  const poissonIR = { kind: 'call', op: 'Poisson',
                      kwargs: { rate: { kind: 'lit', value: M } } };
  return ctx.sendWorker({
    type: 'sampleN', ir: poissonIR, count: N, refArrays: {},
    seed: shared.nameSeed(name + '|counts', ctx.rootKey),
  }).then((countsReply: any) => {
    const countsF = countsReply.samples;
    const counts = new Int32Array(N);
    let tot = 0;
    for (let i = 0; i < N; i++) { counts[i] = countsF[i] | 0; tot += counts[i]; }
    if (tot === 0) {
      const ragged = assemblePoissonRagged(counts, new Float64Array(0));
      return _empirical().raggedMeasure(ragged, null);
    }
    return ctx.sendWorker({
      type: 'sampleN', ir: leafIR, count: tot, refArrays: {},
      seed: shared.nameSeed(name + '|pts', ctx.rootKey),
    }).then((ptsReply: any) => {
      const ragged = assemblePoissonRagged(counts, ptsReply.samples);
      return _empirical().raggedMeasure(ragged, null);
    });
  });
}

// Assemble the ragged process value from per-atom counts + the flat pool
// of all atoms' points (drawn iid from the shape, atom-major). `counts`
// length N; `pointsFlat` length Σ counts. Scalar points (kernelShape=[]).
function assemblePoissonRagged(counts: ArrayLike<number>,
                              pointsFlat: Float64Array): any {
  const N = counts.length;
  const offsets = new Int32Array(N + 1);
  let tot = 0;
  for (let i = 0; i < N; i++) { tot += counts[i] | 0; offsets[i + 1] = tot; }
  if (pointsFlat.length !== tot) {
    throw new Error('assemblePoissonRagged: pointsFlat length ' + pointsFlat.length
      + ' ≠ Σcounts ' + tot);
  }
  return R.ragged(pointsFlat, offsets, []);
}

// Per-atom log-density of a ragged PoissonProcess observation under
// `intensity = weighted(M, shape)`. Returns a uniform `[N]` Value
// (`{shape:[N], data}`) — one log-density per atom (the consume/rest
// walker sums these into the trace logp).
//   - `raggedObs`: a ragged value (scalar points).
//   - `M`: per-atom expected count — a scalar or a length-N array.
//   - `shapeLogpdf`: t → log shape_pdf(t) (the NORMALIZED shape density).
function poissonProcessLogDensity(raggedObs: any, M: number | ArrayLike<number>,
                                  shapeLogpdf: (t: number) => number): any {
  const N = R.raggedCount(raggedObs);
  const out = new Float64Array(N);
  const data = raggedObs.data, offsets = raggedObs.offsets;
  for (let i = 0; i < N; i++) {
    const Mi = (typeof M === 'number') ? M : M[i];
    const lo = offsets[i], hi = offsets[i + 1];
    const k = hi - lo;                       // scalar points: span = count
    let s = (k > 0 ? k * Math.log(Mi) : 0) - Mi;
    for (let p = lo; p < hi; p++) s += shapeLogpdf(data[p]);
    out[i] = s;
  }
  return { shape: [N], data: out };
}

module.exports = {
  assemblePoissonRagged,
  poissonProcessLogDensity,
  matPoissonProcess,
};
