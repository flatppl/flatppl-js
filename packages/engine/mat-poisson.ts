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

// Parse an EXPANDED intensity IR into a list of mixture components
// `[{ weightIR, isLog, shapeLeafIR }]`, or null for an unsupported shape.
// PURE / structural (no ctx) — the intensity must already be expanded (refs
// resolved) by the caller (matPoissonProcess via expandMeasure; the density
// walker via _expandStructural). Handles:
//   - `superpose(c1, …)` (inline) / `select(branches, logweights:null)` (the
//     by-name superpose→select rewrite) — a SUPERPOSITION of components;
//   - `weighted(w, shape)` / `logweighted(g, shape)` — one weighted component;
//   - a bare distribution — one component of weight 1.
// Each component's weight is kept as a value IR (per-θ on the density side);
// `isLog` marks a log-weight (logweighted / select-branch). By the
// superposition theorem the process is the union of its components, so a
// single `weighted(M, shape)` is just the one-component case. Shared by the
// sampler (mat-poisson) and the density walker (density.ts).
function parseIntensityComponents(node: any): any {
  if (!node || node.kind !== 'call') return null;
  if (node.op === 'superpose' && Array.isArray(node.args)) {
    return _componentsFrom(node.args);
  }
  if (node.op === 'select' && Array.isArray(node.branches)) {
    // superpose → select carries logweights:null (each branch self-carries its
    // weight via its own weighted/logweighted sub-IR). A normalized weighted
    // mixture (non-null logweights) is NOT a superposition intensity — reject.
    if (node.logweights != null) return null;
    return _componentsFrom(node.branches);
  }
  const c = _parseComponent(node);
  return c ? [c] : null;
}

function _componentsFrom(arr: any[]): any {
  const out: any[] = [];
  for (let i = 0; i < arr.length; i++) {
    const c = _parseComponent(arr[i]);
    if (!c) return null;
    out.push(c);
  }
  return out.length > 0 ? out : null;
}

function _parseComponent(node: any): any {
  if (!node || node.kind !== 'call') return null;
  if (node.op === 'weighted' && Array.isArray(node.args) && node.args.length === 2) {
    return { weightIR: node.args[0], isLog: false, shapeLeafIR: node.args[1] };
  }
  if (node.op === 'logweighted' && Array.isArray(node.args) && node.args.length === 2) {
    return { weightIR: node.args[0], isLog: true, shapeLeafIR: node.args[1] };
  }
  // Bare measure ⇒ weight 1 (M contribution 1). A non-distribution op falls
  // through to the caller's isKnownDistribution check, which rejects it.
  return { weightIR: { kind: 'lit', value: 1 }, isLog: false, shapeLeafIR: node };
}

// matPoissonProcess — atom-batched draw of variable-length point sets.
//
// Spec §08; engine-concepts §2.3. intensity = a SUPERPOSITION of weighted
// scalar shapes `superpose(weighted(w_1, s_1), …)` (a single `weighted(M, s)`
// is the one-component case). All weights and shape params must be ATOM-
// INDEPENDENT (per-atom-varying generative params are a documented follow-up;
// the density path already handles per-θ).
//
// By the SUPERPOSITION THEOREM a process with intensity Σ_i weighted(w_i, s_i)
// is the union of independent component processes (the i-th: k ~ Poisson(w_i)
// points from s_i). So each component samples exactly like the single case —
// per-atom counts `Poisson(w_i)` + one pooled `sampleN(s_i, Σk)` partitioned
// by counts (exact by iid exchangeability, à la matBinnedPoissonProcess) — and
// `raggedMerge` unions them per atom.
function matPoissonProcess(name: string, d: any, ctx: any): Promise<any> {
  const distIR = d.distIR || {};
  const intensityIR = (distIR.kwargs && distIR.kwargs.intensity)
    || (distIR.args && distIR.args[0]);
  if (!intensityIR) {
    return Promise.reject(new Error('PoissonProcess: requires an intensity argument'));
  }
  const expanded = _orchestrator().expandMeasure(intensityIR, ctx);
  const comps = parseIntensityComponents(expanded);
  if (!comps) {
    return Promise.reject(new Error(
      'PoissonProcess: intensity must be weighted(<expected count>, <scalar '
      + 'distribution>) or a superpose of such components (spec §08)'));
  }
  const orch = _orchestrator();
  const resolved: any[] = [];
  for (let i = 0; i < comps.length; i++) {
    const c = comps[i];
    let wv: any = null;
    try { wv = orch.resolveIRToValue(c.weightIR, ctx.bindings, ctx.fixedValues); }
    catch (_e) { wv = null; }
    if (typeof wv !== 'number') {
      return Promise.reject(new Error(
        'PoissonProcess: cannot resolve an intensity weight to a constant — '
        + 'per-atom (draw-dependent) intensity is deferred for generative sampling'));
    }
    const w = c.isLog ? Math.exp(wv) : wv;
    if (!(w >= 0)) {
      return Promise.reject(new Error(
        'PoissonProcess: an intensity weight (expected count) must be ≥ 0, got ' + w));
    }
    if (!_sampler().isKnownDistribution(c.shapeLeafIR.op)) {
      return Promise.reject(new Error(
        "PoissonProcess: a shape must be a scalar distribution, got '"
        + c.shapeLeafIR.op + "' (d-dim / composite point kernels are a follow-up)"));
    }
    let leafIR: any;
    try { leafIR = _resolveLeafToLiteralIR(c.shapeLeafIR, ctx); }
    catch (err) { return Promise.reject(err); }
    resolved.push({ w, leafIR });
  }
  const N = ctx.sampleCount;
  return Promise.all(resolved.map((r, i) => _sampleComponentRagged(name, i, r.w, r.leafIR, N, ctx)))
    .then((parts: any[]) => _empirical().raggedMeasure(R.raggedMerge(parts), null));
}

// One component of the superposition (also the whole single-component case):
// per-atom counts k ~ Poisson(M), one pooled `sampleN(leaf, Σk)` partitioned
// by counts → a ragged value over the N atoms.
function _sampleComponentRagged(name: string, idx: number, M: number,
                                leafIR: any, N: number, ctx: any): Promise<any> {
  const shared = _shared();
  // M === 0 ⇒ Poisson(0) is degenerate (every atom draws 0; stdlib disallows
  // rate 0, cf. matBinnedPoissonProcess). All-empty component, no worker calls.
  if (M === 0) {
    return Promise.resolve(assemblePoissonRagged(new Float64Array(N), new Float64Array(0)));
  }
  const poissonIR = { kind: 'call', op: 'Poisson',
                      kwargs: { rate: { kind: 'lit', value: M } } };
  return ctx.sendWorker({
    type: 'sampleN', ir: poissonIR, count: N, refArrays: {},
    seed: shared.nameSeed(name + '|c' + idx + '|counts', ctx.rootKey),
  }).then((countsReply: any) => {
    const countsF = countsReply.samples;
    const counts = new Int32Array(N);
    let tot = 0;
    for (let i = 0; i < N; i++) { counts[i] = countsF[i] | 0; tot += counts[i]; }
    if (tot === 0) return assemblePoissonRagged(counts, new Float64Array(0));
    return ctx.sendWorker({
      type: 'sampleN', ir: leafIR, count: tot, refArrays: {},
      seed: shared.nameSeed(name + '|c' + idx + '|pts', ctx.rootKey),
    }).then((ptsReply: any) => assemblePoissonRagged(counts, ptsReply.samples));
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
  parseIntensityComponents,
};
