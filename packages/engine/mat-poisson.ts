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
  // A set-valued parameter (`Uniform`'s support) resolves to a set object.
  if (val && val.kind === 'interval') {
    return { kind: 'call', op: 'interval',
             args: [{ kind: 'lit', value: val.a }, { kind: 'lit', value: val.b }] };
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
  // `truncate` distributes over a superposition: each component is restricted.
  if (node.op === 'truncate' && Array.isArray(node.args) && node.args.length === 2) {
    const inner = parseIntensityComponents(node.args[0]);
    if (!inner || inner.some((c: any) => c.setIR)) return null;
    return inner.map((c: any) => ({ ...c, setIR: node.args[1] }));
  }
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

// One component `{ weights, shapeLeafIR, setIR, normalized }`: the rate is the
// product of `weights` (each `{ ir, isLog }`) times the shape density,
// restricted to `setIR` when it is set, and divided by P_s(S) when `normalized`. Nested `weighted` / `logweighted` multiply, which is
// also how an expanded `normalize` arrives (a `logweighted` by −log mass). A §06
// `truncate(·, S)` restricts without renormalising and commutes with the
// weights, so `truncate(weighted(w, s), S)` and `weighted(w, truncate(s, S))`
// are the same component: rate w·s_pdf(t) on S, expected count w·P_s(S).
function _parseComponent(node: any): any {
  if (!node || node.kind !== 'call') return null;
  const two = Array.isArray(node.args) && node.args.length === 2;
  if ((node.op === 'weighted' || node.op === 'logweighted') && two) {
    const inner = _parseComponent(node.args[1]);
    if (!inner) return null;
    return { ...inner,
             weights: [{ ir: node.args[0], isLog: node.op === 'logweighted' }, ...inner.weights] };
  }
  if (node.op === 'truncate' && two) {
    const inner = _parseComponent(node.args[0]);
    if (!inner || inner.setIR) return null;
    return { ...inner, setIR: node.args[1] };
  }
  // `normalize` drops the weights, and on a truncated shape divides by P_s(S).
  // The density path sees it already expanded to a `logweighted`; the sampler
  // path sees it as written.
  if (node.op === 'normalize' && Array.isArray(node.args) && node.args.length === 1) {
    const inner = _parseComponent(node.args[0]);
    if (!inner) return null;
    return { ...inner, weights: [], normalized: !!inner.setIR };
  }
  // Bare measure ⇒ weight 1 (M contribution 1). A non-distribution op falls
  // through to the caller's isKnownDistribution check, which rejects it.
  return { weights: [], shapeLeafIR: node, setIR: null, normalized: false };
}

// log of a component's total weight, its factors evaluated with `evaluate`.
function componentLogWeight(c: any, evaluate: (ir: any) => number): number {
  let lw = 0;
  for (const f of c.weights) {
    const v = +evaluate(f.ir);
    lw += f.isLog ? v : Math.log(v);   // log 0 = −∞ (the component drops out)
  }
  return lw;
}

// The interval bounds of a component's truncation set, evaluated with
// `evaluate` (an IR → number function), or null for an untruncated component.
function componentBounds(c: any, evaluate: (ir: any) => number): [number, number] | null {
  if (!c.setIR) return null;
  const S = c.setIR;
  if (!(S.kind === 'call' && S.op === 'interval' && S.args?.length === 2)) {
    throw new Error('PoissonProcess: a truncated intensity needs an interval set (spec §06 '
      + "`truncate`), got '" + (S.op || S.name || S.kind) + "'");
  }
  return [+evaluate(S.args[0]), +evaluate(S.args[1])];
}

// The support `[a, b]` of a `Lebesgue(support = S)` shape (§06 writes the
// homogeneous intensity as `weighted(rate, Lebesgue(support = reals))`), or null
// for any other shape. `S` is an interval or a predefined interval set.
function lebesgueSupport(leafIR: any, evaluate: (ir: any) => number): [number, number] | null {
  if (!leafIR || leafIR.op !== 'Lebesgue') return null;
  const S = (leafIR.kwargs && leafIR.kwargs.support) || (leafIR.args && leafIR.args[0]);
  if (S && S.kind === 'const') {
    const known: Record<string, [number, number]> = {
      reals: [-Infinity, Infinity], posreals: [0, Infinity], nonnegreals: [0, Infinity],
      unitinterval: [0, 1],
    };
    if (known[S.name]) return known[S.name];
  }
  if (S && S.kind === 'call' && S.op === 'interval' && S.args?.length === 2) {
    return [+evaluate(S.args[0]), +evaluate(S.args[1])];
  }
  throw new Error('PoissonProcess: a Lebesgue intensity needs an interval support (reals, '
    + 'posreals, nonnegreals, unitinterval or interval(a, b))');
}

// log of the Lebesgue length of [lo, hi] ∩ support.
function logLebesgueMass(support: [number, number], lo: number, hi: number): number {
  return Math.log(Math.max(0, Math.min(hi, support[1]) - Math.max(lo, support[0])));
}

// log P_s([lo, hi]) for the shape leaf with positional `params`, from its CDF.
function logIntervalMass(kernelName: string, params: any[], lo: number, hi: number): number {
  const entry = _sampler()._internal.REGISTRY[kernelName];
  const dist = entry && entry.Ctor ? new entry.Ctor(...params) : null;
  if (!dist || typeof dist.cdf !== 'function' || entry.discrete) {
    throw new Error("PoissonProcess: a truncated intensity needs a continuous shape with a CDF, got '"
      + kernelName + "'");
  }
  return Math.log(dist.cdf(hi) - dist.cdf(lo));
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
  const constant = (ir: any): number => {
    let v: any = null;
    try { v = orch.resolveIRToValue(ir, ctx.bindings, ctx.fixedValues); }
    catch (_e) { v = null; }
    if (typeof v !== 'number') {
      throw new Error('PoissonProcess: cannot resolve an intensity weight or truncation bound '
        + 'to a constant — per-atom (draw-dependent) intensity is deferred for generative sampling');
    }
    return v;
  };
  try {
    for (let i = 0; i < comps.length; i++) {
      const c = comps[i];
      const logw = componentLogWeight(c, constant);
      if (Number.isNaN(logw)) {
        throw new Error('PoissonProcess: an intensity weight (expected count) must be ≥ 0');
      }
      const bounds = componentBounds(c, constant);
      const lebesgue = lebesgueSupport(c.shapeLeafIR, constant);
      if (lebesgue) {
        // Rate w on [lo, hi] ∩ support: count Poisson(w·length), points uniform.
        const [lo, hi] = bounds || [-Infinity, Infinity];
        const a = Math.max(lo, lebesgue[0]), b = Math.min(hi, lebesgue[1]);
        if (!Number.isFinite(b - a)) {
          throw new Error('PoissonProcess: a Lebesgue intensity must be truncated to a bounded '
            + 'set to have finite mass (spec §08)');
        }
        const leafIR = { kind: 'call', op: 'Uniform', args: [{ kind: 'call', op: 'interval',
          args: [{ kind: 'lit', value: a }, { kind: 'lit', value: b }] }] };
        resolved.push({ w: Math.exp(logw + Math.log(Math.max(0, b - a))), leafIR, cdfRange: null });
        continue;
      }
      if (!_sampler().isKnownDistribution(c.shapeLeafIR.op)) {
        throw new Error("PoissonProcess: a shape must be a scalar distribution, got '"
          + c.shapeLeafIR.op + "' (d-dim / composite point kernels are a follow-up)");
      }
      const leafIR = _resolveLeafToLiteralIR(c.shapeLeafIR, ctx);
      // A truncated shape draws by inverse CDF from u ~ Uniform(F(lo), F(hi)).
      let cdfRange: [number, number] | null = null;
      let dist: any = null;
      let logMass = 0;
      if (bounds) {
        const params = _sampler().resolveParams(leafIR, _sampler().lookupDistribution(leafIR), {});
        logMass = logIntervalMass(leafIR.op, params, bounds[0], bounds[1]);
        dist = new (_sampler()._internal.REGISTRY[leafIR.op].Ctor)(...params);
        cdfRange = [dist.cdf(bounds[0]), dist.cdf(bounds[1])];
      }
      resolved.push({ w: Math.exp(logw + (c.normalized ? 0 : logMass)), leafIR, cdfRange, dist });
    }
  } catch (err) { return Promise.reject(err); }
  const N = ctx.sampleCount;
  return Promise.all(resolved.map((r, i) => _sampleComponentRagged(name, i, r, N, ctx)))
    .then((parts: any[]) => _empirical().raggedMeasure(R.raggedMerge(parts), null));
}

// One component of the superposition (also the whole single-component case):
// per-atom counts k ~ Poisson(M), one pooled `sampleN(leaf, Σk)` partitioned
// by counts → a ragged value over the N atoms.
function _sampleComponentRagged(name: string, idx: number, comp: any, N: number,
                                ctx: any): Promise<any> {
  const shared = _shared();
  const M = comp.w;
  const leafIR = comp.cdfRange
    ? { kind: 'call', op: 'Uniform', args: [{ kind: 'call', op: 'interval',
      args: [{ kind: 'lit', value: comp.cdfRange[0] }, { kind: 'lit', value: comp.cdfRange[1] }] }] }
    : comp.leafIR;
  const toPoint = comp.cdfRange
    ? (pts: Float64Array) => pts.map((u) => comp.dist.quantile(u)) : (pts: Float64Array) => pts;
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
    }).then((ptsReply: any) => assemblePoissonRagged(counts, toPoint(ptsReply.samples)));
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
  componentBounds,
  componentLogWeight,
  lebesgueSupport,
  logIntervalMass,
  logLebesgueMass,
  poissonProcessLogDensity,
  matPoissonProcess,
  parseIntensityComponents,
};
