'use strict';
const T = require('./transforms.ts');

// Closed-form prior log-densities for the continuous dists we transform.
// (Mirrors density-prims; kept local so ModelView's prior term is independent
// of the observation-consuming density walker.)
const LN_2PI = Math.log(2 * Math.PI);
const PRIOR_LOGPDF: Record<string, (x: number, p: any) => number> = {
  Normal:      (x, p) => -0.5 * LN_2PI - Math.log(p.sigma) - 0.5 * ((x - p.mu) / p.sigma) ** 2,
  Exponential: (x, p) => Math.log(p.rate) - p.rate * x,
  HalfCauchy:  (x, p) => Math.log(2) - Math.log(Math.PI * p.scale) - Math.log1p((x / p.scale) ** 2),
  HalfNormal:  (x, p) => 0.5 * Math.log(2 / Math.PI) - Math.log(p.sigma) - 0.5 * (x / p.sigma) ** 2,
  // extend as dists are added; absence => throw in buildModelView
};

function priorLogpdf(distOp: string, x: number, params: any): number {
  const f = PRIOR_LOGPDF[distOp];
  if (!f) throw new Error(`model-view: no prior log-density for '${distOp}'`);
  return f(x, params);
}

// ─── Legacy sync ModelView (OLD path, used by model-view.test.ts and the
//     mh-backend.test.ts single-latent path via mat-density.ts).
// ─────────────────────────────────────────────────────────────────────────
function buildModelView(spec: any) {
  const latents = spec.latents;
  const names = latents.map((l: any) => l.name);
  const transforms = latents.map((l: any) => T.transformFor(l.support));
  const hasDiscrete = latents.some((l: any) => l.discrete);
  const dim = latents.length;

  function constrainAll(y: Float64Array) {
    const theta: any = {};
    for (let i = 0; i < dim; i++) theta[names[i]] = transforms[i].constrain(y[i]);
    return theta;
  }
  function unconstrainAll(theta: any) {
    const y = new Float64Array(dim);
    for (let i = 0; i < dim; i++) y[i] = transforms[i].unconstrain(theta[names[i]]);
    return y;
  }
  function logPosteriorConstrained(theta: any) {
    let lp = 0;
    for (let i = 0; i < dim; i++) lp += priorLogpdf(latents[i].distOp, theta[names[i]], latents[i].params);
    lp += spec.logLikelihood(theta);
    return lp;
  }
  function logPosterior(y: Float64Array) {
    const theta = constrainAll(y);
    let lp = logPosteriorConstrained(theta);
    for (let i = 0; i < dim; i++) lp += transforms[i].logDetJ(y[i]);   // change of variables
    return lp;
  }
  return { dim, names, hasDiscrete, constrainAll, unconstrainAll, logPosterior, logPosteriorConstrained };
}

// ─── New async vector-aware ModelView (wires mcmc-density.ts scorer) ─────
//
// Contract:
//   constrainAll(y: Float64Array) → FLAT record:
//     scalars:  { mu: 0.5, tau: 1.2 }
//     vectors:  { 'theta[0]': 1.0, 'theta[1]': 2.0, … }
//   This flat layout satisfies the driver's  `theta[mv.names[d]]`  indexing.
//
//   logPosteriorConstrained(scorerPt) accepts the SCORER format (with
//     Float64Array vectors), calls logPi(scorerPt) directly.
//
//   logPosterior(y) → constrainAll(y) → reassemble scorer format → logPi + Jacobians.
//
//   unconstrainAll(scorerPt) accepts scorer format, returns Float64Array(dim).
//
//   latentNames   — unique latent names in declaration order.
//   latentShapes  — { kind: 'scalar' | 'vector', dims?: [d] } per latent.
// ─────────────────────────────────────────────────────────────────────────
async function buildModelViewFromCtx(ctx: any, posteriorDeriv: any): Promise<any> {
  const { buildLogPi } = require('./mcmc-density.ts');
  const modelSpec = require('./model-spec.ts');
  const empirical = require('./empirical.ts');

  // 1. Enumerate latents in declaration order.
  const leafLatents = modelSpec.enumerateLatents(posteriorDeriv, ctx);

  // 2. Reject discrete latents.
  for (const l of leafLatents) {
    if (l.discrete) {
      throw new Error(
        `backend 'mh' (continuous): latent '${l.name}' is discrete; `
        + `use backend 'is' or a discrete MH path`,
      );
    }
  }

  // 3a. Prior-draw pools per latent (forward samples). These serve two jobs:
  //     (a) the per-atom shape is read from the pool's value.shape — the ground
  //     truth for scalar / vector / MATRIX latents alike (a matrix latent such
  //     as p ~ beta_row_K.(a,b) has a [G,N] per-atom shape that the derivation
  //     walk can't infer statically; the materialised measure carries it); and
  //     (b) one JOINT atom supplies all latents for sampler initialization.
  // Draw bindings share the context's atom axis: matSample reads its parents'
  // atom-i values for its own atom i. Constructor-joint fields instead belong
  // to the joint's independently seeded factor scopes, so read those actual
  // fields rather than looking up their distribution bindings individually.
  const drawPrior = leafLatents.every((l: any) => ctx.bindings.get(l.name)?.type === 'draw');
  let priorMeasure: any = null;
  try { priorMeasure = await ctx.getMeasure(posteriorDeriv.from); } catch (_) { /* density-only setup can still succeed */ }
  const priorPools: any[] = [];   // per latent: { samples, stride, n } or null
  const weightParents: any[] = priorMeasure ? [priorMeasure] : [];
  for (const l of leafLatents) {
    let pool: any = null;
    try {
      // Draws use the shared context cache. A constructor joint's free inputs
      // instead read its materialized fields, including independent factors.
      const m: any = drawPrior ? await ctx.getMeasure(l.name) : priorMeasure?.fields?.[l.name];
      const s = m && (m.samples || (m.value && m.value.data));
      if (s && s.length > 0) {
        // Per-atom size = product of all but the leading (atom) axis.
        let stride = 1;
        if (m.value && Array.isArray(m.value.shape) && m.value.shape.length > 1) {
          for (let k = 1; k < m.value.shape.length; k++) stride *= m.value.shape[k];
        } else if (l.shape.kind === 'vector' && l.shape.dims[0] > 0) {
          stride = l.shape.dims[0];
        }
        pool = { samples: s, stride, n: Math.floor(s.length / stride) };
        if (drawPrior) weightParents.push(m);
        // Reconcile the latent's shape with the pool: a >1 per-atom size is a
        // vector latent of that width (fixes matrix/pool-sized latents that the
        // static walk left as scalar or dims:[0]). `atomShape` keeps the FULL
        // per-atom shape (e.g. [2,16] for a matrix) so the scorer can rebuild a
        // correctly-shaped value for indexing — flattening alone would break a
        // likelihood that indexes p[g][j].
        const atomShape = (m.value && Array.isArray(m.value.shape) && m.value.shape.length > 1)
          ? m.value.shape.slice(1) : (stride > 1 ? [stride] : null);
        if (stride > 1) l.shape = { kind: 'vector', dims: [stride], atomShape };
        else if (stride === 1 && l.shape.kind === 'vector' && l.support?.n == null) l.shape = { kind: 'scalar' };
      }
    } catch (_) { /* no pool → keep the statically-inferred shape */ }
    priorPools.push(pool);
  }

  // 3b. Build the transform BLOCKS, using the reconciled shapes.
  //
  // A block owns one contiguous slice of the unconstrained vector `y` and the
  // bijection onto its constrained coordinates. Most latents give one scalar
  // block per element, so the unconstrained dimension equals the constrained
  // one. Whole-vector supports use ONE block: ordered(K) has K free
  // coordinates, while stdsimplex(K) and zerosum(K) have K−1. Each transform's
  // Jacobian uses that set's reference measure (spec §03).
  //
  // Consequence for consumers: `dim` (unconstrained, what the kernels move in)
  // and `names.length` (constrained output coordinates, what `constrainAll`
  // returns and `drawsByName` keys on) are no longer the same number.
  const blocks: any[]         = [];   // { kind, latent, name?, elem?, K?, yDim?, y0, t }
  const coordNames: string[]  = [];   // e.g. ['mu','tau','p[0]','p[1]','p[2]']
  const latentNames: string[] = [];   // unique latent names
  const latentShapes: any[]   = [];   // { kind:'scalar'|'vector', dims?:[d] }
  let ydim = 0;

  for (const l of leafLatents) {
    latentNames.push(l.name);
    latentShapes.push(l.shape);
    const width = l.shape.kind === 'scalar' ? 1 : l.shape.dims[0];
    const kind = l.support && l.support.kind;
    if (kind === 'unsupported') {
      throw new Error(
        `backend '${(ctx.inferenceOpts && ctx.inferenceOpts.backend) || 'mh'}': latent `
        + `'${l.name}' is a '${l.support.distOp}' draw, whose support has no unconstraining `
        + `transform in this engine; use backend 'is', which forward-samples the prior `
        + `rather than exploring an unconstrained space`);
    }
    if (kind === 'simplex' || kind === 'ordered' || kind === 'zerosum') {
      const minWidth = kind === 'simplex' ? 2 : 1;
      if (l.shape.kind !== 'vector' || !(width >= minWidth)) {
        throw new Error(
          `model-view: ${kind} latent '${l.name}' needs a vector shape of width ≥ ${minWidth}; got `
          + `${JSON.stringify(l.shape)}`);
      }
      const t = kind === 'simplex' ? T.simplexTransform(width)
        : kind === 'ordered' ? T.orderedTransform(width) : T.zerosumTransform(width);
      const yDim = kind === 'ordered' ? width : width - 1;
      blocks.push({ kind: 'vector', latent: l.name, K: width, yDim, y0: ydim, t });
      for (let i = 0; i < width; i++) coordNames.push(`${l.name}[${i}]`);
      ydim += yDim;
      continue;
    }
    const t = kind === 'product' ? null : T.transformFor(l.support);
    if (l.shape.kind === 'scalar') {
      blocks.push({ kind: 'scalar', latent: l.name, name: l.name, elem: null, y0: ydim, t });
      coordNames.push(l.name);
      ydim += 1;
    } else {
      // vector / flattened matrix — one scalar block per coordinate (shared support)
      for (let i = 0; i < width; i++) {
        const nm = `${l.name}[${i}]`;
        blocks.push({ kind: 'scalar', latent: l.name, name: nm, elem: i, y0: ydim,
          t: kind === 'product' ? T.transformFor(l.support.elements[i]) : t });
        coordNames.push(nm);
        ydim += 1;
      }
    }
  }

  const dim = ydim;
  const hasBlockTransform = blocks.some((b: any) => b.kind !== 'scalar');

  // Per-coordinate (μ,σ) of the unconstrained prior IFF it is exactly an
  // independent Gaussian with constant params (every latent Normal-literal,
  // real support) — the elliptical slice sampler's EXACT reference. null ⇒
  // non-Gaussian prior ⇒ that sampler fits a Gaussian reference instead.
  let gaussianPrior: { mu: Float64Array; sigma: Float64Array } | null = null;
  if (!hasBlockTransform) {
    const det = modelSpec.detectGaussianPrior(posteriorDeriv, ctx);
    if (det) {
      const mu = new Float64Array(dim), sigma = new Float64Array(dim);
      let i = 0, ok = true;
      for (let li = 0; li < leafLatents.length && ok; li++) {
        const l = leafLatents[li]; const g = det[l.name];
        if (!g) { ok = false; break; }
        const w = l.shape.kind === 'scalar' ? 1 : l.shape.dims[0];
        for (let j = 0; j < w; j++) { mu[i] = g.mu; sigma[i] = g.sigma; i++; }
      }
      if (ok && i === dim) gaussianPrior = { mu, sigma };
    }
  }

  // 4. Async setup: build logPi / logPiBatch / priorLikBatch / probePrior.
  const { logPi, logPiBatch, priorLikBatch, probePrior, probeLik, likOf } = await buildLogPi(ctx, posteriorDeriv);

  // 4a. Prior-tractability probe. Build a probe point from every latent that HAS
  //     a pool (pool atom 0); latents without a pool are skipped individually but
  //     MUST NOT suppress the probe of their pooled siblings. probePrior iterates
  //     ALL draw names and continues when pt[nm] === undefined, so passing all
  //     pooled latents in one combined point is sufficient: it scores each pooled
  //     latent's measure (providing its upstream deps in the env via buildEnv)
  //     while silently skipping the pool-less ones. A static-error prior
  //     (unannotated non-bijection pushfwd) then refuses loudly instead of running
  //     stuck chains on a silent −∞.
  {
    const probePt: Record<string, any> = {};
    let anyPool = false;
    for (let li = 0; li < leafLatents.length; li++) {
      const l = leafLatents[li], pool = priorPools[li];
      if (!pool) continue;
      anyPool = true;
      if (l.shape.kind === 'scalar') { probePt[l.name] = pool.samples[0]; }
      else {
        const d = pool.stride; const v = new Float64Array(d);
        for (let j = 0; j < d; j++) v[j] = pool.samples[j];
        probePt[l.name] = v;
      }
    }
    if (anyPool) {
      try {
        probePrior(probePt);
      } catch (err: any) {
        const backend = (ctx.inferenceOpts && ctx.inferenceOpts.backend) || 'mh';
        const msg = err && err.message ? err.message : String(err);
        throw new Error(
          `backend '${backend}': ${msg}. MCMC/AMIS must score the prior; use `
          + `backend 'is' (which forward-samples the prior rather than scoring it), `
          + `or annotate the offending transform with bijection(f, f_inv, logvolume).`);
      }
      // 4b. LIKELIHOOD-tractability probe, for the same reason as 4a. At
      //     runtime `likWith` swallows a density throw to −∞, which is right
      //     for a proposal outside support and wrong for a STATIC refusal:
      //     MH then rejects every proposal and reports a constant chain with
      //     no diagnostic. `normalize` of a zero-mass mixture is that case
      //     (spec §06 leaves it undefined) and the IS route already refuses
      //     it, so probing here keeps the two backends from disagreeing on
      //     the same model. A legitimately −∞ likelihood does not throw and
      //     so does not trip this.
      try {
        probeLik(probePt);
      } catch (err: any) {
        const backend = (ctx.inferenceOpts && ctx.inferenceOpts.backend) || 'mh';
        const msg = err && err.message ? err.message : String(err);
        throw new Error(`backend '${backend}': ${msg}`);
      }
    }
  }

  // Build `nWalkers` initial unconstrained positions from prior draws.
  function initFromPrior(nWalkers: number, prng: () => number): Float64Array[] {
    // §04 reified ancestry and §06 joint composition: select the whole row,
    // retaining shared weighting events once. Zero fallback positions are not
    // draws from a missing prior and must never enter SMC at equal weight.
    const N = priorPools[0]?.n;
    if (!priorMeasure || !(N > 0) || priorPools.some((p: any) => !p || p.n !== N)
        || N !== ctx.sampleCount) {
      throw new Error('model-view: prior initialization requires complete, aligned joint prior atoms. '
        + 'Choose a proper prior or supply finite initPositions for posterior MCMC.');
    }
    const weights = empirical.propagateLogWeights(weightParents);
    if (weights && weights.length !== N) {
      throw new Error('model-view: prior initialization requires one weight per joint atom');
    }
    const indices = weights ? empirical.multinomialResample(weights, nWalkers, prng) : null;
    const out: Float64Array[] = [];
    for (let w = 0; w < nWalkers; w++) {
      const a = indices ? indices[w] : Math.min(N - 1, Math.floor(prng() * N));
      const scorerPt: Record<string, any> = {};
      for (let li = 0; li < leafLatents.length; li++) {
        const l = leafLatents[li], pool = priorPools[li];
        if (l.shape.kind === 'vector') {
          const d = pool.stride; const v = new Float64Array(d);
          for (let j = 0; j < d; j++) v[j] = pool.samples[a * d + j];
          scorerPt[l.name] = v;
        } else {
          scorerPt[l.name] = pool.samples[a];
        }
      }
      out.push(unconstrainAll(scorerPt));
    }
    return out;
  }

  // Helper: flat coord record → scorer-format record { name: scalar | Float64Array }.
  function flatToScorerPt(flatPt: Record<string, any>): Record<string, any> {
    const rec: Record<string, any> = {};
    for (let li = 0; li < latentNames.length; li++) {
      const nm  = latentNames[li];
      const shp = latentShapes[li];
      if (shp.kind === 'scalar') {
        rec[nm] = flatPt[nm];
      } else {
        const d = shp.dims[0];
        const vec = new Float64Array(d);
        for (let j = 0; j < d; j++) vec[j] = flatPt[`${nm}[${j}]`];
        // Preserve a matrix's axes and a singleton vector's rank: the scorer
        // otherwise interprets a one-element bare array as a scalar atom.
        rec[nm] = d === 1 || (shp.atomShape && shp.atomShape.length > 1)
          ? { shape: shp.atomShape ? shp.atomShape.slice() : [d], data: vec }
          : vec;
      }
    }
    return rec;
  }

  // Read element `j` of a latent's value in either scorer form (Float64Array /
  // shaped Value) or plain-array form.
  function elemOf(v: any, j: number): number {
    if (v instanceof Float64Array) return v[j];
    if (Array.isArray(v)) return +v[j];
    if (v && v.data) return v.data[j];
    /* c8 ignore start -- elemOf is only reached for a latent the shape walk
       already classified as a vector, so a non-indexable value here is an
       internal inconsistency. Coercing it would give every coordinate the same
       number, the silent-wrong outcome this file exists to stop, so throw. */
    throw new Error(
      `model-view: expected an indexable value for a vector latent, got ${typeof v}`);
    /* c8 ignore stop */
  }

  // 5. Driver-compatible constrainAll: Float64Array(dim) → flat record keyed by
  //    the CONSTRAINED coordinate names (`names`, which for a simplex or
  //    zero-sum latent is one key wider than that latent's slice of `y`).
  function constrainAll(y: Float64Array): Record<string, any> {
    const flat: Record<string, any> = {};
    for (let bi = 0; bi < blocks.length; bi++) {
      const b = blocks[bi];
      if (b.kind === 'vector') {
        const p = b.t.constrain(y.subarray(b.y0, b.y0 + b.yDim));
        for (let i = 0; i < b.K; i++) flat[`${b.latent}[${i}]`] = p[i];
      } else {
        flat[b.name] = b.t.constrain(y[b.y0]);
      }
    }
    return flat;
  }

  // 6. unconstrainAll: scorer-format record → Float64Array(dim).
  function unconstrainAll(scorerPt: Record<string, any>): Float64Array {
    const y = new Float64Array(dim);
    for (let bi = 0; bi < blocks.length; bi++) {
      const b = blocks[bi];
      if (b.kind === 'vector') {
        const src = scorerPt[b.latent];
        const p = new Float64Array(b.K);
        for (let i = 0; i < b.K; i++) p[i] = elemOf(src, i);
        const yv = b.t.unconstrain(p);
        for (let k = 0; k < b.yDim; k++) y[b.y0 + k] = yv[k];
      } else {
        const src = scorerPt[b.latent];
        const v = b.elem == null
          ? (typeof src === 'number' ? src : +src)
          : elemOf(src, b.elem);
        y[b.y0] = b.t.unconstrain(v);
      }
    }
    return y;
  }

  // Total change-of-variables term Σ log|dθ/dy| over the blocks.
  function logDetJTotal(y: Float64Array): number {
    let j = 0;
    for (let bi = 0; bi < blocks.length; bi++) {
      const b = blocks[bi];
      j += b.kind === 'vector'
        ? b.t.logDetJ(y.subarray(b.y0, b.y0 + b.yDim))
        : b.t.logDetJ(y[b.y0]);
    }
    return j;
  }

  // 7. logPosteriorConstrained: scorer-format record → scalar (no Jacobian).
  function logPosteriorConstrained(scorerPt: Record<string, any>): number {
    return logPi(scorerPt);
  }

  // 8. logPosterior: unconstrained Float64Array → scalar (includes Jacobian).
  function logPosterior(y: Float64Array): number {
    const flatPt = constrainAll(y);
    const scorerPt = flatToScorerPt(flatPt);
    let lp = logPi(scorerPt);
    lp += logDetJTotal(y);
    return lp;
  }

  // 8b. logPosteriorBatch: N unconstrained vectors → Float64Array(N). Batches
  // the likelihood through one logDensityN(count=N) pass (see mcmc-density
  // logPiBatch); the change-of-variables Jacobian is added per atom.
  function logPosteriorBatch(ys: Float64Array[]): Float64Array {
    const N = ys.length;
    const pts = new Array(N);
    const jac = new Float64Array(N);
    for (let a = 0; a < N; a++) {
      const y = ys[a];
      pts[a] = flatToScorerPt(constrainAll(y));
      jac[a] = logDetJTotal(y);
    }
    const lp = logPiBatch(pts);
    for (let a = 0; a < N; a++) lp[a] = Number.isFinite(lp[a]) ? lp[a] + jac[a] : -Infinity;
    return lp;
  }

  // 8c. logPriorLikBatch: N unconstrained vectors → { prior, lik } per atom,
  // where prior INCLUDES the change-of-variables Jacobian (so a tempered target
  // log π_β = prior + β·lik is correct for any β, and at β=1 reproduces
  // logPosteriorBatch). Used by the SMC sampler to re-temper a population by
  // arithmetic on cached `lik` with no re-evaluation.
  function logPriorLikBatch(ys: Float64Array[]): { prior: Float64Array; lik: Float64Array } {
    const N = ys.length;
    const pts = new Array(N);
    const jac = new Float64Array(N);
    for (let a = 0; a < N; a++) {
      const y = ys[a];
      pts[a] = flatToScorerPt(constrainAll(y));
      jac[a] = logDetJTotal(y);
    }
    const { prior, lik } = priorLikBatch(pts);
    for (let a = 0; a < N; a++) prior[a] = Number.isFinite(prior[a]) ? prior[a] + jac[a] : -Infinity;
    return { prior, lik };
  }

  return {
    dim,
    // CONSTRAINED output coordinate names. `names.length` is the number of
    // output coordinates and `dim` the unconstrained dimension the kernels move
    // in; they differ for simplex and zero-sum block transforms,
    // so a consumer building `drawsByName` must iterate `names.length`.
    names: coordNames,
    hasDiscrete: false,       // only continuous latents reach here
    constrainAll,
    unconstrainAll,
    logPosterior,
    logPosteriorBatch,
    logPriorLikBatch,
    logPosteriorConstrained,
    likOf,
    initFromPrior,
    gaussianPrior,
    latentNames,
    latentShapes,
  };
}

module.exports = { buildModelView, buildModelViewFromCtx, priorLogpdf };
