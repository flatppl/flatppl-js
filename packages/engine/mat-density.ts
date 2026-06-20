'use strict';

// Density-related materialiser handlers: matScore (the ONE CLM density
// consumer), matBayesupdate, matLogdensityof, matBroadcastLogdensity,
// matLikelihoodDensity, matTotalmass. Plus the normalize-mass
// resolve-then-literalise pass (audit M3) every scoring path runs
// before density evaluation.
//
// jointchain/kchain sampling AND density both flow through the canonical
// CLM lowering now (lowerMeasure → matClm / matScore); the former bespoke
// `matJointchain` handler is retired. The N-ary kchain retained history is
// reconstructed from `node.marginalHistoryBody` (the same structure matClm
// samples), materialised via materialiser.materialiseMeasureIR.

import type {
  DerivationBayesupdate,
  DerivationBroadcastLogdensity,
  DerivationLogdensityof,
  DerivationTotalmass,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const shared       = require('./materialiser-shared.ts');
const mcRecipe     = require('./mc-recipe.ts');
const clm          = require('./clm.ts');
const densityPrims = require('./density-prims.ts');

const {
  nameSeed,
  isFunctionLikeBinding,
  collectRefArrays,
  prepareDensityRefs,
  pushFixedEnv,
  pushModuleRegistry,
  measureFromValue,
  measureFromReply,
  measureToRefValue,
  measureToPerAtomRecords,
  inlineBoundaryDerivations,
  measureN,
  scalarMeasureN,
  resolveFnBody,
} = shared;

// Density-path recognition of a generative-composite measure (spec §06
// case-3 opt-in; engine-concepts §6): re-express it as the canonical
// `iid(mcmarginal{…}, n)` MC form (mc-recipe.buildMcMarginalForm) so the
// density walker scores it via walkMcMarginal. Returns null when the
// measure isn't that shape — the caller keeps its closed-form / refusal
// path. DENSITY-path only; sampling is untouched (the dissolver-side
// unification that feeds the same form to sampling is a later step).
function mcDensityForm(measureIR: any, ctx: any): any {
  try {
    return mcRecipe.buildMcMarginalForm(measureIR, ctx.bindings,
      (name: string) => orchestrator.expandMeasure(name,
        { derivations: ctx.derivations, bindings: ctx.bindings })) || null;
  } catch (_) { return null; }
}

// Marginalisation count + a stable MC seed for the worker's logDensityN
// (a profile sweep reuses common random numbers → a smooth curve). Pure
// closed-form density paths never see these.
function mcDensityOpts(ctx: any): any {
  return {
    mcMarginalizationCount: (ctx.marginalizationCount | 0)
      || (ctx.MARGINALIZATION_COUNT | 0) || 100,
    mcSeed: nameSeed('%mcdensity', ctx.rootKey),
  };
}

// =====================================================================
// matScore — the ONE CLM density consumer (engine-concepts §4 / the
// measure-lowering unification Phase 2-3). Feed a clm node's inputs
// (clm.feedInputs), score its body via the worker's logDensityN, and
// apply the marginal reduce when the node declares one. Collapses the
// bespoke refArrays plumbing that matBayesupdate / matLogdensityof /
// matBroadcastLogdensity each carried (and drifted). Returns the raw
// worker reply (per-atom log-densities in reply.samples) for reduce=null;
// callers that need the reduced scalar pass through `applyReduce`.
// =====================================================================

function matScore(node: any, ctx: any, opts?: any): Promise<any> {
  opts = opts || {};
  // Normalize-mass specs (audit M3) resolve on every matScore path
  // including bayesupdate: the resolution is a pure correctness fix
  // (−log Z literalised from the inner measure's tracked logTotalmass),
  // and walkNormalize refuses an unresolved spec loudly. node.body is
  // freshly built per lowerMeasure call (no cache), so the in-place
  // resolution is safe (mutation hazard — plan critique).
  // A measure body may call standard-module functions (e.g. the
  // `hepphys.interp_*` systematics in a HistFactory expected-rate); the
  // worker resolves `(call target={ns,…})` through `env.__moduleRegistry`,
  // so thread it (idempotent) before scoring.
  const bodyP = pushModuleRegistry(ctx).then(() => resolveNormalizeMasses(node.body, ctx));
  return bodyP.then((body: any) =>
    clm.feedInputs(node, ctx).then((fed: any) => {
      // extraRefArrays carries columns the declared inputs can't source via a
      // plain getMeasure — the N-ary kchain retained-history variates s_i,
      // materialised from node.marginalHistoryBody (matLogdensityof). They
      // override the fed entries.
      const refArrays = opts.extraRefArrays
        ? Object.assign({}, fed.refArrays, opts.extraRefArrays)
        : fed.refArrays;
      // Phase-6 unfed-boundary throw (plan critique F): the worker cannot
      // distinguish a shared ref from an unfed boundary, so verify HERE —
      // after the overlay merge — that every body-referenced boundary input
      // has a fed column. Without this, a feed gap either dies cryptically
      // in the worker or silently reads the stale session env (audit H4).
      clm.assertFedCoverage(node, refArrays, 'matScore');
      return pushFixedEnv(ctx, fed.fixedEnv).then(() => ctx.sendWorker({
        type: 'logDensityN',
        ir: body,
        count: ctx.sampleCount,
        refArrays,
        observed: opts.observed,
        tally: 'clamped',
        // body carries an mcmarginal recipe ⇒ thread the MC marginalisation
        // count + common-random-number seed.
        ...(node.mc ? mcDensityOpts(ctx) : {}),
      }));
    }));
}

// Apply a clm node's marginal reduction to the worker's per-atom log-densities:
// logsumexp_i − log N (engine-concepts §6). Broadcasts the scalar marginal back
// across N so the result is a plain scalar measure. reduce=null ⇒ pass through.
function applyReduce(reply: any, node: any): any {
  // Surface a worker-side density failure loudly instead of dereferencing a
  // missing `samples` (which masked the real error as a cryptic
  // "Cannot read properties of undefined (reading 'length')").
  if (reply && reply.type === 'error') {
    throw new Error('density: worker failed scoring likelihood body: '
      + (reply.message || 'unknown worker error'));
  }
  const perAtom = reply.samples;
  if (!node.reduce || node.reduce.kind !== 'marginal') {
    return scalarMeasureN(perAtom, {
      logWeights: null, logTotalmass: 0, n_eff: perAtom.length,
    });
  }
  const margLogp = empirical.logSumExp(perAtom) - Math.log(perAtom.length);
  const out = new Float64Array(perAtom.length);
  out.fill(margLogp);
  return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: 1 });
}

// =====================================================================
// Bayesupdate — reweight prior atoms by per-atom log-likelihood
// =====================================================================

function matBayesupdate(d: DerivationBayesupdate, ctx: any) {
  // backend:'mh' (or 'nuts') — run Metropolis-Hastings on the latents, then
  // return an equal-weight empirical measure. IS path (default) is below.
  const backend = (ctx.inferenceOpts && ctx.inferenceOpts.backend) || 'is';
  if (backend === 'mh' || backend === 'nuts' || backend === 'emcee' || backend === 'amis' || backend === 'smc' || backend === 'elliptical-slice-sampler') {
    const MV              = require('./model-view.ts');
    const driver          = require('./mcmc-driver.ts');
    const { mhKernel }         = require('./mh-kernel.ts');
    const { makeEmceeKernel }  = require('./emcee-kernel.ts');
    const { makeEllipticalSliceKernel } = require('./elliptical-slice-kernel.ts');
    // Use the new async vector-aware ModelView (wires mcmc-density.ts scorer).
    return MV.buildModelViewFromCtx(ctx, d).then((mv: any) => {
      if (backend === 'nuts' && mv.hasDiscrete) {
        return Promise.reject(new Error(
          "backend 'nuts' cannot sample discrete latents; use 'mh' or 'is'",
        ));
      }
      const o = ctx.inferenceOpts || {};

      // post: { drawsByName (per-coord, length nDraws), diagnostics }. MCMC
      // backends collect equal-weight chains; AMIS produces weighted IS samples
      // re-expressed as the same per-coordinate constrained draws so the output
      // reshape below is shared. `idx` resamples to exactly `ctx.sampleCount`
      // atoms — the count IS measures carry — so the result is a true drop-in
      // (the viewer treats a measure whose length ≠ SAMPLE_COUNT as a constant
      // literal; render-record.measureIsConstant). For MCMC `idx` is a uniform
      // stratified map; for AMIS it is a WEIGHTED systematic resample over the
      // importance weights, so the equal-weight output represents q-weighted π.
      let post: any, nDraws: number, idx: Int32Array;
      const total0 = (ctx.sampleCount | 0);

      const onProgress = typeof ctx.onProgress === 'function' ? ctx.onProgress : null;
      if (backend === 'amis') {
        const { amisSample } = require('./amis-sample.ts');
        const res = amisSample(mv, Object.assign({}, o, { onProgress }));
        nDraws = res.samples.length;
        // Per-coordinate constrained draws from the unconstrained samples.
        const drawsByName: Record<string, Float64Array> = {};
        for (let d2 = 0; d2 < mv.dim; d2++) drawsByName[mv.names[d2]] = new Float64Array(nDraws);
        for (let a = 0; a < nDraws; a++) {
          const flat = mv.constrainAll(res.samples[a]);
          for (let d2 = 0; d2 < mv.dim; d2++) drawsByName[mv.names[d2]][a] = flat[mv.names[d2]];
        }
        // Self-normalised weights + Kish ESS = (Σw)² / Σw² = 1 / Σ w̄².
        const lw = res.logW; let mx = -Infinity;
        for (let i = 0; i < nDraws; i++) if (lw[i] > mx) mx = lw[i];
        let sw = 0; for (let i = 0; i < nDraws; i++) sw += Math.exp(lw[i] - mx);
        const wbar = new Float64Array(nDraws); let sumSq = 0;
        for (let i = 0; i < nDraws; i++) { wbar[i] = Math.exp(lw[i] - mx) / sw; sumSq += wbar[i] * wbar[i]; }
        const ess = sumSq > 0 ? 1 / sumSq : 0;
        const total = total0 > 0 ? total0 : nDraws;
        // Systematic resample of `total` indices over the weight CDF.
        idx = new Int32Array(total);
        let c = wbar[0], j = 0;
        for (let i = 0; i < total; i++) {
          const u = (i + 0.5) / total;
          while (u > c && j < nDraws - 1) { j++; c += wbar[j]; }
          idx[i] = j;
        }
        const perParam: Record<string, any> = {};
        for (let d2 = 0; d2 < mv.dim; d2++) perParam[mv.names[d2]] = { rHat: NaN, essBulk: ess };
        post = { drawsByName, diagnostics: { method: 'amis', ess, essFrac: ess / nDraws, K: res.K, nSamples: nDraws, perParam } };
      } else if (backend === 'smc') {
        const { smcSample } = require('./smc-sample.ts');
        const res = smcSample(mv, Object.assign({}, o, { onProgress }));
        nDraws = res.samples.length;
        // SMC's final particles are EQUAL weight (already resampled), so build
        // per-coordinate draws and resample uniformly (like MH/emcee).
        const drawsByName: Record<string, Float64Array> = {};
        for (let d2 = 0; d2 < mv.dim; d2++) drawsByName[mv.names[d2]] = new Float64Array(nDraws);
        for (let a = 0; a < nDraws; a++) {
          const flat = mv.constrainAll(res.samples[a]);
          for (let d2 = 0; d2 < mv.dim; d2++) drawsByName[mv.names[d2]][a] = flat[mv.names[d2]];
        }
        const total = total0 > 0 ? total0 : nDraws;
        idx = new Int32Array(total);
        for (let i = 0; i < total; i++) idx[i] = Math.floor((i * nDraws) / total) % nDraws;
        post = { drawsByName, diagnostics: { method: 'smc', logZ: res.logZ, rungs: res.rungs, acceptRate: res.acceptRate, nSamples: nDraws } };
      } else {
        const isEss = backend === 'elliptical-slice-sampler';
        const kernel = backend === 'emcee' ? makeEmceeKernel(o.a)
          : isEss ? makeEllipticalSliceKernel() : mhKernel;
        const nWalkers = o.walkers ?? o.chains ?? (backend === 'emcee' ? Math.max(4, 2 * mv.dim + 2) : 4);
        post = driver.runMcmc(mv, kernel, {
          nWalkers, warmup: o.warmup ?? 1000, draws: o.draws ?? 1000, seed: (o.seed ?? 0), a: o.a, essMaxShrink: o.essMaxShrink, onProgress,
        });
        // post.diagnostics is { acceptRate, perParam:{name:{rHat,essBulk}} } — attach as-is.
        nDraws = post.drawsByName[mv.names[0]].length;
        // Record the TRUE draw count so the viewer shows it rather than the
        // sampleCount the output is resampled to for plotting.
        post.diagnostics.nSamples = nDraws;
        const total = total0 > 0 ? total0 : nDraws;
        idx = new Int32Array(total);
        for (let i = 0; i < total; i++) idx[i] = Math.floor((i * nDraws) / total) % nDraws;
        if (isEss) {
          // Elliptical slice is near-rejection-free; the driver's acceptRate is
          // really (sweeps·N)/(shrink evals) = 1/mean-shrinks-per-step. Re-tag.
          const ar = post.diagnostics.acceptRate;
          post.diagnostics.method = 'ess-slice';
          post.diagnostics.mode = mv.gaussianPrior ? 'exact' : 'fitted';
          post.diagnostics.meanShrinks = (ar > 0) ? 1 / ar : NaN;
        }
      }
      const total = idx.length;

      // Reconstruct the draw scorer-point at atom `a` (group vector coords).
      const drawPtAt = (a: number) => {
        const pt: any = {};
        for (let li = 0; li < mv.latentNames.length; li++) {
          const nm = mv.latentNames[li], shp = mv.latentShapes[li];
          if (shp.kind === 'scalar') { pt[nm] = post.drawsByName[nm][a]; }
          else {
            const dd = shp.dims[0]; const v = new Float64Array(dd);
            for (let j = 0; j < dd; j++) v[j] = post.drawsByName[`${nm}[${j}]`][a];
            pt[nm] = v;
          }
        }
        return pt;
      };

      // Output = the PRIOR RECORD's fields, evaluated from the sampled draws —
      // identical field names/shapes to the IS posterior (drop-in). The fields
      // may be draws (alpha, beta, mu, tau, theta) OR deterministic transforms
      // of draws (sigma = sqrt(sigma2)); evaluateDerivedBindings yields all of
      // them in one env per atom. We fetch the prior measure ONLY for its field
      // structure (names + per-field shape), then fill with MCMC values.
      const mcmcDensity = require('./mcmc-density.ts');
      return Promise.resolve(ctx.getMeasure(d.from)).then((parent: any) => {
        const buildScalar = (stream: Float64Array) => {
          const out = new Float64Array(total);
          for (let i = 0; i < total; i++) out[i] = stream[idx[i]];
          return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: nDraws });
        };
        const buildArray = (flat: Float64Array, dlen: number) => {
          const s = new Float64Array(total * dlen);
          for (let i = 0; i < total; i++) {
            const a = idx[i];
            for (let j = 0; j < dlen; j++) s[i * dlen + j] = flat[a * dlen + j];
          }
          const am = empirical.arrayMeasure(s, [dlen], null);
          am.logTotalmass = 0; am.n_eff = nDraws;
          return am;
        };

        if (parent && parent.fields) {
          // Build each prior-record field, resampled to `total`. A field that IS
          // a latent (mu, tau, theta) is read DIRECTLY from the sampled draws —
          // no per-atom evaluation. Only genuinely-derived fields (e.g.
          // sigma = sqrt(sigma2)) need the env, and then we evaluate ONLY those,
          // and only over the unique resampled atoms — not all nDraws × every
          // binding (the per-atom buildEnv that made this O(nDraws) interpreted).
          const fieldNames = Object.keys(parent.fields);
          const latentShape: Record<string, any> = {};
          for (let li = 0; li < mv.latentNames.length; li++) latentShape[mv.latentNames[li]] = mv.latentShapes[li];
          const fields: any = {};
          const derivedFields: string[] = [];
          for (const fn of fieldNames) {
            const shp = latentShape[fn];
            if (!shp) { derivedFields.push(fn); continue; }
            if (shp.kind === 'scalar') {
              fields[fn] = buildScalar(post.drawsByName[fn]);
            } else {
              const dlen = shp.dims[0];
              const flat = new Float64Array(nDraws * dlen);
              for (let a = 0; a < nDraws; a++) for (let j = 0; j < dlen; j++) flat[a * dlen + j] = post.drawsByName[`${fn}[${j}]`][a];
              fields[fn] = buildArray(flat, dlen);
            }
          }
          if (derivedFields.length > 0) {
            // Evaluate derived fields only at the atoms the resample actually
            // selects (deduped), then scatter back through `idx`.
            const uniq: number[] = []; const seen = new Int32Array(nDraws).fill(0);
            for (let i = 0; i < total; i++) { const a = idx[i]; if (!seen[a]) { seen[a] = 1; uniq.push(a); } }
            const acc: any = {};
            for (const a of uniq) {
              const env = mcmcDensity.evaluateDerivedBindings(ctx, drawPtAt(a));
              for (const fn of derivedFields) {
                const v = env[fn];
                if (!(fn in acc)) acc[fn] = (typeof v === 'number')
                  ? { scalar: new Float64Array(nDraws) }
                  : { vec: new Float64Array(nDraws * ((v && v.length) || 1)), d: (v && v.length) || 1 };
                const e = acc[fn];
                if (e.scalar) { e.scalar[a] = (typeof v === 'number') ? v : NaN; }
                else { for (let j = 0; j < e.d; j++) e.vec[a * e.d + j] = (v && v[j] != null) ? +v[j] : NaN; }
              }
            }
            for (const fn of derivedFields) {
              const e = acc[fn];
              fields[fn] = e.scalar ? buildScalar(e.scalar) : buildArray(e.vec, e.d);
            }
          }
          const recM = empirical.recordMeasure(fields, null);
          recM.diagnostics = post.diagnostics;
          return recM;
        }

        // Non-record (single scalar/array latent) prior: the sole draw is the
        // measure. Scalar → scalar measure; vector → array measure.
        const nm = mv.latentNames[0], shp = mv.latentShapes[0];
        let m: any;
        if (shp.kind === 'scalar') {
          m = buildScalar(post.drawsByName[nm]);
        } else {
          const dlen = shp.dims[0];
          const flat = new Float64Array(nDraws * dlen);
          for (let a = 0; a < nDraws; a++) {
            for (let j = 0; j < dlen; j++) flat[a * dlen + j] = post.drawsByName[`${nm}[${j}]`][a];
          }
          m = buildArray(flat, dlen);
        }
        m.diagnostics = post.diagnostics;
        return m;
      });
    });
  }

  // IS path (default, untouched below).
  // Per spec: posterior = bayesupdate(L, prior), L = likelihoodof(K, obs).
  // For each prior atom θ_i, logw_i = logdensityof(K(θ_i), obs); the atoms
  // are the prior's, logWeights = prior.logWeights + per-i logp.
  //
  // The likelihood body is lowered ONCE to a clm node: lowerMeasure peels +
  // expands, inlines derived value bindings to the boundary inputs (H5/H3) or
  // applies the generative MC form (§06 case-3), and declares the kernel's
  // parametric inputs as boundaries fed from the prior — the spec lowering
  // (bayesupdate(L,prior) makes the prior's variate the kernel's parametric
  // input) instead of re-materialising a like-named binding via getMeasure
  // (the boundary-conflation bug, audit §3 / H1/H6). matScore + clm.feedInputs
  // own the feeding; this function keeps only the reweighting.
  const node = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d });
  if (!node) {
    return Promise.reject(new Error('bayesupdate: cannot expand body into measure IR'));
  }
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
  return Promise.all([
    Promise.resolve(ctx.getMeasure(d.from)),
    matScore(node, ctx, { observed }),
  ]).then(([parent, reply]: any[]) => {
    const N = measureN(parent);
    const existingLW = parent.logWeights;
    const uniformLW = -Math.log(N);
    const newLW = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const base = existingLW ? existingLW[i] : uniformLW;
      newLW[i] = base + reply.samples[i];
    }
    const lTM = empirical.logSumExp(newLW);
    const nEff = empirical.effectiveSampleSize({ samples: parent.samples || new Float64Array(N), logWeights: newLW });
    if (parent.fields) {
      return Object.assign(
        empirical.recordMeasure(parent.fields, newLW),
        { logTotalmass: lTM, n_eff: nEff },
      );
    }
    return scalarMeasureN(parent.samples, {
      logWeights: newLW,
      logTotalmass: lTM,
      n_eff: nEff,
    });
  });
}

// =====================================================================
// Standalone likelihood density — pdf(κ(θ), obs) (spec §06, audit H2)
// =====================================================================

function matLikelihoodDensity(d: any, ctx: any) {
  // logdensityof(L, θ) with L = likelihoodof(K, obs): score the kernel at
  // the GIVEN θ against the FIXED obs. The body is lowered ONCE via the
  // same lowerMeasure call matBayesupdate uses (derived value bindings
  // inline down to the kernel's parametric inputs, H5/H3; a generative
  // body takes the MC-marginal form), but the parametric inputs are fed
  // EXPLICITLY from θ (opts.boundaries → feedInputs' explicit path —
  // an atom-independent value broadcast across N) instead of from a
  // prior's atoms. Result is a per-atom scalar (constant across atoms,
  // matching the matTotalmass broadcast convention).
  const theta = orchestrator.resolveIRToValue(d.pointIR, ctx.bindings, ctx.fixedValues);
  if (theta == null) {
    return Promise.reject(new Error(
      'logdensityof(L, θ): cannot resolve θ to a concrete value'));
  }
  // Map θ onto the kernel's parametric inputs: a single input takes the
  // whole value; multiple inputs require a record with one field each
  // (the kernel's input space is the cat of its params, spec §04).
  const kwargs: string[] = d.paramKwargs || [];
  const boundaries: Record<string, any> = {};
  // A resolved record θ is a plain object (no Value `.shape`/`.data`); a
  // scalar/vector point is a number / array / Value.
  const isRecord = theta !== null && typeof theta === 'object'
    && !Array.isArray(theta)
    && (theta as any).shape === undefined && (theta as any).data === undefined;
  if (kwargs.length === 1 && !(isRecord && (kwargs[0] in (theta as any)))) {
    // Single kernel input fed the whole θ value (scalar/vector point).
    boundaries[kwargs[0]] = theta;
  } else {
    // Record θ: project each kernel input from its field (the multi-param
    // case, and any single-param term whose θ is a record — e.g. a
    // joint_likelihood sub-term declaring fewer params than θ carries).
    for (const k of kwargs) {
      if (!isRecord || !(k in (theta as any))) {
        return Promise.reject(new Error(
          'logdensityof(L, θ): θ must be a record with a "' + k
          + '" field (kernel inputs: ' + kwargs.join(', ') + ')'));
      }
      boundaries[k] = (theta as any)[k];
    }
  }
  const node = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d, boundaries });
  if (!node) {
    return Promise.reject(new Error(
      'logdensityof(L, θ): cannot expand the likelihood body into measure IR'));
  }
  // §12 shared-variate product_dist: the body is
  // normalize(logweighted(x -> Σ logdensityof(Mᵢ, x), M0)). The normalizer
  // −log Z = −log ∫ ∏ᵢ gᵢ has no massFrom (it isn't a reweighted ensemble),
  // so resolve it HERE — at the fixed point θ — and rewrite the normalize to a
  // constant logweighted shift, before scoring.
  resolveProductNormalizers(node.body, theta, ctx);
  // normalize(truncate(M, S)): the inline support-restriction normalizer. Per
  // spec §06 truncate carries M(S) as its totalmass and normalize divides by
  // it — but an INLINE normalize(truncate(…)) (the kernel-density Path B body,
  // M's params open as the fed θ) reaches the scorer without a massFrom ref the
  // by-name resolveNormalizeMasses can materialise. −log Z = −log M(S) is a
  // function of θ, so resolve it HERE at the fixed point, the same way the §12
  // product normalizer above does, before scoring.
  resolveTruncateNormalizers(node.body, theta, ctx);
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
  return matScore(node, ctx, { observed })
    .then((reply: any) => applyReduce(reply, node));
}

// Walk a measure IR and rewrite every inline support-restriction normalizer
// `normalize(truncate(M, interval(lo, hi)))` to the constant shift
// `logweighted(−log Z, truncate(M, …))` evaluated at the point θ, where
// Z = M([lo, hi]) is the base measure's mass over the truncation set. The
// truncate node is kept as the inner (its walkTruncate gates out-of-support
// points to −∞); only the missing −log Z normalization factor is supplied.
function resolveTruncateNormalizers(node: any, theta: any, ctx: any, seen?: any) {
  if (!node || typeof node !== 'object') return;
  seen = seen || new Set();
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) resolveTruncateNormalizers(x, theta, ctx, seen);
    return;
  }
  if (node.kind === 'call' && node.op === 'normalize'
      && !node.massFrom
      && Array.isArray(node.args) && node.args.length === 1) {
    const inner = node.args[0];
    if (inner && inner.kind === 'call' && inner.op === 'truncate'
        && Array.isArray(inner.args) && inner.args.length === 2) {
      const bounds = truncateSetBounds(inner.args[1]);
      if (bounds) {
        // A generic_dist (§12) lowers to
        // `normalize(truncate(weighted(w, Lebesgue(reals)), S))`: its base
        // is an unnormalized density `w(x)` over the bounded set S, NOT a
        // scalar reference measure. Recognise the weighted base first and
        // quadrature its support-restricted normalizer Z = ∫_S w(x) dx;
        // a recognised scalar reference measure (Normal closed-form / its
        // narrow midpoint path) falls through to truncateLogMass.
        const weightFn = weightedBaseWeightFn(inner.args[0]);
        const logZ = weightFn != null
          ? weightedBaseLogMass(weightFn, bounds, theta)
          : (() => {
              const base = asScalarFactor(inner.args[0], theta);
              if (!base) {
                throw new Error('density: normalize(truncate(M, S)) — base '
                  + 'measure M is not a recognised scalar reference measure '
                  + 'nor a weighted(<weight-fn>, Lebesgue) density, cannot '
                  + 'resolve its support-restricted normalizer');
              }
              return truncateLogMass(base, bounds, ctx);
            })();
        node.op = 'logweighted';
        node.args = [{ kind: 'lit', value: -logZ }, inner];
        resolveTruncateNormalizers(inner, theta, ctx, seen);
        return;
      }
    }
  }
  for (const k in node) {
    const v = node[k];
    if (v && typeof v === 'object') resolveTruncateNormalizers(v, theta, ctx, seen);
  }
}

// Resolve an `interval(lo, hi)` set IR (literal bounds) to [lo, hi]; null for
// any other set shape (dynamic / named sets defer to the by-name materialiser).
function truncateSetBounds(setIR: any): [number, number] | null {
  if (!setIR || setIR.kind !== 'call' || setIR.op !== 'interval'
      || !Array.isArray(setIR.args) || setIR.args.length !== 2) return null;
  const lo = setIR.args[0], hi = setIR.args[1];
  if (lo && lo.kind === 'lit' && hi && hi.kind === 'lit') return [+lo.value, +hi.value];
  return null;
}

// log Z = log ∫_lo^hi f_base(x) dx for a scalar reference-measure base resolved
// at θ. Closed-form via the CDF for a Normal base (exact); composite-midpoint
// quadrature of the base log-density otherwise (matching numericProductLogZ).
function truncateLogMass(base: any, bounds: [number, number], ctx: any): number {
  const [lo, hi] = bounds;
  if (base.kernel === 'Normal') {
    const mu = +base.input.mu, sigma = +base.input.sigma;
    const normalCdf = require('@stdlib/stats-base-dists-normal-cdf');
    const Z = normalCdf(hi, mu, sigma) - normalCdf(lo, mu, sigma);
    if (!(Z > 0)) {
      throw new Error('density: normalize(truncate(Normal, S)) — mass over the '
        + 'truncation set is 0 (Z = 0 is undefined per spec §06)');
    }
    return Math.log(Z);
  }
  const dx = (hi - lo) / QUAD_POINTS;
  const logIntegrand = new Float64Array(QUAD_POINTS);
  for (let j = 0; j < QUAD_POINTS; j++) {
    const x = lo + (j + 0.5) * dx;
    logIntegrand[j] = densityPrims.builtinLogdensityof(base.kernel, base.input, x);
  }
  return empirical.logSumExp(logIntegrand) + Math.log(dx);
}

// Recognise a `weighted(<weight-fn>, Lebesgue(reals|interval))` base — the
// §12 generic_dist lowering — and return its weight FUNCTION as a single-
// parameter `functionof` IR { paramName, body }; null for any other base.
// The Lebesgue density is ≡ 1, so the base density is the weight `w` itself
// and Z = ∫_S w(x) dx. A constant-weight `weighted` (no function, just a
// log-mass shift) is NOT this shape and returns null (it falls through to
// the scalar-reference path, which already handles plain reference bases).
function weightedBaseWeightFn(baseIR: any): { paramName: string; body: any } | null {
  if (!baseIR || baseIR.kind !== 'call' || baseIR.op !== 'weighted'
      || !Array.isArray(baseIR.args) || baseIR.args.length !== 2) return null;
  const ref = baseIR.args[1];
  // Reference measure must be Lebesgue (its density ≡ 1, so the weight is
  // the full density). `Lebesgue(reals)` or `Lebesgue(interval(...))`.
  if (!ref || ref.kind !== 'call' || ref.op !== 'Lebesgue') return null;
  const fn = baseIR.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length !== 1 || !fn.body) {
    return null;
  }
  return { paramName: fn.params[0], body: fn.body };
}

// log Z = log ∫_lo^hi w(x) dx for a `weighted(w, Lebesgue)` base, where `w`
// is the (non-log) density. Composite-midpoint quadrature over S at the
// fixed point θ, mirroring numericProductLogZ's QUAD_POINTS midpoint rule —
// but evaluating the weight expression `w(x; θ)` per node via the sampler's
// per-expr evaluator (the variate param bound to the node, θ's params bound
// to their values). The weight is the density, NOT a log-density, so it is
// summed directly (not exponentiated): Z = Σ w(x_j) · dx.
function weightedBaseLogMass(
  weightFn: { paramName: string; body: any }, bounds: [number, number], theta: any,
): number {
  const samplerLib = require('./sampler.ts');
  const [lo, hi] = bounds;
  const dx = (hi - lo) / QUAD_POINTS;
  // env binds θ's free params (e.g. `alpha`) plus the variate param to the
  // current quadrature node; evaluateExpr resolves refs by name (ns-agnostic).
  const env: Record<string, any> = {};
  if (theta && typeof theta === 'object') {
    for (const k in theta) env[k] = theta[k];
  }
  let Z = 0;
  let nonFinite = 0;
  for (let j = 0; j < QUAD_POINTS; j++) {
    env[weightFn.paramName] = lo + (j + 0.5) * dx;
    const wj = samplerLib.evaluateExpr(weightFn.body, env);
    const v = +wj;
    if (Number.isFinite(v) && v > 0) Z += v;
    else if (v < 0 || !Number.isFinite(v)) nonFinite++;
  }
  if (nonFinite > 0) {
    // eslint-disable-next-line no-console
    console.warn('normalize(truncate(weighted(w, Lebesgue), S)): ' + nonFinite
      + ' quadrature node(s) with non-positive/non-finite weight treated as 0');
  }
  if (!(Z > 0)) {
    throw new Error('density: normalize(truncate(weighted(w, Lebesgue), S)) — '
      + 'normalizer Z = ∫_S w dx is 0 (an everywhere-zero weight over S has no '
      + 'normalized density per spec §06)');
  }
  return Math.log(Z) + Math.log(dx);
}

// =====================================================================
// Shared-variate product_dist normalizer (§12)
// =====================================================================

// Walk a measure IR and rewrite every shared-variate product_dist normalizer
// `normalize(logweighted(functionof(Σ logdensityof(Mᵢ, x)), M0))` to the
// constant shift `logweighted(−logZ, <inner>)` evaluated at the point `theta`.
// −logZ is closed-form when every factor is a Normal over the shared variate
// (the product of Gaussians is Gaussian); other recognised reference measures
// fall to numeric quadrature. An unrecognised factor mix throws (loud, not a
// silently-unnormalized density).
function resolveProductNormalizers(node: any, theta: any, ctx: any, seen?: any) {
  if (!node || typeof node !== 'object') return;
  seen = seen || new Set();
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) resolveProductNormalizers(x, theta, ctx, seen);
    return;
  }
  if (node.kind === 'call' && node.op === 'normalize'
      && Array.isArray(node.args) && node.args.length === 1) {
    const inner = node.args[0];
    const fold = sharedVariateProductFold(inner);
    if (fold) {
      const negLogZ = -productLogZ(fold.factors, fold.variate, theta, ctx);
      // Rewrite IN PLACE: normalize(inner) → logweighted(−logZ, inner).
      node.op = 'logweighted';
      node.args = [{ kind: 'lit', value: negLogZ }, inner];
      delete node.massFrom;
      resolveProductNormalizers(inner, theta, ctx, seen);
      return;
    }
  }
  for (const k in node) {
    const v = node[k];
    if (v && typeof v === 'object') resolveProductNormalizers(v, theta, ctx, seen);
  }
}

// Recognise `logweighted(functionof(Σ logdensityof(Mᵢ, x)), M0)` and return the
// ordered factor measure IRs [M0, M1, …] plus the shared variate name; null if
// it isn't that shape.
function sharedVariateProductFold(inner: any): any {
  if (!inner || inner.kind !== 'call' || inner.op !== 'logweighted'
      || !Array.isArray(inner.args) || inner.args.length !== 2) return null;
  const fn = inner.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof' || !fn.body) return null;
  const factors = [inner.args[1]];
  const collect = (n: any): boolean => {
    if (n && n.kind === 'call' && n.op === 'add'
        && Array.isArray(n.args) && n.args.length === 2) {
      return collect(n.args[0]) && collect(n.args[1]);
    }
    if (n && n.kind === 'call' && n.op === 'logdensityof'
        && Array.isArray(n.args) && n.args.length === 2) {
      factors.push(n.args[0]);
      return true;
    }
    return false;
  };
  const variate = Array.isArray(fn.paramKwargs) ? fn.paramKwargs[0] : undefined;
  return collect(fn.body) ? { factors, variate } : null;
}

// log Z for a shared-variate product of reference measures, at point `theta`.
// Closed form when every factor is a Normal (the product of Gaussians is
// Gaussian); otherwise numeric quadrature of ∫ ∏ᵢ gᵢ over the variate's
// declared domain.
function productLogZ(factorIRs: any[], variate: any, theta: any, ctx: any): number {
  const kernels = factorIRs.map((m) => asScalarFactor(m, theta));
  if (kernels.some((k) => k == null)) {
    throw new Error('density: shared-variate product_dist factor is not a '
      + 'recognised scalar reference measure — cannot resolve its normalizer');
  }
  if (kernels.every((k: any) => k.kernel === 'Normal')) {
    // Product of Gaussians is Gaussian: log ∫ ∏ N(x|μᵢ,σᵢ) dx.
    //   τ = Σ 1/σᵢ²,  μ* = (Σ μᵢ/σᵢ²)/τ,  C = Σ μᵢ²/σᵢ² − τ·μ*²
    //   logZ = −½ Σ ln(2π σᵢ²) + ½ ln(2π/τ) − ½ C
    let tau = 0, wmu = 0, sumMuSqOverVar = 0, sumLogTerm = 0;
    for (const k of kernels as any[]) {
      const mu = +k.input.mu, sigma = +k.input.sigma;
      const inv = 1 / (sigma * sigma);
      tau += inv;
      wmu += mu * inv;
      sumMuSqOverVar += mu * mu * inv;
      sumLogTerm += Math.log(2 * Math.PI * sigma * sigma);
    }
    const muStar = wmu / tau;
    const C = sumMuSqOverVar - tau * muStar * muStar;
    return -0.5 * sumLogTerm + 0.5 * Math.log(2 * Math.PI / tau) - 0.5 * C;
  }
  return numericProductLogZ(kernels, variate, ctx);
}

// Numeric log ∫ ∏ᵢ gᵢ(x) dx over the variate's declared domain [lo, hi], by
// composite-midpoint quadrature of the (log-summed) integrand. The declared
// domain (HS3 `domains`) is the same bounded support ROOT integrates over.
const QUAD_POINTS = 8192;
function numericProductLogZ(kernels: any[], variate: any, ctx: any): number {
  const dom = resolveVariateInterval(ctx, variate);
  if (!dom) {
    throw new Error('density: numeric shared-variate product_dist normalizer '
      + 'needs the variate `' + variate + '` declared domain (an interval in a '
      + 'cartprod), which is not present in the model');
  }
  const [lo, hi] = dom;
  const dx = (hi - lo) / QUAD_POINTS;
  const logIntegrand = new Float64Array(QUAD_POINTS);
  for (let j = 0; j < QUAD_POINTS; j++) {
    const x = lo + (j + 0.5) * dx;
    let s = 0;
    for (const k of kernels as any[]) {
      s += densityPrims.builtinLogdensityof(k.kernel, k.input, x);
    }
    logIntegrand[j] = s;
  }
  // log ∫ ≈ log( Σ exp(logf_j) · dx ) = logsumexp(logf) + log(dx).
  return empirical.logSumExp(logIntegrand) + Math.log(dx);
}

// If `m` (a factor measure IR) is a scalar reference-measure kernel over the
// shared variate, return { kernel, input } where `input` is the kernel-input
// record resolved at `theta` (a plain {param: value} map). Peels the
// record/relabel wrapper the §12 lowering may put around a variate-keyed
// factor. Returns null for an unrecognised / multivariate factor.
function asScalarFactor(m: any, theta: any): any {
  let k = m;
  while (k && k.kind === 'call' && (k.op === 'relabel' || k.op === 'record')) {
    if (k.op === 'relabel') { k = k.args && k.args[0]; continue; }
    if (Array.isArray(k.fields) && k.fields.length === 1) { k = k.fields[0].value; continue; }
    return null;
  }
  if (!k || k.kind !== 'call' || typeof k.op !== 'string' || !k.kwargs) return null;
  const input: Record<string, number> = {};
  for (const name in k.kwargs) {
    const v = resolveScalarAtPoint(k.kwargs[name], theta);
    if (!Number.isFinite(v)) return null;
    input[name] = v;
  }
  return { kernel: k.op, input };
}

// Resolve the variate's declared domain [lo, hi] from a `cartprod` binding
// whose field for `variate` is an `interval(lo, hi)` (HS3 `domains`). Returns
// null if no such interval is declared.
function resolveVariateInterval(ctx: any, variate: any): [number, number] | null {
  if (!ctx || !ctx.bindings || typeof ctx.bindings.forEach !== 'function' || variate == null) {
    return null;
  }
  let found: [number, number] | null = null;
  ctx.bindings.forEach((binding: any) => {
    if (found) return;
    const ir = binding && (binding.ir || binding.rhs || binding.value);
    if (!ir || ir.kind !== 'call' || ir.op !== 'cartprod' || !Array.isArray(ir.fields)) return;
    for (const f of ir.fields) {
      if (f.name === variate && f.value && f.value.op === 'interval'
          && Array.isArray(f.value.args) && f.value.args.length === 2) {
        const lo = f.value.args[0], hi = f.value.args[1];
        if (lo && lo.kind === 'lit' && hi && hi.kind === 'lit') {
          found = [+lo.value, +hi.value];
          return;
        }
      }
    }
  });
  return found;
}

// Resolve a scalar kwarg IR (a literal or a parameter ref) against the fixed
// point θ (a record of the kernel's free parameters).
function resolveScalarAtPoint(ir: any, theta: any): number {
  if (!ir) return NaN;
  if (ir.kind === 'lit') return +ir.value;
  if (ir.kind === 'ref' && theta && typeof theta === 'object'
      && Object.prototype.hasOwnProperty.call(theta, ir.name)) {
    return +theta[ir.name];
  }
  return NaN;
}

function matJointLikelihoodDensity(d: any, ctx: any) {
  // logdensityof(joint_likelihood(L1, …, Lk), θ): a joint likelihood is the
  // product of independent terms, so its log-density at θ is the SUM of the
  // per-term log-densities (spec §06). Each sub is scored by the standalone
  // likelihood-density path; θ is projected onto each sub's own params
  // inside matLikelihoodDensity (a sub may declare fewer params than θ
  // carries — e.g. an auxiliary constraint touching one nuisance).
  const N = ctx.sampleCount;
  return Promise.all((d.subs || []).map((sub: any) =>
    matLikelihoodDensity({
      kind: 'likelihood_density',
      bodyName: sub.bodyName, bodyIR: sub.bodyIR, obsIR: sub.obsIR,
      paramKwargs: sub.paramKwargs, params: sub.params, pointIR: d.pointIR,
    }, ctx)
  )).then((measures: any[]) => {
    const out = new Float64Array(N);
    for (const m of measures) {
      const s = m.samples;
      for (let i = 0; i < N; i++) out[i] += s[i];
    }
    return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// Totalmass — surface the tracked logTotalmass as a per-atom value
// =====================================================================

function matTotalmass(d: DerivationTotalmass, ctx: any) {
  // totalmass(M): per spec §06, scalar mass of a (possibly
  // unnormalized) measure. The orchestrator tracks each measure's
  // logTotalmass through every materialisation step; expose it as a
  // per-atom scalar value — broadcast since we track a single ensemble
  // logTotalmass per measure today (per-atom tracking is a refinement).
  return ctx.getMeasure(d.measureName).then((m: any) => {
    const N = ctx.sampleCount;
    const tm = Math.exp(typeof m.logTotalmass === 'number' ? m.logTotalmass : 0);
    const samples = new Float64Array(N);
    samples.fill(tm);
    return scalarMeasureN(samples, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// Resolve-then-literalise specs (audit M3)
// =====================================================================
//
// (The former select runtime-weight resolver — a POOLED P̂(true)
// frequency estimated once from the selector ensemble — is retired:
// a deterministic-given-θ selector now carries EXACT per-atom
// indicator logweight IRs straight from classifyIfelse, audit M2.)

// normalize(M) nodes whose −log Z is NOT closed-form carry a massFrom
// spec ({ref: <inner binding>}) — see expandMeasure's normalize case
// (audit M3).
function collectNormalizeMassNodes(node: any, out: any, seen: any) {
  if (!node || typeof node !== 'object') return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) collectNormalizeMassNodes(x, out, seen);
    return;
  }
  if (node.kind === 'call' && node.op === 'normalize' && node.massFrom) {
    out.push(node);
  }
  for (const k in node) {
    const v = node[k];
    if (v && typeof v === 'object') collectNormalizeMassNodes(v, out, seen);
  }
}

// Resolve each massFrom-carrying normalize node by materialising its
// inner measure ONCE and reading the TRACKED logTotalmass (truncate's
// accept-rate estimate, weighted's algebraic shift, …), then rewrite the
// node IN PLACE to the same logweighted(−logZ, inner) form the
// closed-form expansion emits — one downstream representation, the
// worker never sees a mass-carrying normalize. The in-place mutation is
// safe because the body is freshly built per lowerMeasure call (no
// cache; the CLM mutation-hazard rule).
function resolveNormalizeMasses(measureIR: any, ctx: any) {
  const nodes: any[] = [];
  collectNormalizeMassNodes(measureIR, nodes, new Set());
  if (nodes.length === 0) return Promise.resolve(measureIR);
  const refNames = Array.from(new Set(nodes.map((n) => n.massFrom.ref)));
  return Promise.all(refNames.map((r) => Promise.resolve(ctx.getMeasure(r))))
    .then((measures: any[]) => {
      const byName: Record<string, any> = {};
      refNames.forEach((nm: string, i: number) => { byName[nm] = measures[i]; });
      for (const node of nodes) {
        const M = byName[node.massFrom.ref];
        const logZ = M && typeof M.logTotalmass === 'number' ? M.logTotalmass : null;
        if (logZ == null || !Number.isFinite(logZ)) {
          throw new Error('normalize density: inner measure "' + node.massFrom.ref
            + '" did not materialise to a finite tracked logTotalmass'
            + ' (Z = 0 or ∞ is undefined per spec §06)');
        }
        const inner = node.args[0];
        node.op = 'logweighted';
        node.args = [{ kind: 'lit', value: -logZ }, inner];
        delete node.massFrom;
      }
      return measureIR;
    });
}

// =====================================================================
// matLogdensityof — broadcast logdensityof over prior atoms
// =====================================================================

function matLogdensityof(d: DerivationLogdensityof, ctx: any) {
  // Per spec §sec:posterior. For each atom i of M, evaluate
  // logp(obs | M_i). Produces a per-i value (a scalar binding) — no
  // logWeights, no totalmass mutation.
  //
  // chain MARGINALISATION: when the measure was originally a
  // `kchain(prior, K)` (per spec §06 ν(B) = ∫ K(a, B) dμ(a)), the
  // per-atom log-likelihoods we compute below are exactly the
  // integrand evaluated at MC samples a_i ~ μ. The marginal
  // log-density of the chain at obs is logsumexp_i { log p_K(obs |
  // a_i) } − log N. We detect a kchain (marginalising) measure via
  // the first-class derivation (kind:'jointchain' with
  // marginalize:true).
  //
  // For N-ary kchain (engine-concepts §6 chain-associativity), the
  // retained (n−1)-joint history (node.marginalHistoryBody) materialises
  // ONCE and we bind its variate columns as the per-atom refArrays the
  // last kernel's hole-rewired cat consumes.
  // Lower the measure ONCE to its canonical form (lowerMeasure): peel/expand,
  // inline derived value bindings down to the boundary inputs OR apply the
  // generative MC marginalising-pushforward form (§06 case-3), declare the
  // prior / integration-variable boundary inputs (the M1/H1 fix — the kernel's
  // NAMED params score against the FED prior, not the like-named draw), and
  // carry reduce={marginal} for a kchain. The marginal reduction is now a
  // property of the node (applyReduce), not a re-derived isChain/naryKchain
  // flag scattered through this function.
  const node = clm.lowerMeasure(d.measureName, ctx);
  if (!node) {
    return Promise.reject(new Error('logdensityof: cannot expand measure "'
      + d.measureName + '" into a self-contained IR'));
  }
  // N-ary kchain (engine-concepts §6 chain-associativity): the retained
  // (n−1)-joint history's variate columns s_i can't be sourced by a plain
  // getMeasure, so materialise them ONCE and pass as extraRefArrays (the last
  // kernel's hole-rewired cat over s_i consumes them, overriding the boundary
  // feeds feedInputs anchors to base.ref). lowerMeasure attaches the retained
  // history body (node.marginalHistoryBody) — the SAME structure matClm
  // samples on the sample side — so density and sample reconstruct the history
  // through ONE mechanism (the dependent-threaded retain joint), not two.
  const innerJointP = node.marginalHistoryBody
    ? require('./materialiser.ts').materialiseMeasureIR(node.marginalHistoryBody, ctx)
    : Promise.resolve(null);
  const observed = orchestrator.resolveIRToValue(
    d.obsIR, ctx.bindings, ctx.fixedValues);
  return innerJointP.then((innerJoint: any) => {
    const extraRefArrays: Record<string, any> = {};
    if (innerJoint) {
      const comps = innerJoint.elems
        || (innerJoint.fields
          ? Object.keys(innerJoint.fields).map((k) => innerJoint.fields[k])
          : null);
      if (!comps) {
        throw new Error('logdensityof: N-ary kchain prior history did '
          + 'not materialise to a joint (tuple/record) measure');
      }
      for (let i = 0; i < comps.length; i++) {
        const cs = comps[i] && comps[i].samples;
        if (!cs || !cs.BYTES_PER_ELEMENT) {
          throw new Error('logdensityof: N-ary kchain history component '
            + i + ' did not materialise to a scalar ensemble');
        }
        extraRefArrays['s' + i] = cs;
      }
    }
    return matScore(node, ctx, { observed, extraRefArrays })
      .then((reply: any) => applyReduce(reply, node));
  });
}

// =====================================================================
// matBroadcastLogdensity — broadcast(logdensityof, M, pts)
// =====================================================================

function matBroadcastLogdensity(d: DerivationBroadcastLogdensity, ctx: any) {
  // broadcast(logdensityof, M, pts) = [logdensityof(M, p) ∀ p ∈ pts].
  //
  // FAST PATH (engine-concepts §11/§12; flatppl-js is the EAGER
  // engine): when M is a self-contained tractable measure — no
  // per-atom prior refs, not a kchain MC-marginal — its log-density
  // is a closed-form function of the point, so evaluate it at ALL
  // points in ONE batched, sampling-free pass: the density spine's
  // N-axis indexes the evaluation points (`pointsBatched`). This
  // reuses walkSelect / walkWeighted / walkNormalize / walkLeaf
  // verbatim and is bit-identical to the per-point reference (same
  // logpdf catalogue + params). Otherwise (kchain marginal, or M with
  // per-atom prior refs where the points axis would collide with the
  // prior-atom axis) fall back to the per-point reference route
  // (correctness over speed). Result is a value array (length =
  // #points), same shape as a static `array` binding.
  const ptsVal = orchestrator.resolveIRToValue(
    d.pointsIR, ctx.bindings, ctx.fixedValues);
  const pts = Array.isArray(ptsVal) ? ptsVal
    : (ptsVal && ptsVal.data ? Array.from(ptsVal.data) : null);
  if (!pts || pts.length === 0) {
    return Promise.reject(new Error(
      'broadcast(logdensityof, M, pts): points must resolve to a '
      + 'non-empty array (got ' + JSON.stringify(ptsVal) + ')'));
  }

  // Eligibility for the batched closed-form pass: expand M and
  // classify its self-refs exactly as matLogdensityof does.
  const measureDeriv = ctx.derivations[d.measureName];
  const isChain = !!(measureDeriv
    && measureDeriv.kind === 'jointchain' && measureDeriv.marginalize);
  const measureIR0 = orchestrator.expandMeasure(d.measureName,
    { derivations: ctx.derivations, bindings: ctx.bindings });
  // The batched fast path sends its own worker message (not via matScore),
  // so it must resolve normalize-mass specs (audit M3) here as well.
  return resolveNormalizeMasses(measureIR0, ctx)
    .then((measureIR) => {
  let batchable = !!measureIR && !isChain;
  const fixedRefs: any[] = [];
  if (batchable) {
    for (const n of orchestrator.collectSelfRefs(measureIR)) {
      if (isFunctionLikeBinding(ctx.bindings && ctx.bindings.get(n))) continue;
      if (!(ctx.bindings && ctx.bindings.has(n))
          && !(ctx.fixedValues && ctx.fixedValues.has(n))) continue;
      if (ctx.fixedValues && ctx.fixedValues.has(n)) { fixedRefs.push(n); continue; }
      // A per-atom prior ref: the points axis would collide with the
      // prior-atom axis ⇒ not closed-form-batchable here.
      batchable = false;
      break;
    }
  }

  if (batchable) {
    let setEnvP = Promise.resolve();
    if (fixedRefs.length > 0) {
      const fixedEnv: Record<string, any> = {};
      for (const n of fixedRefs) fixedEnv[n] = ctx.fixedValues.get(n);
      setEnvP = ctx.sendWorker({ type: 'setEnv', env: fixedEnv, merge: true });
    }
    return setEnvP.then(() => ctx.sendWorker({
      type: 'logDensityN',
      ir: measureIR,
      count: pts.length,
      refArrays: {},
      observed: pts,
      pointsBatched: true,
    })).then((reply) => {
      const s = reply.samples;
      return scalarMeasureN(
        s instanceof Float64Array ? s : Float64Array.from(s),
        { logWeights: null, logTotalmass: 0, n_eff: pts.length });
    });
  }

  // Fallback: per-point reference route (kchain marginal / per-atom-
  // ref measures). Correct, non-batched, sequential.
  const out = new Float64Array(pts.length);
  let chain = Promise.resolve();
  for (let k = 0; k < pts.length; k++) {
    const kk = k;
    chain = chain
      .then(() => matLogdensityof({
        kind: 'logdensityof',
        measureName: d.measureName,
        obsIR: { kind: 'lit', value: pts[kk] },
      }, ctx))
      .then((m) => { out[kk] = m.samples[0]; });
  }
  return chain.then(() => scalarMeasureN(out, {
    logWeights: null, logTotalmass: 0, n_eff: out.length,
  }));
  });
}

module.exports = {
  matBayesupdate,
  matLogdensityof,
  matLikelihoodDensity,
  matJointLikelihoodDensity,
  matBroadcastLogdensity,
  matTotalmass,
};
