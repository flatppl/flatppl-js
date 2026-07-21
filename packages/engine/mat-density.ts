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
const { totalMassExpr } = require('./normalize-mass.ts');

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

// Derive runNested's {useRegion, region} from the viewer's single-field
// regionMetric contract ('off' | 'whitened' | 'identity' | 'cluster' — see
// types.d.ts Ctx.inferenceOpts and mlfriends.ts's header comment for the
// per-metric tradeoffs). 'off' (region-free) is the default. Back-compat: if
// o.regionMetric is undefined (a direct runNested caller / older saved state
// that predates this field), fall back to the raw o.useRegion/o.region pair
// that mat-density.ts passed straight through before this field existed.
function deriveRegionOpts(o: any): { useRegion: boolean; region: any } {
  if (o && o.regionMetric != null) {
    const metric = o.regionMetric;
    return {
      useRegion: metric !== 'off',
      region: metric === 'off' ? undefined : { metric },
    };
  }
  return { useRegion: !!(o && o.useRegion === true), region: o ? o.region : undefined };
}

// =====================================================================
// Bayesupdate — reweight prior atoms by per-atom log-likelihood
// =====================================================================

function matBayesupdate(d: DerivationBayesupdate, ctx: any) {
  // backend:'mh' (or 'nuts') — run Metropolis-Hastings on the latents, then
  // return an equal-weight empirical measure. IS path (default) is below.
  const backend = (ctx.inferenceOpts && ctx.inferenceOpts.backend) || 'is';
  if (backend === 'mh' || backend === 'ram' || backend === 'slice' || backend === 'nuts' || backend === 'emcee' || backend === 'amis' || backend === 'smc' || backend === 'elliptical-slice-sampler' || backend === 'nested' || backend === 'demcz') {
    const MV              = require('./model-view.ts');
    const driver          = require('./mcmc-driver.ts');
    const { mhKernel }         = require('./mh-kernel.ts');
    const { makeEmceeKernel }  = require('./emcee-kernel.ts');
    const { makeEllipticalSliceKernel } = require('./elliptical-slice-kernel.ts');
    const { makeRamKernel } = require('./ram-kernel.ts');
    const { makeSliceKernel } = require('./slice-kernel.ts');
    const { makeDemczKernel } = require('./demcz-kernel.ts');
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
        // Log-evidence: AMIS's logW = logπ̃(x) − log q_mix(x) are RAW importance
        // weights, so logmeanexp(logW) is an unbiased log Ẑ (same estimator the
        // IS path puts on logTotalmass). Reported for the evidence readout.
        const amisLogZ = empirical.logSumExp(lw) - Math.log(nDraws);
        post = { drawsByName, diagnostics: { method: 'amis', ess, essFrac: ess / nDraws, K: res.K, nSamples: nDraws, logZ: amisLogZ, perParam } };
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
      } else if (backend === 'nested') {
        const { buildPriorTransform } = require('./prior-transform.ts');
        const { runNested } = require('./nested-sample.ts');
        const pt = buildPriorTransform(ctx, d);
        // seeded LCG so a run is reproducible (matches AMIS/SMC determinism expectations)
        let sState = ((o.seed ?? 1) >>> 0) || 1;
        const prng = () => { sState = (1103515245 * sState + 12345) >>> 0; return sState / 4294967296; };
        // Default sliceSweeps 2 (nested-sample.ts's own internal default is 5):
        // region-free slice is the default replacement step now (region is
        // opt-in), and 2 sweeps roughly halves the per-replacement eval cost
        // while the closed-form Gaussian evidence test still passes.
        // useRegion/region: opt-in region-based constrained draws (mlfriends.ts),
        // derived from the single-field regionMetric contract the viewer's gear
        // panel exposes ('off' = region-free default) — see deriveRegionOpts.
        const { useRegion, region } = deriveRegionOpts(o);
        const res = runNested(pt.transform, pt.dim, mv.likOf,
          { nLive: o.nLive || 400, dlogz: o.dlogz ?? 0.5, sliceSweeps: o.sliceSweeps != null ? o.sliceSweeps : 2, prng, onProgress,
            useRegion, region, rebuild: o.rebuild, regionTries: o.regionTries });
        nDraws = res.samples.length;
        // θ-space scorer records → per-coordinate draws (nested samples are already
        // CONSTRAINED, so read fields directly — no mv.constrainAll).
        const drawsByName: Record<string, Float64Array> = {};
        for (let li = 0; li < mv.latentNames.length; li++) {
          const nm = mv.latentNames[li], shp = mv.latentShapes[li];
          if (shp.kind === 'scalar') drawsByName[nm] = new Float64Array(nDraws);
          else for (let j = 0; j < shp.dims[0]; j++) drawsByName[`${nm}[${j}]`] = new Float64Array(nDraws);
        }
        for (let a = 0; a < nDraws; a++) {
          const rec = res.samples[a];
          for (let li = 0; li < mv.latentNames.length; li++) {
            const nm = mv.latentNames[li], shp = mv.latentShapes[li];
            if (shp.kind === 'scalar') { drawsByName[nm][a] = +rec[nm]; }
            else { const v: any = rec[nm]; for (let j = 0; j < shp.dims[0]; j++) drawsByName[`${nm}[${j}]`][a] = +v[j]; }
          }
        }
        // Weighted systematic resample over normalized exp(logWeights) (AMIS pattern).
        const lw = res.logWeights; let mx = -Infinity;
        for (let i = 0; i < nDraws; i++) if (lw[i] > mx) mx = lw[i];
        let sw = 0; for (let i = 0; i < nDraws; i++) sw += Math.exp(lw[i] - mx);
        const wbar = new Float64Array(nDraws); let sumSq = 0;
        for (let i = 0; i < nDraws; i++) { wbar[i] = Math.exp(lw[i] - mx) / sw; sumSq += wbar[i] * wbar[i]; }
        const ess = sumSq > 0 ? 1 / sumSq : 0;
        const total = total0 > 0 ? total0 : nDraws;
        idx = new Int32Array(total);
        { let c = wbar[0], j = 0; for (let i = 0; i < total; i++) { const uu = (i + 0.5) / total; while (uu > c && j < nDraws - 1) { j++; c += wbar[j]; } idx[i] = j; } }
        const perParam: Record<string, any> = {};
        for (let d2 = 0; d2 < mv.dim; d2++) perParam[mv.names[d2]] = { rHat: NaN, essBulk: ess };
        post = { drawsByName, diagnostics: { method: 'nested', logZ: res.logZ, logZerr: res.logZerr, nLive: res.nLive, efficiency: res.efficiency, ess, nSamples: nDraws, perParam } };
      } else {
        const isEss = backend === 'elliptical-slice-sampler';
        const kernel = backend === 'emcee' ? makeEmceeKernel(o.a)
          : backend === 'demcz' ? makeDemczKernel(o)
          : backend === 'ram' ? makeRamKernel()
          : backend === 'slice' ? makeSliceKernel()
          : isEss ? makeEllipticalSliceKernel() : mhKernel;
        const nWalkers = o.walkers ?? o.chains ?? (backend === 'emcee' ? Math.max(4, 2 * mv.dim + 2) : backend === 'demcz' ? Math.max(8, 2 * mv.dim) : 4);
        post = driver.runMcmc(mv, kernel, {
          nWalkers, warmup: o.warmup ?? 1000, draws: o.draws ?? 1000, seed: (o.seed ?? 0), a: o.a, essMaxShrink: o.essMaxShrink, onProgress,
          // Freeze-then-parallel (mh across a worker pool): `initAdapt` (a frozen
          // {L,scale} from a single warmup worker) + `initPositions` (that chain's
          // warmed start) make this a fixed-proposal sampling run — no re-adaptation
          // — so full-length independent chains run on separate workers. Undefined
          // on a normal single-phase run, leaving behaviour unchanged.
          initPositions: o.initPositions, initAdapt: o.initAdapt,
        });
        // Warmup phase of a freeze-then-parallel run: surface the adapted proposal
        // ({L,scale}, from this ONE worker's full sequential warmup — the exact old
        // proposal, not a pooled one) and the warmed end positions, so the sampling
        // phase can distribute the chains under that frozen proposal. The measure
        // itself is discarded in this phase.
        if (o.mcmcPhase === 'warmup') {
          const a = post.adaptState || {};
          post.diagnostics.warmup = { L: a.L, scale: a.scale, endPositions: post.endPositions };
        }
        // post.diagnostics is { acceptRate, perParam:{name:{rHat,essBulk}} } — attach as-is.
        nDraws = post.drawsByName[mv.names[0]].length;
        // Record the TRUE draw count so the viewer shows it rather than the
        // sampleCount the output is resampled to for plotting.
        post.diagnostics.nSamples = nDraws;
        // Sampling phase distributes full-length chains across the pool; ship the
        // raw per-chain draws so split-R̂ / bulk-ESS are computed ONCE over ALL
        // chains globally (a worker holding a subset can't pool them itself). The
        // ensemble backend (emcee) pools independent ensembles and doesn't need it.
        if (backend !== 'emcee' && backend !== 'demcz') post.diagnostics.rawChains = post.walkers;
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
        if (backend === 'slice') post.diagnostics.method = 'slice';
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
          // Carry the sampler diagnostics onto BOTH the record measure and each
          // field measure: the viewer renders a single field on its own (e.g.
          // gamma-reparam's derived scalar), and without this the field has no
          // diagnostics.nSamples, so the count label falls back to the resampled
          // atom count (~10^5) instead of the true draw count.
          for (const fn in fields) {
            if (fields[fn] && typeof fields[fn] === 'object') fields[fn].diagnostics = post.diagnostics;
          }
          const recM = empirical.recordMeasure(fields, null);
          recM.diagnostics = post.diagnostics;
          if (post.diagnostics && post.diagnostics.method === 'nested') recM.logTotalmass = post.diagnostics.logZ;
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
        if (post.diagnostics && post.diagnostics.method === 'nested') m.logTotalmass = post.diagnostics.logZ;
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
    // IS log-evidence: with prior atoms and logWeights = priorLW + logp, the
    // total mass logsumexp(newLW) IS the log marginal likelihood estimate — the
    // viewer surfaces it via the "total mass" badge (a weighted measure carries
    // its evidence on logTotalmass). No diagnostics object here: the weighted IS
    // measure keeps its PSIS-k̂ / ESS readout, unlike the equal-weight
    // MCMC/AMIS/SMC outputs that report a `diagnostics.method` instead.
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
  // Match a kernel input name to a θ record field BY PUBLIC NAME. When the
  // kernel comes from a `load_module` instance the linker prefixes the input
  // node's name with the instance alias (`model$theta1`, spec §11 internal
  // flat-graph naming), but the module's boundary-input INTERFACE — the names
  // θ is fed by (its own `default_pars`, its prior variate labels, and
  // load-time substitution keywords, spec §04 "Module composition" /
  // sec:functionof) — is the bare module-local name (`theta1`). So prefer an
  // exact field, then fall back to the bare suffix after the last `$` when
  // that public name is unambiguous among the inputs. Returns the θ field name
  // to read, or null if none matches. (#360)
  const publicOf = (k: string): string => {
    const i = k.lastIndexOf('$');
    return i < 0 ? k : k.slice(i + 1);
  };
  const matchField = (k: string): string | null => {
    if (!isRecord) return null;
    if (k in (theta as any)) return k;
    const pub = publicOf(k);
    if (pub === k || !(pub in (theta as any))) return null;
    // Only strip when no OTHER input shares this public name (else ambiguous).
    let shared = 0;
    for (const kk of kwargs) if (publicOf(kk) === pub) shared++;
    return shared === 1 ? pub : null;
  };
  if (kwargs.length === 1 && matchField(kwargs[0]) === null) {
    // Single kernel input fed the whole θ value (scalar/vector point).
    boundaries[kwargs[0]] = theta;
  } else {
    // Record θ: project each kernel input from its field (the multi-param
    // case, and any single-param term whose θ is a record — e.g. a
    // joint_likelihood sub-term declaring fewer params than θ carries).
    for (const k of kwargs) {
      const field = matchField(k);
      if (field === null) {
        const pub = publicOf(k);
        const hint = pub !== k ? '" (public "' + pub + '")' : '"';
        return Promise.reject(new Error(
          'logdensityof(L, θ): θ must be a record with a "' + k
          + hint + ' field (kernel inputs: ' + kwargs.join(', ') + ')'));
      }
      boundaries[k] = (theta as any)[field];
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
      // parseTruncationBox recognises `interval(...)` (1-D) and
      // `cartprod(interval, ...)` (N-D, ≤ TRUNCATE_DIM_CAP axes) — it throws
      // above the cap, and returns null for any other set shape (a dynamic /
      // named set defers to the by-name materialiser, unchanged).
      const axes = parseTruncationBox(inner.args[1]);
      if (axes) {
        const weightFn = weightedBaseWeightFn(inner.args[0]);
        let logZ: number;
        if (axes.length === 1 && weightFn == null) {
          // 1-D scalar reference measure (Normal→CDF closed form / its
          // narrow composite-midpoint quadrature): unchanged fast path.
          const base = asScalarFactor(inner.args[0], theta);
          if (!base) {
            throw new Error('density: normalize(truncate(M, S)) — base measure M '
              + 'is not a recognised scalar reference measure nor a '
              + 'weighted(<weight-fn>, Lebesgue) density, cannot resolve its '
              + 'support-restricted normalizer');
          }
          logZ = truncateLogMass(base, [axes[0].lo, axes[0].hi], ctx);
        } else if (weightFn != null) {
          // A generic_dist (§12) lowers to
          // `normalize(truncate(weighted(w, Lebesgue), S))`: its base is an
          // unnormalized density `w(x)` over the (possibly N-D, ≤3-axis) set
          // S, NOT a scalar reference measure. Z = ∫_S w(x) dx via adaptive
          // cubature over the unit box (makeIntegrandND applies the
          // per-axis change-of-variables + Jacobian for unbounded axes).
          const { adaptiveCubature } = require('./quadrature.ts');
          const integ = makeIntegrandND(weightFn.body, weightFn.paramNames, axes, theta, ctx);
          const res = adaptiveCubature(integ, axes.length);
          if (!(res.Z > 0)) {
            throw new Error('density: normalize(truncate(weighted(w, Lebesgue), S)) — '
              + 'normalizer Z = ∫_S w dx is 0 (an everywhere-zero weight over S has no '
              + 'normalized density per spec §06)');
          }
          const relErr = res.err / Math.abs(res.Z);
          if (relErr > 1e-3) {
            throw new Error('density: normalize(truncate(weighted(w, Lebesgue), S)) — '
              + 'adaptive quadrature did not converge (rel-err ' + relErr.toExponential(2)
              + ' after ' + res.evals + ' evals); the integrand may be non-integrable '
              + 'or too singular for the ' + axes.length + '-D quadrature');
          } else if (relErr > 1e-6) {
            // eslint-disable-next-line no-console
            console.warn('normalize(truncate(weighted(w, Lebesgue), S)): quadrature '
              + 'converged to rel-err ' + relErr.toExponential(2) + ' (' + res.evals + ' evals)');
          }
          logZ = Math.log(res.Z);
        } else {
          throw new Error('density: normalize(truncate(M, cartprod)) — a '
            + 'multi-dimensional truncation requires a weighted(w, Lebesgue) base '
            + '(scalar reference measures are 1-D only)');
        }
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

// Resolve a bound expression to a plain number: a `lit` literal, or a spec
// §03 numeric `const` (`pi`/`e`/`inf`/`-inf`) — `interval(0.0, inf)` lowers
// its `inf` bound to `{kind:'const',name:'inf'}`, NOT a `lit`, so a bound
// resolver that only reads `.value` off a `lit` node silently treats every
// semi-/doubly-infinite interval as "not a literal bound" (defers instead
// of recognising the unbounded axis). null for any other expression shape
// (a bound that needs θ or module context defers to the by-name path).
function numericBoundValue(node: any): number | null {
  if (!node) return null;
  if (node.kind === 'lit') return +node.value;
  if (node.kind === 'const') {
    const samplerLib = require('./sampler.ts');
    const v = samplerLib._internal.resolveConst(node.name);
    return typeof v === 'number' ? v : null;
  }
  return null;
}

// Resolve an `interval(lo, hi)` set IR (literal/const bounds) to [lo, hi];
// null for any other set shape (dynamic / named sets defer to the by-name
// materialiser).
function truncateSetBounds(setIR: any): [number, number] | null {
  if (!setIR || setIR.kind !== 'call' || setIR.op !== 'interval'
      || !Array.isArray(setIR.args) || setIR.args.length !== 2) return null;
  const lo = numericBoundValue(setIR.args[0]), hi = numericBoundValue(setIR.args[1]);
  if (lo != null && hi != null) return [lo, hi];
  return null;
}

const TRUNCATE_DIM_CAP = 3;

type TruncAxis = { lo: number; hi: number; kind: 'finite' | 'semi-lo' | 'semi-hi' | 'infinite' };

// One interval(lo,hi) set IR → an axis, classifying finiteness; null if not a
// literal-bound interval. Reuses the literal check truncateSetBounds uses.
function axisFromInterval(setIR: any): TruncAxis | null {
  const b = truncateSetBounds(setIR); // [lo,hi] for interval(lit,lit), else null
  if (!b) return null;
  const [lo, hi] = b;
  const loF = Number.isFinite(lo), hiF = Number.isFinite(hi);
  const kind = loF && hiF ? 'finite' : loF ? 'semi-lo' : hiF ? 'semi-hi' : 'infinite';
  return { lo, hi, kind };
}

// Set IR → per-axis truncation box. Recognizes interval(...) (1-D) and
// cartprod(interval, ...) (N-D). null for any other shape (caller defers).
// Throws above the dimension cap.
function parseTruncationBox(setIR: any): TruncAxis[] | null {
  if (!setIR || setIR.kind !== 'call') return null;
  if (setIR.op === 'interval') {
    const a = axisFromInterval(setIR);
    return a ? [a] : null;
  }
  if (setIR.op === 'cartprod' && Array.isArray(setIR.args)) {
    if (setIR.args.length > TRUNCATE_DIM_CAP) {
      throw new Error('density: normalize(truncate(M, cartprod)) — region '
        + 'dimension ' + setIR.args.length + ' exceeds the quadrature cap of '
        + TRUNCATE_DIM_CAP + '; higher-dimensional truncation is not supported');
    }
    const axes: TruncAxis[] = [];
    for (const f of setIR.args) {
      const a = axisFromInterval(f);
      if (!a) return null; // a non-interval factor → defer whole set
      axes.push(a);
    }
    return axes;
  }
  return null;
}

// log Z = log ∫_lo^hi f_base(x) dx for a scalar reference-measure base resolved
// at θ. Closed-form via forward-cdf.ts's CDF ladder: an O(1) Φ(hi)−Φ(lo)
// evaluation, NOT a quadrature — critical when `base`'s params are
// θ-dependent, since this whole function reruns from scratch on EVERY
// density call for an inline normalize(truncate(M(θ), S)), e.g. inside a
// kernelof/likelihoodof body scored once per MCMC step; composite-midpoint
// quadrature of the base log-density for any kernel forward-cdf.ts doesn't
// carry (matching numericProductLogZ).
function truncateLogMass(base: any, bounds: [number, number], ctx: any): number {
  // A discrete base is never reached here: an inline normalize(truncate(...))
  // this path handles is the §12 generic_dist form (weighted(w, Lebesgue), a
  // continuous base); a discrete truncate carries a massFrom ref and is
  // resolved (and refused) by resolveNormalizeMasses instead (Buffy #73).
  const [lo, hi] = bounds;
  const forwardCdf = require('./forward-cdf.ts');
  if (forwardCdf.hasCdf(base.kernel)) {
    // Φ(hi) − Φ(lo), O(1) regardless of how many evals this runs across (no
    // per-θ caching needed — the closed form IS the fast path). Infinite
    // bounds map to the CDF's limit (0 / 1) rather than evaluating the CDF
    // function at ±Infinity, mirroring forward-cdf.ts's own truncatedQuantile.
    const Flo = Number.isFinite(lo) ? forwardCdf.cdf(base.kernel, lo, base.input) : 0;
    const Fhi = Number.isFinite(hi) ? forwardCdf.cdf(base.kernel, hi, base.input) : 1;
    const Z = Fhi - Flo;
    if (!(Z > 0)) {
      throw new Error('density: normalize(truncate(' + base.kernel + ', S)) — mass over the '
        + 'truncation set is 0 (Z = 0 is undefined per spec §06)');
    }
    return Math.log(Z);
  }
  // The fixed-width midpoint fallback has no change-of-variables, so an
  // unbounded interval gives dx = inf and a silent NaN. Refuse-don't-mislower:
  // this scalar-reference path has no CDF for a kernel forward-cdf.ts doesn't
  // carry (the branch above handles every ladder entry, incl. unbounded
  // supports, via the CDF's 0/1 limits). A weighted(w, Lebesgue) base takes
  // the N-D adaptive-cubature path instead, which maps unbounded axes with a
  // Jacobian.
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    throw new Error('density: normalize(truncate(' + base.kernel + ', S)) — '
      + 'unbounded truncation of this reference measure has no deterministic '
      + 'quadrature (no closed-form CDF is registered for this kernel in '
      + 'forward-cdf.ts, and a weighted(w, Lebesgue) base would take the '
      + 'adaptive-cubature path instead); refusing rather than emitting NaN');
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
// §12 generic_dist lowering — and return its weight FUNCTION as an N-
// parameter `functionof` IR { paramNames, body }; null for any other base.
// The Lebesgue density is ≡ 1, so the base density is the weight `w` itself
// and Z = ∫_S w(x) dx. A constant-weight `weighted` (no function, just a
// log-mass shift) is NOT this shape and returns null (it falls through to
// the scalar-reference path, which already handles plain reference bases).
// N ≥ 1: the 1-D callers below use paramNames[0]; the N-D cartprod path
// (Task 4) binds paramNames[i] to axis i.
function weightedBaseWeightFn(baseIR: any): { paramNames: string[]; body: any } | null {
  if (!baseIR || baseIR.kind !== 'call' || baseIR.op !== 'weighted'
      || !Array.isArray(baseIR.args) || baseIR.args.length !== 2) return null;
  const ref = baseIR.args[1];
  // Reference measure must be Lebesgue (its density ≡ 1, so the weight is
  // the full density). `Lebesgue(reals)` or `Lebesgue(interval(...))`.
  if (!ref || ref.kind !== 'call' || ref.op !== 'Lebesgue') return null;
  const fn = baseIR.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length < 1 || !fn.body) {
    return null;
  }
  return { paramNames: fn.params.slice(), body: fn.body };
}

// One-axis coordinate + Jacobian from a unit coordinate u ∈ (0,1). Mirrors
// the change-of-variables table for the unbounded TruncAxis kinds:
//   finite [a,b]:    x = a + u·(b-a),        J = (b-a)
//   semi-lo [a,∞):    x = a + u/(1-u),        J = 1/(1-u)²
//   semi-hi (-∞,b]:   x = b - (1-u)/u,        J = 1/u²
//   infinite (-∞,∞):  s = 2u-1, x = atanh(s),  J = 2/(1-s²)
function axisMap(axis: TruncAxis, u: number): { x: number; J: number } {
  switch (axis.kind) {
    case 'finite':  { const h = axis.hi - axis.lo; return { x: axis.lo + u * h, J: h }; }
    case 'semi-lo': { const t = 1 - u;             return { x: axis.lo + u / t, J: 1 / (t * t) }; }
    case 'semi-hi': { return { x: axis.hi - (1 - u) / u, J: 1 / (u * u) }; }
    case 'infinite': { const s = 2 * u - 1; return { x: Math.atanh(s), J: 2 / (1 - s * s) }; }
  }
}

// Non-log integrand w(x(u))·∏Jᵢ on the unit box (u ∈ (0,1)^dims), suitable
// for `adaptiveCubature` (quadrature.ts). paramNames[i] binds to axes[i]'s
// real coordinate xᵢ(uᵢ) — the weight body's free variate refs, exactly as
// `weightedBaseLogMass` below binds the 1-D case. theta's own params are
// bound once (θ is fixed across quadrature nodes); ctx.moduleRegistry
// threads through for any standard-module call inside the weight (e.g.
// `poly.chebyshev`), matching weightedBaseLogMass's env construction.
// A non-finite or negative weight at a node is treated as 0 (per spec §06
// a normalizer only integrates the non-negative part of a density; a
// weight function that goes negative or blows up off the intended support
// contributes nothing rather than corrupting the quadrature).
function makeIntegrandND(
  weightBody: any, paramNames: string[], axes: TruncAxis[], theta: any, ctx: any,
): (u: number[]) => number {
  const samplerLib = require('./sampler.ts');
  const base: Record<string, any> = {};
  if (ctx && ctx.moduleRegistry) base.__moduleRegistry = ctx.moduleRegistry;
  if (theta && typeof theta === 'object') for (const k in theta) base[k] = theta[k];
  return (u: number[]): number => {
    const env: Record<string, any> = Object.assign({}, base);
    let J = 1;
    for (let i = 0; i < axes.length; i++) {
      const m = axisMap(axes[i], u[i]);
      env[paramNames[i]] = m.x;
      J *= m.J;
    }
    const w = +samplerLib.evaluateExpr(weightBody, env);
    if (!Number.isFinite(w) || w < 0) return 0;
    return w * J;
  };
}

// log Z = log ∫_lo^hi w(x) dx for a `weighted(w, Lebesgue)` base, where `w`
// is the (non-log) density. Composite-midpoint quadrature over S at the
// fixed point θ, mirroring numericProductLogZ's QUAD_POINTS midpoint rule —
// but evaluating the weight expression `w(x; θ)` per node via the sampler's
// per-expr evaluator (the variate param bound to the node, θ's params bound
// to their values). The weight is the density, NOT a log-density, so it is
// summed directly (not exponentiated): Z = Σ w(x_j) · dx.
function weightedBaseLogMass(
  weightFn: { paramNames: string[]; body: any }, bounds: [number, number], theta: any,
  ctx: any,
): number {
  const samplerLib = require('./sampler.ts');
  const [lo, hi] = bounds;
  const dx = (hi - lo) / QUAD_POINTS;
  // env binds θ's free params (e.g. `alpha`) plus the variate param to the
  // current quadrature node; evaluateExpr resolves refs by name (ns-agnostic).
  // The weight `w(x)` may call a standard-module function (e.g. the converter's
  // `poly.chebyshev(k, t)` for a generic_dist chebychev density). evaluateExpr
  // resolves `(call target={ns:<alias>,…})` through `env.__moduleRegistry`, so
  // thread it from the ctx — this is a LOCAL quadrature env, not the worker
  // session, so pushModuleRegistry (worker setEnv) doesn't cover it.
  const env: Record<string, any> = {};
  if (ctx && ctx.moduleRegistry) env.__moduleRegistry = ctx.moduleRegistry;
  if (theta && typeof theta === 'object') {
    for (const k in theta) env[k] = theta[k];
  }
  let Z = 0;
  let nonFinite = 0;
  for (let j = 0; j < QUAD_POINTS; j++) {
    env[weightFn.paramNames[0]] = lo + (j + 0.5) * dx;
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

// #322: for a `weighted(w, Lebesgue(interval(a,b)))` base whose weight `w` is
// θ-INDEPENDENT — it references only the variate, module-level FIXED constants,
// and function bindings (e.g. `poly.chebyshev`) — the normalizer Z = ∫_a^b w dx
// is a single constant computable by the deterministic quadrature above.
// Return the {name: value} env of the weight's fixed constant refs, or null if
// the weight references any LATENT / fed param (Z is then per-θ and must be
// resolved by the Monte-Carlo materialise path instead). Mirrors the
// fixed-vs-latent ref split matLogdensityof's batchable check uses.
function weightedFixedWeightEnv(
  weightFn: { paramNames: string[]; body: any }, ctx: any,
): Record<string, any> | null {
  const env: Record<string, any> = {};
  for (const n of orchestrator.collectSelfRefs(weightFn.body)) {
    const b = ctx && ctx.bindings && ctx.bindings.get(n);
    if (isFunctionLikeBinding(b)) continue;                      // a called fn / module member
    if (ctx && ctx.fixedValues && ctx.fixedValues.has(n)) {      // a resolved constant
      env[n] = ctx.fixedValues.get(n); continue;
    }
    return null;                                                 // latent / fed / unknown → per-θ, use MC
  }
  return env;
}

// Recognise `weighted(w, <ref-over-interval(a,b)>)` (no truncate) and, when `w`
// is θ-independent, resolve its normalizer deterministically by quadrature.
// The base reference is a bounded `Lebesgue(interval)` — which the lowering
// rewrites to `Uniform(interval)` (density 1/(b−a), the density walker applies
// it in the numerator), so the correct −log Z for the normalize to cancel is
// `log ∫_a^b w dx − log(b−a)`. `Lebesgue(reals)` / an unbounded or non-literal
// support / a θ-dependent weight all return null (caller → Monte-Carlo).
function weightedLebesgueQuadLogZ(node: any, ctx: any): number | null {
  const inner = node.args && node.args[0];
  if (!inner || inner.kind !== 'call' || inner.op !== 'weighted'
      || !Array.isArray(inner.args) || inner.args.length !== 2) return null;
  const fn = inner.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length !== 1 || !fn.body) return null;
  // The base reference is "Lebesgue over [a,b]" (density ≡ 1), which the lowering
  // expresses as one of: `Lebesgue(interval)`, `Uniform(interval)` (only when
  // b−a = 1, where the density is already 1), or — for a non-unit interval —
  // `logweighted(log(b−a), Uniform(interval))` (the #307 Lebesgue-restoring
  // shift that scales Uniform's 1/(b−a) back to 1). Unwrap that shift so all
  // three forms are recognised; the effective base density is 1 in every case,
  // so Z = ∫_a^b w dx with NO extra 1/(b−a) factor.
  let ref = inner.args[1];
  if (ref && ref.kind === 'call' && ref.op === 'logweighted'
      && Array.isArray(ref.args) && ref.args.length === 2) {
    ref = ref.args[1];
  }
  if (!ref || ref.kind !== 'call' || (ref.op !== 'Lebesgue' && ref.op !== 'Uniform')) return null;
  const supp = (ref.kwargs && ref.kwargs.support) || (Array.isArray(ref.args) ? ref.args[0] : null);
  const bounds = truncateSetBounds(supp);
  if (!bounds) return null;    // unbounded (Lebesgue(reals)) / non-literal interval
  const weightFn = { paramNames: [fn.params[0]], body: fn.body };
  const env = weightedFixedWeightEnv(weightFn, ctx);
  if (env == null) return null;   // θ-dependent weight → MC materialise
  return weightedBaseLogMass(weightFn, bounds, env, ctx);   // log ∫_a^b w dx, base density ≡ 1
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

function matPosteriorDensity(d: any, ctx: any) {
  // logdensityof(bayesupdate(L, prior), θ) (#309). Per spec §06
  // `bayesupdate(L, prior) ≡ logweighted(fn(logdensityof(L, _)), prior)`, so the
  // posterior's log-density at θ is `logdensityof(L, θ) + logdensityof(prior, θ)`.
  // Score the two independently — the likelihood via the standalone likelihood-
  // density path (feeds the kernel's params EXPLICITLY from θ), the prior via the
  // generic logdensityof path (scores the prior measure's density at θ) — and sum
  // the per-atom scalars. Both produce a broadcast-constant per-atom scalar; the
  // sum is the posterior density. This is exactly what the determiniser-then-eval
  // route (det-js) computes, applied natively.
  const N = ctx.sampleCount;
  return Promise.all([
    matLikelihoodDensity({
      kind: 'likelihood_density',
      bodyName: d.bodyName, bodyIR: d.bodyIR, obsIR: d.obsIR,
      paramKwargs: d.paramKwargs, params: d.params, pointIR: d.pointIR,
    }, ctx),
    matLogdensityof({
      kind: 'logdensityof', measureName: d.priorName, obsIR: d.pointIR,
    } as any, ctx),
  ]).then((measures: any[]) => {
    const out = new Float64Array(N);
    for (const m of measures) {
      const s = m.samples;
      // Each sub is a broadcast-constant per-atom scalar; a sub materialised at
      // N=1 (or any shorter length) broadcasts its single value across N atoms.
      for (let i = 0; i < N; i++) out[i] += s[s.length === 1 ? 0 : i];
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
  // First pass: per-θ normalizer when the inner is a recognised
  // superpose-of-weighted probability measures (Z depends on the latent
  // weights); rewrite to logweighted(−log Z(θ), inner) without materialising.
  // Only un-recognised nodes reach the materialise/constant-bake fallback.
  const needMaterialise: any[] = [];
  const samplerLib = require('./sampler.ts');
  for (const node of nodes) {
    // normalize(truncate(<discrete base>, S)): the materialised logTotalmass
    // comes from the worker's continuous CDF-difference (F(hi)−F(lo)), which
    // is wrong for a discrete base (drops the lower endpoint; the true mass is
    // Σ pmf over S). §06 defines no discrete-truncate normalizer — fail loud
    // rather than materialise a silently-wrong Z (Buffy #73). Sampling of a
    // truncated discrete leaf is untouched (this guards only the normalizer).
    const innerT = node.args[0];
    const truncBase = innerT && innerT.op === 'truncate' && innerT.args && innerT.args[0];
    const baseOp = truncBase && typeof truncBase.op === 'string' ? truncBase.op : null;
    if (baseOp && samplerLib.isKnownDistribution(baseOp)
        && samplerLib.lookupDistribution({ kind: 'call', op: baseOp }).discrete) {
      throw new Error('density: normalize(truncate(' + baseOp + ', S)) — a '
        + 'discrete base measure’s support-restricted normalizer (Σ pmf over S) '
        + 'is not implemented; refusing to materialise a silently-wrong Z (spec §06)');
    }
    // Per-θ normalizer when the inner is a recognised superpose-of-weighted
    // probability measures (Z depends on the latent weights); else materialise.
    const massExpr = totalMassExpr(node.args[0]);
    if (massExpr != null) {
      node.op = 'logweighted';
      node.args = [{ kind: 'call', op: 'neg', args: [{ kind: 'call', op: 'log', args: [massExpr] }] }, node.args[0]];
      delete node.massFrom;
      continue;   // do NOT add this node to the materialise list
    }
    // #322: normalize(weighted(w, Lebesgue(interval))) with a θ-independent
    // weight → deterministic quadrature Z (∫ w over the interval), instead of
    // the seeded Monte-Carlo materialise below.
    const quadLogZ = weightedLebesgueQuadLogZ(node, ctx);
    if (quadLogZ != null) {
      node.op = 'logweighted';
      node.args = [{ kind: 'lit', value: -quadLogZ }, node.args[0]];
      delete node.massFrom;
      continue;
    }
    needMaterialise.push(node);
  }
  if (needMaterialise.length === 0) return Promise.resolve(measureIR);
  const refNames = Array.from(new Set(needMaterialise.map((n: any) => n.massFrom.ref as string)));
  return Promise.all(refNames.map((r) => Promise.resolve(ctx.getMeasure(r))))
    .then((measures: any[]) => {
      const byName: Record<string, any> = {};
      refNames.forEach((nm: string, i: number) => { byName[nm] = measures[i]; });
      for (const node of needMaterialise) {
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
  matPosteriorDensity,
  matBroadcastLogdensity,
  matTotalmass,
  parseTruncationBox,
  makeIntegrandND,
  weightedBaseWeightFn,
  resolveTruncateNormalizers,
  deriveRegionOpts,
};
