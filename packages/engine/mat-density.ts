'use strict';

// Density-related materialiser handlers: matScore (the ONE CLM density
// consumer), matBayesupdate, matLogdensityof, matBroadcastLogdensity,
// matTotalmass. Plus the runtime-weight selector resolution
// (collectRuntimeWeightNodes / resolveRuntimeWeights) that matLogdensityof
// and matBroadcastLogdensity both run before density evaluation
// (engine-concepts §11 MC-weight selector).
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

const {
  nameSeed,
  isFunctionLikeBinding,
  collectRefArrays,
  prepareDensityRefs,
  pushFixedEnv,
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
  // Runtime-weight selectors (engine-concepts §11): resolve the body's
  // non-closed-form mixture/select weights to literal logweights before
  // scoring. matLogdensityof/matBroadcastLogdensity opt in; bayesupdate did
  // not, so it stays off there (preserving the Phase-2 byte-identical path).
  // node.body is freshly built per lowerMeasure call (no cache), so the
  // in-place resolution is safe (mutation hazard — plan critique).
  const bodyP = opts.resolveWeights
    ? resolveRuntimeWeights(node.body, ctx)
    : Promise.resolve(node.body);
  return bodyP.then((body: any) =>
    clm.feedInputs(node, ctx).then((fed: any) => {
      // extraRefArrays carries columns the declared inputs can't source via a
      // plain getMeasure — the N-ary kchain retained-history variates s_i,
      // materialised by matJointchain. They override the fed entries.
      const refArrays = opts.extraRefArrays
        ? Object.assign({}, fed.refArrays, opts.extraRefArrays)
        : fed.refArrays;
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
// Runtime-weight selector resolution (engine-concepts §11)
// =====================================================================
//
// Walk an expanded measure IR collecting every `select` node that
// still carries an unresolved runtime-weight spec (`weightsFrom`).
// A non-closed-form selector condition yields a structurally-exact
// mixture whose per-branch weights must be estimated ONCE from the
// materialised selector ensemble. expandMeasureIR is pure and cannot
// reduce an ensemble, so it threads the spec through; the materialiser
// resolves it here. The estimate is a single scalar per branch (the
// selector is marginalised in the density), so it fits the existing
// IR `logweights` slot exactly, just supplied at materialisation
// instead of by classify.

function collectRuntimeWeightNodes(node: any, out: any, seen: any) {
  if (!node || typeof node !== 'object') return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) collectRuntimeWeightNodes(x, out, seen);
    return;
  }
  if (node.kind === 'call' && node.op === 'select'
      && node.weightsFrom && node.logweights == null) {
    out.push(node);
  }
  for (const k in node) {
    const v = node[k];
    if (v && typeof v === 'object') collectRuntimeWeightNodes(v, out, seen);
  }
}

function resolveRuntimeWeights(measureIR: any, ctx: any) {
  const nodes: any[] = [];
  collectRuntimeWeightNodes(measureIR, nodes, new Set());
  if (nodes.length === 0) return Promise.resolve(measureIR);
  const refNames = Array.from(new Set(nodes.map((n) => n.weightsFrom.ref)));
  return Promise.all(refNames.map(ctx.getMeasure)).then((measures) => {
    const byName: Record<string, any> = {};
    refNames.forEach((nm: string, i: number) => { byName[nm] = measures[i]; });
    for (const node of nodes) {
      const spec = node.weightsFrom;
      const M = byName[spec.ref];
      if (!M || !M.samples || !M.samples.length) {
        throw new Error('select runtime weights: selector "' + spec.ref
          + '" did not materialise to a sampled ensemble');
      }
      const sel = M.samples;
      const Nn = sel.length;
      const K = spec.K | 0;
      const base = spec.base | 0;
      const cnt = new Float64Array(K);
      if (K === 2 && base === 0) {
        let t = 0;
        for (let i = 0; i < Nn; i++) if (sel[i]) t++;
        cnt[0] = t; cnt[1] = Nn - t;
      } else {
        for (let i = 0; i < Nn; i++) {
          let idx = (sel[i] | 0) - base;
          if (idx < 0) idx = 0; else if (idx >= K) idx = K - 1;
          cnt[idx]++;
        }
      }
      const lw = new Array(K);
      for (let k = 0; k < K; k++) {
        const p = cnt[k] / Nn;
        lw[k] = { kind: 'lit', value: p > 0 ? Math.log(p) : -Infinity };
      }
      node.logweights = lw;
      delete node.weightsFrom;
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
  // For N-ary kchain (engine-concepts §6 chain-associativity),
  // matJointchain materialises the retained (n−1)-joint history ONCE
  // and we bind its variate columns as the per-atom refArrays the
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
  // through ONE mechanism (the dependent-threaded retain joint), not two. (Was
  // a direct matJointchain call; that path is now only reached as the
  // lowerMeasure-returns-null sample fallback.)
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
    return matScore(node, ctx, { observed, resolveWeights: true, extraRefArrays })
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
  return resolveRuntimeWeights(measureIR0, ctx).then((measureIR) => {
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

// =====================================================================
// MC-marginalised log-likelihood (generative model, intractable density)
// =====================================================================
//
// For a generative model whose observation is a BIJECTIVE pushforward of
// one RETAINED internal draw (the innovation) but MARGINALISES another
// latent draw, the density is intractable in closed form yet estimable
// efficiently (spec §06 case-3 opt-in; engine-concepts §6). This lifts
// the §21 generative-density refusal for that tractable sub-case.
//
// Per-event law  z = f(retained, marginalised; θ), f bijective in
// `retained`, the marginalised latent integrated by Monte Carlo:
//   log p(z=d | θ) ≈ logsumexp_m[ logp_ret(u*_m) − ladj_fwd(u*_m) ] − log M
//   u*_m = f⁻¹(d; x_m, θ),   x_m ~ marginalised,   m = 1..M.
// The retained-draw inversion f⁻¹ and the forward LADJ are EXACT
// (bijection-registry.invertExpr); only the marginalisation over x is MC,
// so M is small (ctx.MARGINALIZATION_COUNT ≈ 100). iid data ⇒
//   log L(θ) = Σ_d log p(z=d | θ).
//
// Because invertExpr returns ordinary value IR, the whole D×M grid of
// (data point, MC draw) inversions evaluates in TWO batched worker passes
// (inverse, LADJ) — the §20.10 vertical collapse, count = D·M.
//
// v1 scope: `retained` is a Uniform innovation on `retainedInterval`
// [lo,hi] (its logp is the interval indicator −log(hi−lo)); a general
// leaf-density retained latent is the extension. `recipeIR` must carry θ
// baked as lits (or resolvable refs supplied to the worker session env);
// the retained + marginalised leaves are refs named `retainedRef.name` /
// `marginalRef.name`. Returns null when `recipeIR` is not bijective in
// `retainedRef` — the caller then refuses (or, later, falls back to KDE).

function mcMarginalLogDensity(opts: any): Promise<number | null> {
  const bijReg = require('./bijection-registry.ts');
  const { recipeIR, retainedRef, retainedInterval, marginalRef, marginalDistIR,
          data, M, ctx, seedTag, frozenEnv } = opts;
  // Output-value placeholder the inverse is expressed in terms of.
  const OUT = { kind: 'ref', ns: '%mc', name: '__mc_z__' };
  const inv = bijReg.invertExpr({ outputExpr: recipeIR, freeRef: retainedRef, outputValue: OUT });
  if (!inv) return Promise.resolve(null);   // not bijective in the retained draw
  const D = data.length;
  if (D === 0 || M <= 0) return Promise.resolve(0);
  const count = D * M;
  const tag = seedTag || '%mcmarg';
  // Frozen leaves (θ point + fixed globals like `sigma`) flow through the
  // worker session env — the recipe / marginal-dist refs (`pars`, `sigma`)
  // resolve there. merge:true; a sweep overwrites `pars` per point.
  const setup = (frozenEnv && Object.keys(frozenEnv).length > 0)
    ? ctx.sendWorker({ type: 'setEnv', env: frozenEnv, merge: true })
    : Promise.resolve(null);
  // 1. Draw M marginalised latents x_m ~ marginalDistIR.
  return setup.then(() => ctx.sendWorker({
    type: 'sampleN', ir: marginalDistIR, count: M, refArrays: {},
    seed: nameSeed(tag + ':marg', ctx.rootKey),
  })).then((xReply: any) => {
    const xs = xReply.samples;
    // 2. Lay out the D×M grid, d-major: atom d*M+m ↦ (z=data[d], x=xs[m]).
    const zcol = new Float64Array(count);
    const xcol = new Float64Array(count);
    for (let d = 0; d < D; d++) {
      const zd = data[d];
      for (let m = 0; m < M; m++) {
        zcol[d * M + m] = zd;
        xcol[d * M + m] = xs[m];
      }
    }
    // 3. u*_m = inverseIR(z; x), one batched pass.
    return ctx.sendWorker({
      type: 'evaluateN', ir: inv.inverseIR, count,
      refArrays: { [OUT.name]: zcol, [marginalRef.name]: xcol },
    }).then((uReply: any) => {
      const ustar = uReply.samples;
      // 4. ladj_fwd(u*; x), one batched pass.
      return ctx.sendWorker({
        type: 'evaluateN', ir: inv.ladjIR, count,
        refArrays: { [retainedRef.name]: ustar, [marginalRef.name]: xcol },
      }).then((ladjReply: any) => {
        const ladj = ladjReply.samples;
        // 5. Conditional log-density per atom: logp_ret(u*) − ladj_fwd(u*).
        //    Uniform innovation on [lo,hi]: logp_ret = −log(hi−lo) inside,
        //    −inf outside (the support mask that bounds the fiber).
        const lo = retainedInterval[0], hi = retainedInterval[1];
        const logWidth = Math.log(hi - lo);
        const cond = new Float64Array(count);
        for (let i = 0; i < count; i++) {
          const u = ustar[i];
          cond[i] = (u >= lo && u <= hi && Number.isFinite(ladj[i]))
            ? (-logWidth - ladj[i]) : -Infinity;
        }
        // 6. Per data point: logsumexp over the M innovations − log M.
        //    iid ⇒ sum the per-event log-densities.
        const logM = Math.log(M);
        let logL = 0;
        for (let d = 0; d < D; d++) {
          const slice = cond.subarray(d * M, d * M + M);
          logL += empirical.logSumExp(slice) - logM;
        }
        return logL;
      });
    });
  });
}

// =====================================================================
// Auto-derivation of the MC-likelihood recipe from model bindings
// =====================================================================

// Inline value bindings into a self-contained recipe (cf. mat-broadcast
// `_inlineGenerativeBody`): draws / boundary / fixed / function-like
// bindings stay as refs; every other value binding is inlined. Bounded
// recursion (spec §04 modules are DAGs).
function _inlineRecipe(ir: any, bindings: any, drawNames: Set<string>, boundary: Set<string>): any {
  if (!ir || typeof ir !== 'object') return ir;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const nm = ir.name;
    if (drawNames.has(nm) || boundary.has(nm)) return ir;
    const b = bindings.get(nm);
    if (!b || !b.ir || shared.isFunctionLikeBinding(b) || b.phase === 'fixed') return ir;
    return _inlineRecipe(b.ir, bindings, drawNames, boundary);
  }
  if (ir.kind !== 'call') return ir;
  const out: any = { kind: 'call' };
  if (ir.op) out.op = ir.op;
  if (ir.target) out.target = ir.target;
  if (Array.isArray(ir.args)) out.args = ir.args.map((a: any) => _inlineRecipe(a, bindings, drawNames, boundary));
  if (ir.kwargs) { out.kwargs = {}; for (const k in ir.kwargs) out.kwargs[k] = _inlineRecipe(ir.kwargs[k], bindings, drawNames, boundary); }
  if (Array.isArray(ir.fields)) out.fields = ir.fields.map((f: any) => ({ ...f, value: _inlineRecipe(f && f.value, bindings, drawNames, boundary) }));
  return out;
}

// Follow a draw binding `name = draw(ref M)` to its measure IR (M's ir).
function _drawMeasureIR(name: string, bindings: any): any {
  const b = bindings.get(name);
  if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'draw') return null;
  const arg = b.ir.args && b.ir.args[0];
  if (arg && arg.kind === 'ref' && arg.ns === 'self') {
    const mb = bindings.get(arg.name);
    return (mb && mb.ir) ? mb.ir : arg;
  }
  return arg || null;
}

// [lo,hi] from a Uniform(interval(lo,hi)) measure IR (constant bounds),
// else null. v1 retained-innovation support.
function _uniformInterval(mIR: any, bindings: any): number[] | null {
  if (!mIR || mIR.kind !== 'call' || mIR.op !== 'Uniform') return null;
  let supp = (mIR.args && mIR.args[0]) || (mIR.kwargs && (mIR.kwargs.support || mIR.kwargs.set));
  if (supp && supp.kind === 'ref' && supp.ns === 'self') {
    const sb = bindings.get(supp.name); if (sb && sb.ir) supp = sb.ir;
  }
  if (!supp || supp.kind !== 'call' || supp.op !== 'interval') return null;
  const resolveConstant = require('./ir-shared.ts').resolveConstant;
  const lo = resolveConstant(supp.args[0], bindings, new Set());
  const hi = resolveConstant(supp.args[1], bindings, new Set());
  if (typeof lo !== 'number' || typeof hi !== 'number') return null;
  return [lo, hi];
}

/**
 * Auto-derive the MC-likelihood inputs from the model bindings, for a
 * scalar per-event variate `eventVar` reified over boundary params
 * `boundaryNames` (e.g. 'z' over ['pars'] for the transport model).
 *
 * Walks eventVar's ancestor DAG to its draw bindings, inlines the recipe
 * (boundary + draws left as refs), and classifies the draws by trying
 * `invertExpr` on each — NO heuristic: the invertible draw is the
 * RETAINED innovation, the rest are MARGINALISED. Resolves the retained
 * latent's support (Uniform, v1) and each marginalised latent's
 * distribution IR. Returns null when the structure isn't the v1 shape
 * (no invertible-Uniform draw, or ≠1 marginalised latent).
 */
function deriveMcLikelihoodRecipe(eventVar: string, boundaryNames: string[], bindings: any): any {
  const boundary = new Set(boundaryNames);
  const drawNames = new Set<string>();
  const seen = new Set<string>();
  (function walk(name: string) {
    if (seen.has(name) || boundary.has(name)) return;
    seen.add(name);
    const b = bindings.get(name);
    if (!b || !b.ir) return;
    if (b.ir.kind === 'call' && b.ir.op === 'draw') { drawNames.add(name); return; }
    for (const r of orchestrator.collectSelfRefs(b.ir)) walk(r);
  })(eventVar);
  const evB = bindings.get(eventVar);
  if (!evB || !evB.ir) return null;
  const recipeIR = _inlineRecipe(evB.ir, bindings, drawNames, boundary);
  const bijReg = require('./bijection-registry.ts');
  const OUT = { kind: 'ref', ns: '%mc', name: '__mc_z__' };
  let retainedRef: any = null; let retainedInterval: any = null;
  const marginals: string[] = [];
  for (const dn of drawNames) {
    const r = !retainedRef && bijReg.invertExpr({ outputExpr: recipeIR, freeRef: { name: dn }, outputValue: OUT });
    const ivl = r ? _uniformInterval(_drawMeasureIR(dn, bindings), bindings) : null;
    if (r && ivl) { retainedRef = { name: dn }; retainedInterval = ivl; }
    else marginals.push(dn);
  }
  if (!retainedRef || marginals.length !== 1) return null;   // v1 shape
  const marginalDistIR = _drawMeasureIR(marginals[0], bindings);
  if (!marginalDistIR) return null;
  return { recipeIR, retainedRef, retainedInterval, marginalRef: { name: marginals[0] }, marginalDistIR };
}

module.exports = {
  matBayesupdate,
  matLogdensityof,
  matBroadcastLogdensity,
  matTotalmass,
  mcMarginalLogDensity,
  deriveMcLikelihoodRecipe,
};
