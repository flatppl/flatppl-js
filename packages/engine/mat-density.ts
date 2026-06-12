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
  // Normalize-mass specs (audit M3) resolve on every matScore path
  // including bayesupdate: the resolution is a pure correctness fix
  // (−log Z literalised from the inner measure's tracked logTotalmass),
  // and walkNormalize refuses an unresolved spec loudly. node.body is
  // freshly built per lowerMeasure call (no cache), so the in-place
  // resolution is safe (mutation hazard — plan critique).
  const bodyP = resolveNormalizeMasses(node.body, ctx);
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
  if (kwargs.length === 1) {
    boundaries[kwargs[0]] = theta;
  } else {
    for (const k of kwargs) {
      if (theta === null || typeof theta !== 'object' || !(k in theta)) {
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
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
  return matScore(node, ctx, { observed })
    .then((reply: any) => applyReduce(reply, node));
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
  matBroadcastLogdensity,
  matTotalmass,
};
