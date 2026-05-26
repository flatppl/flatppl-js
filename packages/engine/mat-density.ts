'use strict';

// Density-related materialiser handlers: matBayesupdate,
// matLogdensityof, matBroadcastLogdensity, matTotalmass + the
// jointchain handler (matJointchain) that matLogdensityof's N-ary
// kchain path depends on. Plus the runtime-weight selector resolution
// (collectRuntimeWeightNodes / resolveRuntimeWeights) that
// matLogdensityof and matBroadcastLogdensity both run before density
// evaluation (engine-concepts §11 MC-weight selector).
//
// matJointchain materialises a first-class jointchain/kchain measure
// (spec §06): structurally walk the explicit step structure, sampling
// each step with the prior variates threaded as refArrays. kchain
// drops all but the last step's variate(s); jointchain keeps them all.
// matLogdensityof calls matJointchain to materialise the (n-1)-joint
// history for N-ary kchain MC density (engine-concepts §6).

import type {
  DerivationBayesupdate,
  DerivationBroadcastLogdensity,
  DerivationJointchain,
  DerivationLogdensityof,
  DerivationTotalmass,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const shared       = require('./materialiser-shared.ts');

const {
  nameSeed,
  isFunctionLikeBinding,
  collectRefArrays,
  prepareDensityRefs,
  pushFixedEnv,
  measureFromValue,
  measureFromReply,
  measureN,
  scalarMeasureN,
  resolveFnBody,
} = shared;

// =====================================================================
// Bayesupdate — reweight prior atoms by per-atom log-likelihood
// =====================================================================

function matBayesupdate(d: DerivationBayesupdate, ctx: any) {
  // Per spec: posterior = bayesupdate(L, prior), L = likelihoodof(K, obs)
  // For each prior atom θ_i, logw_i = logdensityof(K(θ_i), obs).
  // The atoms are the prior's; logWeights = prior.logWeights + per-i logp.
  const bodyIR = orchestrator.expandMeasure(d.bodyIR || d.bodyName,
    { derivations: ctx.derivations, bindings: ctx.bindings });
  if (!bodyIR) {
    return Promise.reject(new Error('bayesupdate: cannot expand body into measure IR'));
  }
  return Promise.all([
    ctx.getMeasure(d.from),
    prepareDensityRefs(bodyIR, ctx, 'bayesupdate'),
  ]).then(([parent, prep]: [any, any]) => {
    const { refArrays, fixedEnv } = prep;
    const observed = orchestrator.resolveIRToValue(
      d.obsIR, ctx.bindings, ctx.fixedValues);
    return pushFixedEnv(ctx, fixedEnv).then(() => ctx.sendWorker({
      type: 'logDensityN',
      ir: bodyIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      observed: observed,
      tally: 'clamped',
    })).then((reply: any) => {
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
  const measureDeriv = ctx.derivations[d.measureName];
  const isChain = !!(measureDeriv
    && measureDeriv.kind === 'jointchain' && measureDeriv.marginalize);

  const measureIR0 = orchestrator.expandMeasure(d.measureName,
    { derivations: ctx.derivations, bindings: ctx.bindings });
  if (!measureIR0) {
    return Promise.reject(new Error('logdensityof: cannot expand measure "'
      + d.measureName + '" into a self-contained IR'));
  }
  return resolveRuntimeWeights(measureIR0, ctx).then((measureIR) => {
  const naryKchain = isChain && measureDeriv
    && Array.isArray(measureDeriv.steps) && measureDeriv.steps.length > 2
    && !measureDeriv.labels;
  const innerJointP = naryKchain
    ? matJointchain(d.measureName + '$jchist',
        { kind: 'jointchain', marginalize: false, labels: null,
          steps: measureDeriv.steps.slice(0, -1) }, ctx)
    : Promise.resolve(null);
  return Promise.all([
    prepareDensityRefs(measureIR, ctx, 'logdensityof'),
    innerJointP,
  ]).then(([prep, innerJoint]: [any, any]) => {
    const { refArrays, fixedEnv } = prep;
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
        refArrays['s' + i] = cs;
      }
    }
    const observed = orchestrator.resolveIRToValue(
      d.obsIR, ctx.bindings, ctx.fixedValues);
    return pushFixedEnv(ctx, fixedEnv).then(() => ctx.sendWorker({
      type: 'logDensityN',
      ir: measureIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      observed: observed,
      tally: 'clamped',
    })).then((reply: any) => {
      if (!isChain) {
        return scalarMeasureN(reply.samples, {
          logWeights: null,
          logTotalmass: 0,
          n_eff: reply.samples.length,
        });
      }
      // chain marginalisation reduction.
      const perAtom = reply.samples;
      const lse = empirical.logSumExp(perAtom);
      const margLogp = lse - Math.log(perAtom.length);
      const out = new Float64Array(perAtom.length);
      out.fill(margLogp);
      return scalarMeasureN(out, {
        logWeights: null,
        logTotalmass: 0,
        n_eff: 1,
      });
    });
  });
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
// matJointchain — first-class jointchain/kchain materialisation
// =====================================================================
//
// Spec §06 stochastic-node equivalence WITHOUT the inlineChainOps
// rewrite. Sample the base then iteratively expand each kernel's
// body, threading the prior step variates as refArrays. NAMED params
// auto-splat to like-named prior fields (spec §04); a lone HOLE param
// rewires to the prior cat (spec §06 `c~K3([a,b])`).
//
// `marginalize=true` (kchain) ⇒ keep only the last step's variate(s);
// the prior is integrated out by matLogdensityof's MC reduction.

function matJointchain(name: string, d: DerivationJointchain, ctx: any) {
  const steps = (d && d.steps) || [];
  if (steps.length < 2) {
    return Promise.reject(new Error('jointchain: need at least 2 steps'));
  }
  const base = steps[0];
  // Kernel-first base rejection lives at materialiseMeasure's entry-
  // point gate (engine-concepts §19.2 + §19.5) — the binding's
  // inferredType.kind === 'kernel' surfaces a clear diagnostic
  // before dispatch. matJointchain itself only handles closed-first
  // (residual = ∅) materialisation; reaching it with base.kernel
  // would indicate a missing classifier path or a synthetic call
  // site outside materialiseMeasure, so we keep a defensive
  // assertion here.
  if (base.kernel) {
    return Promise.reject(new Error(
      'matJointchain: kernel-first base reached the handler — should '
      + 'have been rejected at materialiseMeasure entry-point gate '
      + '(engine-concepts §19); a synthetic dispatch may have skipped '
      + 'the gate.'));
  }
  const baseP = base.ref != null
    ? ctx.getMeasure(base.ref)
    : ctx.sendWorker({
        type: 'sampleN', ir: base.measureIR, count: ctx.sampleCount,
        refArrays: {}, seed: nameSeed(name + ':jc0', ctx.rootSeed),
      }).then((r: any) => measureFromReply(r, ctx.sampleCount,
        { logTotalmass: 0, n_eff: ctx.sampleCount }));

  return Promise.resolve(baseP).then((M0) => {
    const priorVars: any[] = [];
    if (M0 && M0.samples) {
      priorVars.push({ name: 's0', m: M0 });
    } else if (M0 && M0.fields) {
      for (const k of Object.keys(M0.fields)) {
        priorVars.push({ name: k, m: M0.fields[k] });
      }
    } else {
      return Promise.reject(new Error(
        'jointchain: base measure must be scalar- or record-shaped '
        + '(tuple/iid base is a follow-up)'));
    }
    const N = priorVars[0].m.samples.length;
    const baseRefName = base.ref;

    const kernelLeaves = (kstep: any, fallbackName: any) => {
      let f = kstep.kernelIR;
      if (!f && kstep.ref != null) {
        const kb = ctx.bindings && ctx.bindings.get(kstep.ref);
        if (kb && kb.ir && kb.ir.kind === 'call'
            && kb.ir.op === 'functionof') f = kb.ir;
        else {
          const fi = resolveFnBody(kb, ctx.bindings);
          if (fi) f = { params: [fi.paramName], body: fi.body };
        }
      }
      if (!f || !f.body || !Array.isArray(f.params)
          || f.params.length === 0) return null;
      let body = f.body;
      if (body.kind === 'ref' && body.ns === 'self') {
        body = orchestrator.expandMeasure(body.name,
          { derivations: ctx.derivations, bindings: ctx.bindings });
        if (!body) return null;
      }
      let leaves;
      if (body.kind === 'call'
          && (body.op === 'joint' || body.op === 'record')
          && Array.isArray(body.fields)) {
        leaves = body.fields.map((fl: any) => ({ name: fl.name, ir: fl.value }));
      } else {
        leaves = [{ name: fallbackName, ir: body }];
      }
      return { params: f.params, leaves };
    };

    const PRIOR = (j: any) => '__jc$' + j;
    const bindLeaf = (leafIR: any, params: any) => {
      const refArrays: any = {};
      const named = params.filter((p: any) =>
        priorVars.some((v) => v.name === p));
      const holes = params.filter((p: any) => named.indexOf(p) === -1);
      for (const p of named) {
        const v = priorVars.find((q) => q.name === p);
        refArrays[p] = v.m.samples;
      }
      let ir = leafIR;
      if (holes.length > 0) {
        if (params.length !== 1) return null;
        const catVars = priorVars.length > 0
          ? priorVars
          : [{ name: baseRefName, m: M0 }];
        const k = catVars.length;
        for (let j = 0; j < k; j++) refArrays[PRIOR(j)] = catVars[j].m.samples;
        const param = holes[0];
        const sub: any = (node: any) => {
          if (node == null || typeof node !== 'object') return node;
          if (Array.isArray(node)) return node.map(sub);
          if (node.kind === 'ref'
              && (node.ns === '%local' || node.ns === 'self')
              && node.name === param) {
            if (k === 1) {
              return { kind: 'ref', ns: 'self', name: PRIOR(0) };
            }
            return { kind: 'call', op: 'vector',
              args: catVars.map((_, j) =>
                ({ kind: 'ref', ns: 'self', name: PRIOR(j) })) };
          }
          const out: Record<string, any> = {};
          for (const key in node) out[key] = sub(node[key]);
          return out;
        };
        ir = sub(leafIR);
      }
      return { ir, refArrays };
    };

    let chainP: Promise<any[]> = Promise.resolve([]);
    for (let i = 1; i < steps.length; i++) {
      const kstep = steps[i];
      chainP = chainP.then((produced) => {
        const ke = kernelLeaves(kstep, 's' + i);
        if (!ke) {
          return Promise.reject(new Error(
            `jointchain: step ${i} kernel `
            + `${kstep.ref ? `'${kstep.ref}' ` : '(inline) '}`
            + 'has no resolvable functionof body'));
        }
        let p: Promise<any[]> = Promise.resolve(produced);
        for (let li = 0; li < ke.leaves.length; li++) {
          const leaf = ke.leaves[li];
          p = p.then((acc: any[]) => {
            const vn = leaf.name != null ? leaf.name : ('s' + i);
            // Composite kernel-body field: an `iid(D, n)` field.
            // Reuse matIid's worker mechanism — sampleN of the inner
            // leaf with repeat=n — so the iid draws share the per-atom
            // parameter context.
            if (leaf.ir && leaf.ir.kind === 'call'
                && leaf.ir.op === 'iid'
                && Array.isArray(leaf.ir.args) && leaf.ir.args.length === 2) {
              let inner = leaf.ir.args[0];
              if (inner && inner.kind === 'ref' && inner.ns === 'self') {
                inner = orchestrator.leafSampleIR(inner.name, ctx.derivations)
                  || inner;
              }
              const cIR = leaf.ir.args[1];
              let reps: number | null = null;
              if (cIR && cIR.kind === 'lit' && typeof cIR.value === 'number') {
                reps = cIR.value | 0;
              } else {
                try {
                  const env: Record<string, any> = {};
                  if (ctx.fixedValues) {
                    for (const [k2, v2] of ctx.fixedValues) env[k2] = v2;
                  }
                  const cv = require('./sampler.ts').evaluateExpr(cIR, env);
                  if (typeof cv === 'number' && Number.isFinite(cv)) {
                    reps = cv | 0;
                  }
                } catch (_) { reps = null; }
              }
              if (reps == null || reps < 0 || !inner
                  || inner.kind !== 'call'
                  || !require('./sampler.ts').isKnownDistribution(inner.op)) {
                return Promise.reject(new Error(
                  `jointchain: step ${i} composite kernel-body field `
                  + `'${vn}' — only iid(<leaf dist>, <fixed n>) is `
                  + 'supported (deeper nesting is a follow-up)'));
              }
              const bnd = bindLeaf(inner, ke.params);
              if (!bnd) {
                return Promise.reject(new Error(
                  `jointchain: step ${i} iid-field kernel param binding `
                  + 'unsupported'));
              }
              return collectRefArrays(bnd.ir, ctx.fixedValues, ctx.getMeasure)
                .then((extra: any) => ctx.sendWorker({
                  type: 'sampleN', ir: bnd.ir, count: N, repeat: reps,
                  refArrays: Object.assign({}, extra, bnd.refArrays),
                  seed: nameSeed(name + ':jc' + i + '$' + li, ctx.rootSeed),
                })).then((reply: any) => {
                  const Mi = Object.assign(
                    empirical.arrayMeasure(reply.samples, [reps], null),
                    {
                      value: { shape: [N, reps], data: reply.samples },
                      logTotalmass: 0, n_eff: N,
                    });
                  priorVars.push({ name: vn, m: Mi });
                  return acc.concat([{ name: vn, m: Mi }]);
                });
            }
            const bound = bindLeaf(leaf.ir, ke.params);
            if (!bound) {
              return Promise.reject(new Error(
                `jointchain: step ${i} multi-param kernel with an `
                + 'unmatched param (not a prior field) is unsupported'));
            }
            return ctx.sendWorker({
              type: 'sampleN', ir: bound.ir, count: N,
              refArrays: bound.refArrays,
              seed: nameSeed(name + ':jc' + i + '$' + li, ctx.rootSeed),
            }).then((reply: any) => {
              const Mi = measureFromReply(reply, N, {
                logWeights: priorVars[0].m.logWeights,
                logTotalmass: 0, n_eff: priorVars[0].m.n_eff,
              });
              if (!Mi.samples) {
                return Promise.reject(new Error(
                  `jointchain: step ${i} kernel leaf produced a `
                  + 'non-scalar variate (follow-up)'));
              }
              priorVars.push({ name: vn, m: Mi });
              return acc.concat([{ name: vn, m: Mi }]);
            });
          });
        }
        return p;
      });
    }

    return chainP.then((produced) => {
      const lastStepCount = (() => {
        const ke = kernelLeaves(steps[steps.length - 1], 's');
        return ke ? ke.leaves.length : 1;
      })();
      let outPairs;
      if (d.marginalize) {
        outPairs = produced.slice(produced.length - lastStepCount);
      } else {
        const basePairs = (M0 && M0.fields)
          ? Object.keys(M0.fields).map((k) => ({ name: k, m: M0.fields[k] }))
          : [{ name: (d.labels && d.labels[0]) || 's0', m: M0 }];
        outPairs = basePairs.concat(produced);
      }
      const subs = outPairs.map((x: any) => x.m);
      const lw = empirical.propagateLogWeights(subs);
      let nEff = N;
      for (const s of subs) {
        if (s.n_eff != null) nEff = Math.min(nEff, s.n_eff);
      }
      // Variate shape mirrors the kernel/base structure. Unwrap to a
      // bare scalar measure ONLY for a lone SYNTHETIC-named scalar
      // (scalar-base kchain / hole-kernel scalar body). Multiple /
      // named ⇒ record; positional all-synthetic-scalar ⇒ tuple.
      if (outPairs.length === 1
          && /^s\d+$/.test(outPairs[0].name)
          && outPairs[0].m && outPairs[0].m.samples
          && !outPairs[0].m.fields) {
        return Object.assign({}, outPairs[0].m,
          { logTotalmass: 0, n_eff: nEff });
      }
      const named = !!d.labels || (M0 && M0.fields)
        || outPairs.some((x: any) => !/^s\d+$/.test(x.name));
      if (named) {
        const fields: any = {};
        for (let i = 0; i < outPairs.length; i++) {
          const nm = (d.labels && !d.marginalize && d.labels[i] != null)
            ? d.labels[i] : outPairs[i].name;
          fields[nm] = outPairs[i].m;
        }
        return Object.assign(empirical.recordMeasure(fields, lw),
          { logTotalmass: 0, n_eff: nEff });
      }
      return Object.assign(empirical.tupleMeasure(subs, lw),
        { logTotalmass: 0, n_eff: nEff });
    });
  });
}

module.exports = {
  matBayesupdate,
  matLogdensityof,
  matBroadcastLogdensity,
  matTotalmass,
  matJointchain,
};
