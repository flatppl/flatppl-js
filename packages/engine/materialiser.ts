'use strict';

// Per-binding measure materialisation.
//
// Given a binding name and a context bundle (derivations, bindings,
// fixedValues, a recursive getMeasure callback, a worker handle, the
// global sample count, and a root seed for per-binding RNG streams),
// materialiseMeasure(name, ctx) returns a Promise<Measure> describing
// the binding's empirical distribution.
//
// EmpiricalMeasure (engine-types.d.ts) shape:
//
//   { samples:      Float64Array,             // per-atom values (scalar-leaf view)
//     value?:       Value,                    // shape-explicit view
//     logWeights:   Float64Array | null,      // null = uniform 1/N
//     logTotalmass: number,                   // default 0 (= totalmass 1)
//     n_eff:        number,                   // default = samples.length
//     fields?:      { name → Measure },       // record/joint shape
//     elems?:       Measure[],                // tuple shape
//     shape?:       'record' | 'tuple' | 'array',  // kind discriminator (string)
//     dims?:        number[],                 // intrinsic-dim suffix for
//                                             //   vector-atom measures (matIid)
//     ... }
//
// The materialiser was split (engine-concepts §17.5) across:
//   - materialiser-shared.ts — shape builders, refArrays plumbing,
//     fixedValueToMeasure, RNG helpers, setBoundsForMat.
//   - mat-multivariate.ts — 8 multivariate distribution handlers.
//   - mat-broadcast.ts — matKernelBroadcast (array-product measure).
//   - mat-density.ts — matBayesupdate / matLogdensityof /
//     matBroadcastLogdensity / matTotalmass / matJointchain +
//     runtime-weight selector resolution.
//   - mat-transformations.ts — matPushfwd, matTruncate.
//
// This file keeps the basic / algebra handlers (sample, alias,
// evaluate, array, weighted, normalize, iid, tuple, record,
// superpose, select), the KIND_HANDLERS dispatch, the
// materialiseMeasure entry point, and the facade re-exports.

import type {
  DerivationAlias,
  DerivationArray,
  DerivationEvaluate,
  DerivationIid,
  DerivationNormalize,
  DerivationRecord,
  DerivationSample,
  DerivationSelect,
  DerivationSuperpose,
  DerivationTuple,
  DerivationWeighted,
  EmpiricalMeasure,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const shared       = require('./materialiser-shared.ts');
const multivariate = require('./mat-multivariate.ts');
const broadcast    = require('./mat-broadcast.ts');
const density      = require('./mat-density.ts');
const transforms   = require('./mat-transformations.ts');

const {
  nameSeed,
  makeMainThreadPrng,
  collectRefArrays,
  prepareDensityRefs,
  classifyProfileSelfRefs,
  pushFixedEnv,
  fixedValueToMeasure,
  measureFromValue,
  measureFromReply,
  scalarMeasureN,
  valueOf,
} = shared;

// =====================================================================
// Per-kind handlers — basic + algebra
// =====================================================================

function matSample(name: string, d: DerivationSample, ctx: any) {
  return collectRefArrays(d.distIR, ctx.fixedValues, ctx.getMeasure)
    .then((refArrays: any) => ctx.sendWorker({
      type: 'sampleN',
      ir: d.distIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      seed: nameSeed(name, ctx.rootSeed),
    }))
    .then((reply: any) => {
      const lw = reply.logWeights || null;
      return scalarMeasureN(reply.samples, {
        logWeights: lw,
        logTotalmass: 0,
        n_eff: reply.samples.length,
      });
    });
}

function matAlias(d: DerivationAlias, ctx: any) {
  // Alias: same measure record — reference equality is intentional so
  // click-flipping between a variate and its measure is free, and the
  // shared logWeights ref preserves propagateLogWeights's dedupe contract.
  return ctx.getMeasure(d.from);
}

function matEvaluate(d: DerivationEvaluate, ctx: any) {
  // Deterministic transform of variates. Per-atom: c_i = f(parents_i).
  // logWeights propagate via joint IS (sum independent events, dedupe
  // shared via reference identity). logTotalmass follows from the
  // resulting logWeights; n_eff is min(parents') as a fast bound.
  return prepareDensityRefs(d.ir, ctx, 'matEvaluate').then((prep: any) => {
    const { refArrays, fixedEnv, perAtomNames } = prep;
    return Promise.all(perAtomNames.map(ctx.getMeasure)).then((parentMeasures: any[]) => {
      return pushFixedEnv(ctx, fixedEnv).then(() => ctx.sendWorker({
        type: 'evaluateN',
        ir: d.ir,
        count: ctx.sampleCount,
        refArrays: refArrays,
      })).then((reply: any) => {
        const lw = empirical.propagateLogWeights(parentMeasures);
        let n_eff = ctx.sampleCount;
        for (const p of parentMeasures) {
          if (typeof p.n_eff === 'number') n_eff = Math.min(n_eff, p.n_eff);
        }
        const logTotalmass = lw ? empirical.logSumExp(lw) : 0;
        return measureFromReply(reply, ctx.sampleCount, {
          logWeights: lw,
          logTotalmass: logTotalmass,
          n_eff: n_eff,
        });
      });
    });
  });
}

function matArray(d: DerivationArray) {
  // Static array literal — values verbatim, no sampling, no worker
  // round-trip. Length equals the array's literal length, NOT the
  // sample count; downstream plot dispatches by mode for this kind.
  const samples = Float64Array.from(d.values);
  return Promise.resolve(scalarMeasureN(samples, {
    logWeights: null,
    logTotalmass: 0,
    n_eff: samples.length,
  }));
}

function matWeighted(d: DerivationWeighted, ctx: any) {
  // weighted(w, base) / logweighted(lw, base): shift each parent
  // atom's logWeight by log(w_i) (or lw_i directly). totalmass scales
  // by the average weight; the empirical log-total-mass is
  // logSumExp(resulting logWeights).
  return ctx.getMeasure(d.from).then((parent: any) => {
    const lifted = empirical.materialiseUniform(parent);
    const N = lifted.logWeights.length;
    const w = new Float64Array(N);
    if (d.weightIR) {
      return collectRefArrays(d.weightIR, ctx.fixedValues, ctx.getMeasure).then((refArrays: any) =>
        ctx.sendWorker({
          type: 'evaluateN',
          ir: d.weightIR,
          count: ctx.sampleCount,
          refArrays: refArrays,
        })
      ).then((reply: any) => {
        const weights = reply.samples;
        let nonPos = 0;
        if (d.isLog) {
          for (let i = 0; i < N; i++) w[i] = lifted.logWeights[i] + weights[i];
        } else {
          for (let j = 0; j < N; j++) {
            const v = weights[j];
            if (v > 0) {
              w[j] = lifted.logWeights[j] + Math.log(v);
            } else {
              w[j] = -Infinity;
              if (v < 0) nonPos++;
            }
          }
          if (nonPos > 0) {
            // eslint-disable-next-line no-console
            console.warn('weighted: ' + nonPos
              + ' negative weight sample(s) treated as zero mass');
          }
        }
        const lTM = empirical.logSumExp(w);
        const nEff = empirical.effectiveSampleSize({ samples: lifted.samples, logWeights: w });
        return scalarMeasureN(lifted.samples,
          { logWeights: w, logTotalmass: lTM, n_eff: nEff });
      });
    }
    // Constant fast path: orchestrator pre-computed d.logShift (a
    // uniform per-atom additive shift). totalmass simply scales.
    for (let i = 0; i < N; i++) w[i] = lifted.logWeights[i] + d.logShift;
    const parentLTM = (typeof parent.logTotalmass === 'number') ? parent.logTotalmass : 0;
    return scalarMeasureN(lifted.samples, {
      logWeights: w,
      logTotalmass: parentLTM + d.logShift,
      n_eff: (typeof parent.n_eff === 'number') ? parent.n_eff : N,
    });
  });
}

function matNormalize(d: DerivationNormalize, ctx: any) {
  // normalize(base): shift weights so they sum to 1 (logTotalmass = 0).
  return ctx.getMeasure(d.from).then((parent: any) => {
    const lifted = empirical.materialiseUniform(parent);
    const N = lifted.logWeights.length;
    const lse = empirical.logSumExp(lifted.logWeights);
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) w[i] = lifted.logWeights[i] - lse;
    const nEff = empirical.effectiveSampleSize({ samples: lifted.samples, logWeights: w });
    return scalarMeasureN(lifted.samples,
      { logWeights: w, logTotalmass: 0, n_eff: nEff });
  });
}

function matIid(name: string, d: DerivationIid, ctx: any) {
  // iid(M, n, …): N atoms × k inner draws, atom-major packed into
  // one Float64Array. Worker's sampleN takes an optional repeat=k.
  // Parameters are pinned per atom (refArrays), so iid samples within
  // atom i share atom-i's parameter context.
  //
  // totalmass: M^k (in log: k · M.logTotalmass).
  // n_eff: inherits M's n_eff.
  const distIR = orchestrator.leafSampleIR(d.from, ctx.derivations);
  if (!distIR) {
    return Promise.reject(new Error('iid: cannot resolve leaf sample IR for ' + d.from));
  }
  const k = d.dims.reduce((p: any, n: any) => p * n, 1);
  return collectRefArrays(distIR, ctx.fixedValues, ctx.getMeasure)
    .then((refArrays: any) => ctx.sendWorker({
      type: 'sampleN', ir: distIR, count: ctx.sampleCount, repeat: k,
      refArrays: refArrays,
      seed: nameSeed(name, ctx.rootSeed),
    }))
    .then((reply: any) => {
      const value = { shape: [ctx.sampleCount | 0].concat(d.dims), data: reply.samples };
      return Object.assign(
        empirical.arrayMeasure(reply.samples, d.dims, null),
        { value: value, logTotalmass: 0, n_eff: ctx.sampleCount },
      );
    });
}

function matTuple(d: DerivationTuple, ctx: any) {
  // Positional analogue of record. Each element materialises
  // independently; combine into a tuple Measure whose components live
  // in elems. Top-level logWeights is the join of components'.
  return Promise.all(d.elems.map(ctx.getMeasure)).then((subs: any[]) => {
    const lw = empirical.propagateLogWeights(subs);
    let lTM = 0;
    let nEff = ctx.sampleCount;
    for (const s of subs) {
      if (typeof s.logTotalmass === 'number') lTM += s.logTotalmass;
      if (typeof s.n_eff === 'number') nEff = Math.min(nEff, s.n_eff);
    }
    return Object.assign(
      empirical.tupleMeasure(subs, lw),
      { logTotalmass: lTM, n_eff: nEff },
    );
  });
}

function matRecord(d: DerivationRecord, ctx: any) {
  // Multivariate (record / joint): each field's source binding gets
  // materialised; assembled into a record-shaped Measure (SoA — one
  // sub-measure per field). Top-level logWeights is the join across
  // fields; logTotalmass is the sum of fields' (independent product
  // measures multiply masses).
  const fieldNames = Object.keys(d.fields);
  const fieldDeps  = fieldNames.map((k) => d.fields[k]);
  return Promise.all(fieldDeps.map(ctx.getMeasure)).then((subs: any[]) => {
    const fields: any = {};
    let lTM = 0;
    let nEff = ctx.sampleCount;
    for (let i = 0; i < fieldNames.length; i++) {
      fields[fieldNames[i]] = subs[i];
      if (typeof subs[i].logTotalmass === 'number') lTM += subs[i].logTotalmass;
      if (typeof subs[i].n_eff === 'number') nEff = Math.min(nEff, subs[i].n_eff);
    }
    const lw = empirical.propagateLogWeights(subs);
    return Object.assign(
      empirical.recordMeasure(fields, lw),
      { logTotalmass: lTM, n_eff: nEff },
    );
  });
}

function matSuperpose(name: string, d: DerivationSuperpose, ctx: any) {
  // Superpose: concat parents' samples + logWeights, systematic-
  // resample to ctx.sampleCount. Mass-faithful: result's totalmass
  // equals the sum of parents' totalmasses; resampling produces
  // equally-weighted atoms each carrying (totalInputMass / N) of mass.
  return Promise.all(d.fromNames.map(ctx.getMeasure)).then((parents: any[]) => {
    let totalN = 0;
    for (const p of parents) totalN += p.samples.length;
    if (totalN === 0) {
      return measureFromValue(
        { shape: [0], data: new Float64Array(0) },
        { logWeights: null, logTotalmass: -Infinity, n_eff: 0 });
    }
    const combinedSamples = new Float64Array(totalN);
    const combinedLogWeights = new Float64Array(totalN);
    let offset = 0;
    for (const p of parents) {
      const lifted = empirical.materialiseUniform(p);
      combinedSamples.set(lifted.samples, offset);
      combinedLogWeights.set(lifted.logWeights, offset);
      offset += lifted.samples.length;
    }
    const prng = makeMainThreadPrng(nameSeed(name, ctx.rootSeed));
    const idx = empirical.systematicResample(combinedLogWeights, ctx.sampleCount, prng);
    const out = new Float64Array(ctx.sampleCount);
    for (let i = 0; i < ctx.sampleCount; i++) out[i] = combinedSamples[idx[i]];
    const totalLogMass = empirical.logSumExp(combinedLogWeights);
    const perAtom = totalLogMass - Math.log(ctx.sampleCount);
    const outW = new Float64Array(ctx.sampleCount);
    outW.fill(perAtom);
    return scalarMeasureN(out, {
      logWeights: outW,
      logTotalmass: totalLogMass,
      // After systematic resampling the atoms are uniform → n_eff = N.
      n_eff: ctx.sampleCount,
    });
  });
}

function matSelect(name: string, d: DerivationSelect, ctx: any) {
  // Discrete-selector mixture generation (engine-concepts §11): the
  // SAMPLING half of the shared select core (density is walkSelect).
  // Eval-all-branches-then-gather — draw every branch's full N-atom
  // batch AND the realised per-atom selector, then pick per atom:
  //
  //   out[i] = branch_{idx(selector[i])}.samples[i]
  //
  // Drawing all branches for every atom (even unpicked ones) keeps
  // the per-atom RNG threading consistent (the same property matIid /
  // the worker sampleN rely on); the gather itself is deterministic.
  // `marginalize` is a DENSITY-only concept — for sampling the
  // selector is always realised (never marginalised).
  const branchEntries = d.branches || [];
  if (branchEntries.length === 0) {
    return Promise.reject(new Error('matSelect: select has no branches'));
  }
  if (!d.selectorRef) {
    return Promise.reject(new Error(
      'matSelect: sampling a select/ifelse needs a materialisable '
      + 'selector (a named Bernoulli/Categorical condition). Its '
      + 'density is tractable, but this binding cannot be sampled in '
      + 'the current scope (inline / non-closed-form selector).'));
  }
  const branchP = branchEntries.map((b: any, bi: any) => {
    if (b && b.ref != null) return ctx.getMeasure(b.ref);
    return collectRefArrays(b.ir, ctx.fixedValues, ctx.getMeasure)
      .then((refArrays: any) => ctx.sendWorker({
        type: 'sampleN', ir: b.ir, count: ctx.sampleCount,
        refArrays: refArrays, seed: nameSeed(name + ':b' + bi, ctx.rootSeed),
      }))
      .then((reply: any) => scalarMeasureN(reply.samples, {
        logWeights: reply.logWeights || null, logTotalmass: 0,
        n_eff: reply.samples.length }));
  });
  const want = branchP.concat([ctx.getMeasure(d.selectorRef)]);
  return Promise.all(want).then((ms) => {
    const sel = empirical.materialiseUniform(ms[ms.length - 1]).samples;
    const branches = ms.slice(0, ms.length - 1)
      .map((m: any) => empirical.materialiseUniform(m).samples);
    const K = branches.length;
    const N = ctx.sampleCount;
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let k;
      if (K === 2) {
        k = sel[i] ? 0 : 1;
      } else {
        const base = (d.selectorBase != null) ? d.selectorBase : 1;
        k = (sel[i] | 0) - base;
      }
      if (k < 0) k = 0; else if (k >= K) k = K - 1;
      out[i] = branches[k][i];
    }
    return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// Top-level dispatch
// =====================================================================

const KIND_HANDLERS = {
  // basic
  alias:        (name: any, d: any, ctx: any) => matAlias(d, ctx),
  sample:       (name: any, d: any, ctx: any) => matSample(name, d, ctx),
  evaluate:     (name: any, d: any, ctx: any) => matEvaluate(d, ctx),
  array:        (name: any, d: any) =>      matArray(d),
  // algebra
  weighted:     (name: any, d: any, ctx: any) => matWeighted(d, ctx),
  normalize:    (name: any, d: any, ctx: any) => matNormalize(d, ctx),
  iid:          (name: any, d: any, ctx: any) => matIid(name, d, ctx),
  tuple:        (name: any, d: any, ctx: any) => matTuple(d, ctx),
  record:       (name: any, d: any, ctx: any) => matRecord(d, ctx),
  superpose:    (name: any, d: any, ctx: any) => matSuperpose(name, d, ctx),
  select:       (name: any, d: any, ctx: any) => matSelect(name, d, ctx),
  // broadcast (mat-broadcast.ts)
  kernelbroadcast: (name: any, d: any, ctx: any) => broadcast.matKernelBroadcast(name, d, ctx),
  // density-related (mat-density.ts)
  bayesupdate:  (name: any, d: any, ctx: any) => density.matBayesupdate(d, ctx),
  logdensityof: (name: any, d: any, ctx: any) => density.matLogdensityof(d, ctx),
  broadcast_logdensity: (name: any, d: any, ctx: any) => density.matBroadcastLogdensity(d, ctx),
  totalmass:    (name: any, d: any, ctx: any) => density.matTotalmass(d, ctx),
  // jointchain/kchain first-class kind
  jointchain:   (name: any, d: any, ctx: any) => density.matJointchain(name, d, ctx),
  // transformations (mat-transformations.ts)
  truncate:     (name: any, d: any, ctx: any) => transforms.matTruncate(d, ctx),
  pushfwd:      (name: any, d: any, ctx: any) => transforms.matPushfwd(name, d, ctx),
  // multivariate distributions (mat-multivariate.ts)
  mvnormal:     (name: any, d: any, ctx: any) => multivariate.matMvNormal(name, d, ctx),
  dirichlet:    (name: any, d: any, ctx: any) => multivariate.matDirichlet(name, d, ctx),
  multinomial:  (name: any, d: any, ctx: any) => multivariate.matMultinomial(name, d, ctx),
  wishart:           (name: any, d: any, ctx: any) => multivariate.matWishart(name, d, ctx),
  inversewishart:    (name: any, d: any, ctx: any) => multivariate.matInverseWishart(name, d, ctx),
  lkjcholesky:       (name: any, d: any, ctx: any) => multivariate.matLKJCholesky(name, d, ctx),
  lkj:               (name: any, d: any, ctx: any) => multivariate.matLKJ(name, d, ctx),
  binnedpoissonprocess: (name: any, d: any, ctx: any) => multivariate.matBinnedPoissonProcess(name, d, ctx),
};

/**
 * Entry point. ctx = {
 *   derivations:  Object,     // name → derivation
 *   bindings:     Map,        // name → binding (lifted)
 *   fixedValues:  Map,        // name → pre-evaluated value
 *   getMeasure:   (name) => Promise<Measure>,    // recursive callback
 *   sendWorker:   (msg)  => Promise<reply>,      // worker handle
 *   sampleCount:  number,                        // global N
 *   rootSeed:     number,                        // base for nameSeed
 * }
 *
 * Returns Promise<Measure>. The caller (typically a viewer cache) is
 * responsible for memoising — this function performs no caching of
 * its own, so the recursion through ctx.getMeasure must short-circuit
 * already-computed measures.
 */
function materialiseMeasure(name: string, ctx: any): Promise<EmpiricalMeasure> {
  // Fixed-phase short-circuit. The orchestrator's pre-eval may have
  // computed a value (a scalar from a deterministic expression, an
  // array from rand, a record from a literal). Synthesize the measure
  // record directly; the worker never has to see fixed-phase values.
  if (ctx.fixedValues && ctx.fixedValues.has(name)) {
    const fxm = fixedValueToMeasure(ctx.fixedValues.get(name), ctx.sampleCount);
    if (fxm) return Promise.resolve(fxm);
    // Opaque fixed value (rngstate) — fall through; if the binding has
    // no derivation either, the next check rejects cleanly.
  }
  const d = ctx.derivations[name];
  if (!d) return Promise.reject(new Error("no derivation for '" + name + "'"));
  const handler = (KIND_HANDLERS as any)[d.kind];
  if (!handler) {
    return Promise.reject(new Error('unknown derivation kind: ' + d.kind));
  }
  return handler(name, d, ctx);
}

/**
 * Materialise a `broadcast(<Dist>, args…)` IR directly, without a
 * binding-graph derivation. The viewer's kernel-sample path produces
 * such IR after `expandMeasureRefsInIR` + `substituteLocals` flatten
 * a kernel body that contained a `~ Dist.(…)` draw — the resulting
 * IR is `record(field = broadcast(<ref Dist>, …))`, and the field
 * value needs an independent-product sample without going through
 * a named binding.
 *
 * The IR shape this accepts matches what `classifyKernelBroadcast`
 * recognises: head is `{kind:'ref', ns:'self', name:<Dist>}` with
 * `<Dist>` in SAMPLEABLE_DISTRIBUTIONS, followed by positional or
 * kwarg parameter sources. Anything else rejects with a clear
 * error.
 *
 * Synthesises a kernelbroadcast derivation from the IR and dispatches
 * to `matKernelBroadcast` — same impl as the binding-graph path, so
 * conformance with `y ~ Normal.(means, sigma)` (when y is a top-
 * level binding) is automatic.
 */
function materialiseKernelBroadcastIR(ir: any, ctx: any) {
  if (!ir || ir.kind !== 'call' || ir.op !== 'broadcast') {
    return Promise.reject(new Error(
      'materialiseKernelBroadcastIR: not a broadcast call'));
  }
  if (!Array.isArray(ir.args) || ir.args.length < 1) {
    return Promise.reject(new Error(
      'materialiseKernelBroadcastIR: broadcast has no head arg'));
  }
  const head = ir.args[0];
  if (!head || head.kind !== 'ref' || head.ns !== 'self') {
    return Promise.reject(new Error(
      'materialiseKernelBroadcastIR: head must be a self-ref to a distribution'));
  }
  const irShared = require('./ir-shared.ts');
  if (!irShared.SAMPLEABLE_DISTRIBUTIONS.has(head.name)) {
    return Promise.reject(new Error(
      "materialiseKernelBroadcastIR: '" + head.name + "' is not a sampleable distribution"));
  }
  const argIRs = ir.args.slice(1);
  const kwargIRs = ir.kwargs ? Object.assign({}, ir.kwargs) : null;
  if (argIRs.length === 0 && (!kwargIRs || Object.keys(kwargIRs).length === 0)) {
    return Promise.reject(new Error(
      'materialiseKernelBroadcastIR: broadcast has no parameter inputs'));
  }
  const d: any = { kind: 'kernelbroadcast', distOp: head.name,
                   argIRs: argIRs, kwargIRs: kwargIRs };
  // Synthetic name for `nameSeed(name, ctx.rootSeed)` inside
  // matKernelBroadcast — not used as a binding lookup. Callers
  // that need a stable distinct sample stream can override
  // `ctx.rootSeed` before calling.
  return broadcast.matKernelBroadcast('%broadcast_ir', d, ctx);
}

module.exports = {
  materialiseMeasure,
  materialiseKernelBroadcastIR,
  // Helpers exposed for the viewer's plot-plan fallbacks (which
  // sometimes need to compose the same primitives outside the kind-
  // dispatch path — e.g., to render a kernel-applied measure that
  // doesn't have a binding-graph derivation).
  fixedValueToMeasure,
  collectRefArrays,
  classifyProfileSelfRefs,
  nameSeed,
  makeMainThreadPrng,
  // Value ↔ Measure bridges.
  valueOf,
  measureFromValue,
};
