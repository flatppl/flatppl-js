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
  inlineCallableRefs,
  measureFromValue,
  measureFromReply,
  scalarMeasureN,
  valueOf,
  isFunctionLikeBinding,
  isCallableLayerBinding,
} = shared;

// =====================================================================
// Per-kind handlers — basic + algebra
// =====================================================================

function matSample(name: string, d: DerivationSample, ctx: any) {
  return collectRefArrays(d.distIR, ctx)
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
  //
  // Inline callable refs before the worker hand-off. The worker's
  // session env doesn't carry `__resolveFnBody`, so a bare `self`-ref
  // to a user-defined `functionof` (e.g. `Y = polyeval.([C], X)`
  // where polyeval is a user fn) would fail `_resolveFn` at the
  // broadcast dispatcher. Splicing the inline `functionof` IR in
  // place lets `_resolveFn`'s inline branch resolve directly. The
  // refs walk below also skips function-like bindings (the inlined
  // body's `%local` placeholder refs aren't `self`-refs, so they
  // don't enter the perAtomNames / fixedEnv pre-fetch).
  const inlinedIR = inlineCallableRefs(d.ir, ctx.bindings);
  return prepareDensityRefs(inlinedIR, ctx, 'matEvaluate').then((prep: any) => {
    const { refArrays, fixedEnv, perAtomNames } = prep;
    return Promise.all(perAtomNames.map(ctx.getMeasure)).then((parentMeasures: any[]) => {
      return pushFixedEnv(ctx, fixedEnv).then(() => ctx.sendWorker({
        type: 'evaluateN',
        ir: inlinedIR,
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
  //
  // Record-typed parent: a record measure has no top-level .samples,
  // just per-field sub-measures (with their own samples + logWeights)
  // and a top-level logWeights array joining the per-field weights.
  // Weighting a record measure shifts the top-level logWeights — the
  // per-field samples and weights stay untouched (the weight is a
  // function of the variate, not of any one field). The constant-
  // shift fast path also leaves the field structure intact and only
  // adjusts the top-level logWeights / logTotalmass. Used by the
  // complement-route restrict expansion `logweighted(logdensityof(
  // marginal, x), kernel(x))`, where `kernel(x)` materialises as a
  // record measure (the kernel body) and the per-atom log-weight is
  // a single scalar from `logdensityof(marginal, x)`.
  return ctx.getMeasure(d.from).then((parent: any) => {
    const isRecord = parent && parent.shape === 'record' && parent.fields;
    const isTuple  = parent && parent.shape === 'tuple'  && parent.elems;
    if (isRecord || isTuple) {
      const N = ctx.sampleCount;
      const baseLW = parent.logWeights
        ? parent.logWeights
        : (() => {
            const c = N > 0 ? -Math.log(N) : 0;
            const a = new Float64Array(N);
            for (let i = 0; i < N; i++) a[i] = c;
            return a;
          })();
      const w = new Float64Array(N);
      const finalise = (logTotalmass: number, weights: Float64Array) => {
        const out = isRecord
          ? Object.assign(empirical.recordMeasure(parent.fields, weights),
              { logTotalmass, n_eff: (typeof parent.n_eff === 'number' ? parent.n_eff : N) })
          : Object.assign(empirical.tupleMeasure(parent.elems, weights),
              { logTotalmass, n_eff: (typeof parent.n_eff === 'number' ? parent.n_eff : N) });
        return out;
      };
      if (d.weightIR) {
        return collectRefArrays(d.weightIR, ctx).then((refArrays: any) =>
          ctx.sendWorker({
            type: 'evaluateN',
            ir: d.weightIR,
            count: N,
            refArrays: refArrays,
          })
        ).then((reply: any) => {
          const weights = reply.samples;
          if (d.isLog) {
            for (let i = 0; i < N; i++) w[i] = baseLW[i] + weights[i];
          } else {
            let nonPos = 0;
            for (let j = 0; j < N; j++) {
              const v = weights[j];
              if (v > 0) w[j] = baseLW[j] + Math.log(v);
              else { w[j] = -Infinity; if (v < 0) nonPos++; }
            }
            if (nonPos > 0) {
              // eslint-disable-next-line no-console
              console.warn('weighted: ' + nonPos
                + ' negative weight sample(s) treated as zero mass');
            }
          }
          return finalise(empirical.logSumExp(w), w);
        });
      }
      // Constant shift fast path.
      for (let i = 0; i < N; i++) w[i] = baseLW[i] + d.logShift;
      const parentLTM = (typeof parent.logTotalmass === 'number') ? parent.logTotalmass : 0;
      return finalise(parentLTM + d.logShift, w);
    }
    const lifted = empirical.materialiseUniform(parent);
    const N = lifted.logWeights.length;
    const w = new Float64Array(N);
    if (d.weightIR) {
      return collectRefArrays(d.weightIR, ctx).then((refArrays: any) =>
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
  // Record/tuple-typed parents: same logic as matWeighted's record
  // branch — the top-level logWeights array carries the joint
  // weighting; per-field samples and weights stay untouched.
  // Used by `normalize(restrict(M, x))` (complement route emits a
  // record-shaped kernel application; selector route emits a
  // bayesupdate of a record-shaped prior).
  return ctx.getMeasure(d.from).then((parent: any) => {
    const isRecord = parent && parent.shape === 'record' && parent.fields;
    const isTuple  = parent && parent.shape === 'tuple'  && parent.elems;
    if (isRecord || isTuple) {
      const N = ctx.sampleCount;
      const baseLW = parent.logWeights
        ? parent.logWeights
        : (() => {
            const c = N > 0 ? -Math.log(N) : 0;
            const a = new Float64Array(N);
            for (let i = 0; i < N; i++) a[i] = c;
            return a;
          })();
      const lse = empirical.logSumExp(baseLW);
      const w = new Float64Array(N);
      for (let i = 0; i < N; i++) w[i] = baseLW[i] - lse;
      const out = isRecord
        ? Object.assign(empirical.recordMeasure(parent.fields, w),
            { logTotalmass: 0,
              n_eff: (typeof parent.n_eff === 'number' ? parent.n_eff : N) })
        : Object.assign(empirical.tupleMeasure(parent.elems, w),
            { logTotalmass: 0,
              n_eff: (typeof parent.n_eff === 'number' ? parent.n_eff : N) });
      return out;
    }
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
  //
  // Resolution: peel through alias / normalize (both sample-
  // preserving — normalize just adjusts logTotalmass, samples
  // unchanged) to reach the underlying sample / truncate derivation,
  // then dispatch to sampleN (vanilla) or truncateSampleN (truncated)
  // with `count = N * k` so iid blocks are produced in one worker
  // round-trip.
  const resolved = _resolveIidLeaf(d.from, ctx.derivations);
  if (!resolved) {
    // Fallback for iid-of-composite-measure (e.g. iid(zib_one, N)
    // where zib_one is a superpose / mixture / pushfwd / etc.):
    // materialise the inner measure BY NAME with an inflated
    // sample count = N × k, then reshape to [N, …dims]. Going by
    // name re-enters the kind-dispatch pipeline so the right
    // kind-specific handler (matSuperpose / matSelect / matPushfwd)
    // runs — including per-atom-weight resolution that the
    // materialiseMeasureIR fast paths can't do.
    //
    // Uses a child ctx with a FRESH cache + inflated sampleCount
    // so the inflated-count materialisation doesn't pollute the
    // parent ctx's cache for the same name at N atoms.
    if (!ctx.derivations || !ctx.derivations[d.from]) {
      return Promise.reject(new Error('iid: cannot resolve leaf sample IR for ' + d.from));
    }
    const k = d.dims.reduce((p: any, n: any) => p * n, 1);
    const N = ctx.sampleCount;
    const inflatedCache = new Map();
    const inflatedCtx: any = Object.assign({}, ctx, {
      sampleCount: N * k,
      rootSeed: nameSeed(name + ':iid_fallback', ctx.rootSeed),
    });
    inflatedCtx.getMeasure = function(nn: string) {
      if (inflatedCache.has(nn)) return inflatedCache.get(nn);
      const p = materialiseMeasure(nn, inflatedCtx);
      inflatedCache.set(nn, p);
      return p;
    };
    return inflatedCtx.getMeasure(d.from).then((innerM: any) => {
      const samples = innerM.samples
        || (innerM.value && innerM.value.data);
      if (!samples) {
        return Promise.reject(new Error(
          'iid: inner measure for ' + d.from + ' produced no samples'));
      }
      const value = { shape: [N | 0].concat(d.dims), data: samples };
      return Object.assign(
        empirical.arrayMeasure(samples, d.dims, null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  }
  const k = d.dims.reduce((p: any, n: any) => p * n, 1);
  const N = ctx.sampleCount;

  if (resolved.kind === 'truncate') {
    // Truncated leaf: route to truncateSampleN with count = N*k.
    // Output is shape=[N, ...dims] atom-major; the worker returns
    // a flat N*k Float64Array which we wrap directly.
    return collectRefArrays(resolved.distIR, ctx)
      .then((refArrays: any) => ctx.sendWorker({
        type: 'truncateSampleN',
        ir: resolved.distIR,
        setDescr: resolved.setDescr,
        count: N * k,
        mode: 'rejection',
        budget: ctx.rejectionBudget != null ? ctx.rejectionBudget : 1000,
        refArrays: refArrays,
        seed: nameSeed(name, ctx.rootSeed),
      }))
      .then((reply: any) => {
        const value = { shape: [N | 0].concat(d.dims), data: reply.samples };
        return Object.assign(
          empirical.arrayMeasure(reply.samples, d.dims, null),
          // logTotalmass scales by k (independent product of k iid
          // truncated draws — each contributes the same logShift).
          { value: value, logTotalmass: k * (reply.logShift || 0), n_eff: N },
        );
      });
  }

  // Standard sample leaf — sampleN with repeat=k.
  return collectRefArrays(resolved.distIR, ctx)
    .then((refArrays: any) => ctx.sendWorker({
      type: 'sampleN', ir: resolved.distIR, count: N, repeat: k,
      refArrays: refArrays,
      seed: nameSeed(name, ctx.rootSeed),
    }))
    .then((reply: any) => {
      const value = { shape: [N | 0].concat(d.dims), data: reply.samples };
      return Object.assign(
        empirical.arrayMeasure(reply.samples, d.dims, null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
}

// Walk the derivation graph to resolve an iid's inner measure into
// (a) a sample leaf — returning the distribution IR — or (b) a
// truncate leaf — returning the underlying dist IR + set descriptor.
// Peels through alias and normalize (both sample-preserving). Returns
// null if the chain bottoms out at something we can't iid (e.g.
// weighted with non-trivial weight, pushfwd, bayesupdate).
function _resolveIidLeaf(
  name: string, derivations: any, visited?: Set<string>,
): { kind: 'sample' | 'truncate'; distIR: any; setDescr?: any } | null {
  visited = visited || new Set();
  if (visited.has(name)) return null;
  visited.add(name);
  const d = derivations[name];
  if (!d) return null;
  if (d.kind === 'alias') return _resolveIidLeaf(d.from, derivations, visited);
  if (d.kind === 'normalize') {
    return _resolveIidLeaf(d.from, derivations, visited);
  }
  if (d.kind === 'sample') {
    return { kind: 'sample', distIR: d.distIR };
  }
  if (d.kind === 'truncate') {
    // Truncate's parent must reduce to a sample leaf — we need the
    // underlying dist IR + the set descriptor.
    const parentLeaf = _resolveIidLeaf(d.from, derivations, visited);
    if (!parentLeaf || parentLeaf.kind !== 'sample') return null;
    return { kind: 'truncate', distIR: parentLeaf.distIR, setDescr: d.setDescr };
  }
  return null;
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
    return collectRefArrays(b.ir, ctx)
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
// ---------------------------------------------------------------------
// Materialiser pipeline (P3b; engine-concepts §18.11 / §20.10.5 item 5)
// ---------------------------------------------------------------------
//
// `materialiseMeasure(name, ctx)` runs through a pipeline of stages.
// Each stage has signature `(name, ctx, next) => Promise<Measure>`
// where `next(name, ctx)` invokes the rest of the pipeline. Stages
// compose via direct function nesting (middleware / MLIR-pass
// pattern). Default pipeline is `[kindDispatchStage]`; orthogonal
// concerns (tracing, MCMC handlers, gradient capture, etc.) get
// added as additional stages without touching per-kind code.
//
// Today's `KIND_HANDLERS` dispatch lives in `kindDispatchStage` —
// the final stage. Earlier stages can wrap, short-circuit, or
// instrument the dispatch. The pipeline is intentionally append-
// extensible via `registerMaterialiserStage(stage)`; the default
// pipeline ships with only the kindDispatch stage so backward
// compat is exact.
//
// Engine-concepts §18.11 / §20.10.5 item 5: this is the "not-
// Messenger" version of the effect-handler pattern — a static
// pipeline over IR-walking requests, not a runtime handler stack
// over message dicts.

interface MaterialiserStage {
  (name: string, ctx: any, next: (name: string, ctx: any) => Promise<EmpiricalMeasure>):
    Promise<EmpiricalMeasure>;
  // Optional label for debug / instrumentation.
  label?: string;
}

// Default pipeline. New stages get prepended via
// `registerMaterialiserStage(stage)` so they wrap the existing
// dispatch (caller-side first, dispatch-side last). The terminal
// stage (kindDispatchStage) runs the actual KIND_HANDLERS lookup.
const _MATERIALISER_PIPELINE: MaterialiserStage[] = [];

function registerMaterialiserStage(stage: MaterialiserStage): void {
  if (typeof stage !== 'function') {
    throw new Error('registerMaterialiserStage: stage must be a function');
  }
  _MATERIALISER_PIPELINE.push(stage);
}

// Reset to default state — clears stages then reinstalls the
// default pre-dispatch guards (callable-layer rejection + fixed-
// phase short-circuit). Used in tests that register transient
// stages and want to restore working production state.
function _resetMaterialiserPipeline(): void {
  _MATERIALISER_PIPELINE.length = 0;
  // Defaults are installed once at module load by
  // _installDefaultMaterialiserStages; re-install them here so
  // post-reset state is operational, not bare.
  if (typeof _installDefaultMaterialiserStages === 'function') {
    _installDefaultMaterialiserStages();
  }
}

// Clear ALL stages, including the default pre-dispatch guards.
// Tests that want a truly empty pipeline (and will manually drive
// `_runPipeline` with their own terminal stage) use this. Production
// code should never call this.
function _clearMaterialiserPipeline(): void {
  _MATERIALISER_PIPELINE.length = 0;
}

// The terminal stage: existing KIND_HANDLERS dispatch. Receives
// `next` for symmetry but never calls it — it IS the final step.
const kindDispatchStage: MaterialiserStage = function (name, ctx, _next) {
  const d = ctx.derivations[name];
  if (!d) return Promise.reject(new Error("no derivation for '" + name + "'"));
  const handler = (KIND_HANDLERS as any)[d.kind];
  if (!handler) {
    return Promise.reject(new Error('unknown derivation kind: ' + d.kind));
  }
  return handler(name, d, ctx);
};
kindDispatchStage.label = 'kindDispatch';

// Pre-dispatch stages live in _MATERIALISER_PIPELINE; the terminal
// is kindDispatchStage. Compose them right-to-left so the pipeline
// reads outer-to-inner: stage[0] wraps stage[1] wraps … wraps
// kindDispatch.
function _runPipeline(name: string, ctx: any): Promise<EmpiricalMeasure> {
  let chain: (n: string, c: any) => Promise<EmpiricalMeasure> =
    (n, c) => kindDispatchStage(n, c, () => {
      throw new Error('materialiser: terminal stage cannot call next');
    });
  // Build the chain inside-out: start from kindDispatch (terminal),
  // wrap each registered stage AROUND it.
  for (let i = _MATERIALISER_PIPELINE.length - 1; i >= 0; i--) {
    const stage = _MATERIALISER_PIPELINE[i];
    const inner = chain;
    chain = (n, c) => stage(n, c, inner);
  }
  return chain(name, ctx);
}

// Callable-layer rejection stage (engine-concepts §19.2 + §19.5).
// Bindings in the function or kernel layer (functionof / kernelof /
// fn / bijection / fchain — and, when its inferredType is
// kernelType, a kernel-first jointchain/kchain) are not
// materialisable as a closed measure. They're consulted by name at
// density / sample dispatch. The gate consolidates what used to be
// per-handler defensive throws into one diagnostic at the pipeline
// front.
const callableGuardStage: MaterialiserStage = function (name, ctx, next) {
  const binding = ctx.bindings && ctx.bindings.get(name);
  const isCallableByTag = binding && isFunctionLikeBinding(binding);
  const isCallableByType = binding && isCallableLayerBinding(binding);
  if (isCallableByTag || isCallableByType) {
    return Promise.reject(new Error(
      "callable-layer binding '" + name + "'"
      + (binding && binding.type ? " (type '" + binding.type + "')" : '')
      + " is not materialisable as a closed measure (engine-concepts §19); "
      + "it is consulted by name at density / sample dispatch."));
  }
  return next(name, ctx);
};
callableGuardStage.label = 'callableGuard';

// Fixed-phase short-circuit stage. The orchestrator's pre-eval may
// have computed a value (a scalar from a deterministic expression,
// an array from rand, a record from a literal). Synthesise the
// measure record directly; the worker never has to see fixed-phase
// values. Opaque fixed values (rngstate) fall through to the next
// stage — if the binding has no derivation either, the kindDispatch
// stage rejects cleanly.
const fixedPhaseStage: MaterialiserStage = function (name, ctx, next) {
  if (ctx.fixedValues && ctx.fixedValues.has(name)) {
    const fxm = fixedValueToMeasure(ctx.fixedValues.get(name), ctx.sampleCount);
    if (fxm) return Promise.resolve(fxm);
  }
  return next(name, ctx);
};
fixedPhaseStage.label = 'fixedPhaseShortCircuit';

// Default pipeline ships with the two pre-dispatch stages above,
// in callable-guard → fixed-phase → kindDispatch order. Tests that
// need a clean slate call `_resetMaterialiserPipeline()` first, then
// re-register if they want the defaults back via
// `_installDefaultMaterialiserStages()`.
function _installDefaultMaterialiserStages(): void {
  registerMaterialiserStage(callableGuardStage);
  registerMaterialiserStage(fixedPhaseStage);
}
_installDefaultMaterialiserStages();

// ---------------------------------------------------------------------
// Tracing stage example (P3b observability use case)
// ---------------------------------------------------------------------
//
// `makeTracingStage(opts?)` returns a pipeline stage that records
// per-binding materialise timings into a buffer (or a user-supplied
// callback). Useful for performance profiling: how long does each
// derivation kind take, which bindings are bottlenecks. The stage
// forwards transparently — adding it to the pipeline doesn't change
// any materialise result, only adds an observation side-channel.
//
// Usage:
//   const buffer: TraceEntry[] = [];
//   const stage = makeTracingStage({ buffer });
//   registerMaterialiserStage(stage);
//   // ... drive materialise calls ...
//   console.log(buffer);   // [{name, durationMs, kind?, ok}, …]
//
// Or for streaming:
//   const stage = makeTracingStage({
//     onEvent: (e) => myMetricsBackend.emit(e),
//   });
//
// The stage is added OUTSIDE the default guards so it observes the
// guards' decisions too (a callable-layer rejection produces a
// trace entry with `ok: false`).

interface TraceEntry {
  name: string;
  durationMs: number;
  kind?: string;       // derivation kind, when known
  ok: boolean;         // true on resolved measure; false on reject
  error?: string;      // error message when ok=false
}

interface TracingStageOpts {
  // Optional buffer to push entries to. Mutated in place.
  buffer?: TraceEntry[];
  // Optional callback invoked per entry. When both buffer and
  // onEvent are set, both fire.
  onEvent?: (entry: TraceEntry) => void;
  // Optional clock; defaults to performance.now() in environments
  // that have it, Date.now() otherwise. Tests pass a deterministic
  // clock to pin timing semantics.
  now?: () => number;
}

function makeTracingStage(opts?: TracingStageOpts): MaterialiserStage {
  const o = opts || {};
  const buf = o.buffer;
  const onEvent = o.onEvent;
  const now = o.now || (typeof performance !== 'undefined' && performance.now
    ? () => performance.now()
    : () => Date.now());
  const stage: MaterialiserStage = function (name, ctx, next) {
    const t0 = now();
    // Derivation kind is helpful for grouping; available pre-dispatch.
    const kind = ctx.derivations && ctx.derivations[name]
      && ctx.derivations[name].kind;
    return next(name, ctx).then(
      (m) => {
        const entry: TraceEntry = {
          name,
          durationMs: now() - t0,
          ok: true,
        };
        if (kind !== undefined) entry.kind = kind;
        if (buf) buf.push(entry);
        if (onEvent) onEvent(entry);
        return m;
      },
      (err) => {
        const entry: TraceEntry = {
          name,
          durationMs: now() - t0,
          ok: false,
          error: err && err.message ? err.message : String(err),
        };
        if (kind !== undefined) entry.kind = kind;
        if (buf) buf.push(entry);
        if (onEvent) onEvent(entry);
        throw err;
      },
    );
  };
  stage.label = 'tracing';
  return stage;
}

function materialiseMeasure(name: string, ctx: any): Promise<EmpiricalMeasure> {
  // Pre-dispatch guards (callable-layer rejection, fixed-phase
  // short-circuit) plus the terminal kindDispatch all live in the
  // pipeline now. Each concern composes orthogonally; new concerns
  // (tracing, MCMC handlers, gradient capture) add stages without
  // touching this function.
  return _runPipeline(name, ctx);
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

/**
 * IR-direct sampling of a `select(branches=[weighted(w_i, M_i)], …)`
 * shape (engine-concepts §11). Used by the viewer's kernel-plot path
 * (`plot-plan.materialiseConcreteMeasure`) when a kernel body's
 * realised measure contains a superpose / mixture / ifelse — the
 * lift form is a `select` IR with weighted branches but no
 * binding-graph derivation (the kernel body is sampled after
 * placeholder substitution; matSelect's derivation-keyed path
 * doesn't fit).
 *
 * Implementation mirrors matSelect's eval-all-branches-then-gather:
 *   - Fold per-branch weights to JS numbers (lits + arith over
 *     lits + Math consts).
 *   - Build a Bernoulli (K=2) / Categorical (K≥3) selector IR with
 *     the normalised probs.
 *   - Sample selector + each branch independently over N atoms.
 *   - Gather per atom from the chosen branch.
 *
 * Limitation: weights must reduce to numbers. Variable-weight
 * mixtures (e.g. `weighted(w_atom, …)` where w_atom is a per-atom
 * ref) need either a derivation-keyed materialiser (matSelect's
 * runtimeWeights branch) or an extension here. Out of scope today.
 */
function materialiseSelectIR(ir: any, ctx: any) {
  if (!ir || ir.kind !== 'call' || ir.op !== 'select') {
    return Promise.reject(new Error('materialiseSelectIR: not a select call'));
  }
  const branches = ir.branches;
  if (!Array.isArray(branches) || branches.length === 0) {
    return Promise.reject(new Error('materialiseSelectIR: select has no branches'));
  }
  const K = branches.length;
  const N = ctx.sampleCount;
  const branchPairs: Array<{ weight: number; measureIR: any }> = [];
  for (const b of branches) {
    if (!b || b.kind !== 'call' || b.op !== 'weighted'
        || !Array.isArray(b.args) || b.args.length !== 2) {
      return Promise.reject(new Error(
        'materialiseSelectIR: select branch must be weighted(w, m); got '
        + JSON.stringify(b).slice(0, 80)));
    }
    const w = _foldNumericIR(b.args[0]);
    if (w == null) {
      return Promise.reject(new Error(
        'materialiseSelectIR: select branch weight must reduce to a '
        + 'number (literal or arith over literals); got '
        + JSON.stringify(b.args[0]).slice(0, 100)));
    }
    branchPairs.push({ weight: w, measureIR: b.args[1] });
  }
  const wSum = branchPairs.reduce((a, p) => a + p.weight, 0);
  if (!(wSum > 0)) {
    return Promise.reject(new Error('materialiseSelectIR: weights sum to non-positive'));
  }
  const probs = branchPairs.map((p) => p.weight / wSum);
  // Selector IR. K=2 → Bernoulli (1 → branch 0, 0 → branch 1,
  // matching matSelect's gather convention). K≥3 → Categorical
  // (1-based per engine convention; subtract 1 at gather time).
  let selectorIR: any;
  if (K === 2) {
    selectorIR = { kind: 'call', op: 'Bernoulli',
      kwargs: { p: { kind: 'lit', value: probs[0], numType: 'real' } } };
  } else {
    selectorIR = { kind: 'call', op: 'Categorical',
      kwargs: { p: { kind: 'call', op: 'vector',
        args: probs.map((pp) => ({ kind: 'lit', value: pp, numType: 'real' })) } } };
  }
  const selectorSeed = nameSeed('%select_ir:selector', ctx.rootSeed);
  const selectorP = ctx.sendWorker({
    type: 'sampleN', ir: selectorIR, count: N, seed: selectorSeed,
  });
  // Sample each branch's measure. The measureIR can itself be any
  // measure shape — recurse via the viewer's materialiseConcrete-
  // Measure (which routes back here for nested selects). To keep
  // this engine helper self-contained AND avoid a viewer↔engine
  // cycle, we require branches to be either:
  //   - a leaf distribution (sampleN-able directly), OR
  //   - a closed-measure ref the caller's ctx.getMeasure resolves.
  // For richer nested shapes (iid inside select etc.) the caller's
  // materialiseConcreteMeasure should peel structure FIRST and
  // call materialiseSelectIR once it reaches a select node.
  const branchPs = branchPairs.map((p, bi) => {
    const inner = p.measureIR;
    if (inner && inner.kind === 'ref' && inner.ns === 'self') {
      return ctx.getMeasure(inner.name);
    }
    // Leaf-distribution branch: collectRefArrays + worker sampleN
    // (the engine's standard sample path).
    return shared.collectRefArrays(inner, ctx)
      .then((refArrays: any) => ctx.sendWorker({
        type: 'sampleN', ir: inner, count: N, refArrays: refArrays,
        seed: nameSeed('%select_ir:b' + bi, ctx.rootSeed),
      }))
      .then((reply: any) => scalarMeasureN(reply.samples, {
        logWeights: reply.logWeights || null,
        logTotalmass: 0, n_eff: reply.samples.length,
      }));
  });
  return Promise.all(([selectorP] as any[]).concat(branchPs)).then((results: any[]) => {
    const sel = empirical.materialiseUniform(results[0]).samples
      || results[0].samples;
    const branchMs = results.slice(1).map((m: any) =>
      empirical.materialiseUniform(m).samples || m.samples);
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let k;
      if (K === 2) {
        k = sel[i] ? 0 : 1;
      } else {
        k = (sel[i] | 0) - 1;
      }
      if (k < 0) k = 0; else if (k >= K) k = K - 1;
      out[i] = branchMs[k][i];
    }
    return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// Fold a numeric IR expression (literals + arithmetic ops over
// literals + named constants) to a JS number. Returns null when
// the expression isn't fully constant — viewer-side
// substituteLocals inlines `%local` refs to lits but doesn't
// constant-fold downstream arith.
function _foldNumericIR(ir: any): number | null {
  if (!ir || typeof ir !== 'object') return null;
  if (ir.kind === 'lit' && typeof ir.value === 'number') return ir.value;
  if (ir.kind === 'const') {
    if (ir.name === 'pi') return Math.PI;
    if (ir.name === 'inf') return Infinity;
    return null;
  }
  if (ir.kind !== 'call' || !Array.isArray(ir.args)) return null;
  const args = ir.args.map(_foldNumericIR);
  for (const a of args) if (a == null) return null;
  const op = ir.op;
  if (args.length === 2) {
    const a = args[0] as number, b = args[1] as number;
    if (op === 'add') return a + b;
    if (op === 'sub') return a - b;
    if (op === 'mul') return a * b;
    if (op === 'div' || op === 'divide') return a / b;
    if (op === 'pow') return Math.pow(a, b);
    if (op === 'mod') return a % b;
  }
  if (args.length === 1) {
    const a = args[0] as number;
    if (op === 'neg') return -a;
    if (op === 'pos') return +a;
    if (op === 'exp')  return Math.exp(a);
    if (op === 'log')  return Math.log(a);
    if (op === 'sqrt') return Math.sqrt(a);
    if (op === 'abs')  return Math.abs(a);
  }
  return null;
}

/**
 * IR-direct measure materialisation — single engine entry for
 * callers that have a CONCRETE (closed, no-refs) measure IR and
 * need samples back. The viewer's kernel-plot path uses this
 * after substituting placeholder values into a kernel body's
 * IR: the result is a closed measure IR that needs sampling
 * without going through a binding-graph derivation.
 *
 * Dispatches on `ir.op`:
 *   - `lawof(M)`         → peel, recurse on M.
 *   - `iid(M, n…)`       → sampleN with `repeat=k` for leaf-dist
 *                          M, recurse for nested.
 *   - `joint`/`record`   → per-field materialise, combine via
 *                          `empirical.recordMeasure`.
 *   - `broadcast(<Dist>, args…)` → `materialiseKernelBroadcastIR`.
 *   - `select(branches=…)`       → `materialiseSelectIR`.
 *   - leaf distribution → `collectRefArrays` + worker `sampleN`.
 *
 * Single source of truth for IR-direct measure dispatch — viewer
 * helpers like `plot-plan.materialiseConcreteMeasure` delegate
 * here so per-op routing logic lives in the engine, not duplicated
 * across consumers. (Per TODO-flatppl-js: "Refactor
 * `materialiseConcreteMeasure` to delegate to engine KIND_HANDLERS"
 * — this entry IS that delegation point for the IR-direct path.)
 */
function materialiseMeasureIR(ir: any, ctx: any): Promise<any> {
  if (!ir) return Promise.reject(new Error('materialiseMeasureIR: null IR'));
  if (ir.kind !== 'call') {
    return Promise.reject(new Error(
      "materialiseMeasureIR: non-call IR (kind '" + ir.kind + "')"));
  }
  // lawof(M) → peel, recurse on M.
  if (ir.op === 'lawof' && Array.isArray(ir.args) && ir.args.length === 1) {
    return materialiseMeasureIR(ir.args[0], ctx);
  }
  // iid(M, n…) — sample-major output [N, ...dims].
  if (ir.op === 'iid' && Array.isArray(ir.args) && ir.args.length >= 2) {
    const inner = ir.args[0];
    const dims: number[] = [];
    for (let di = 1; di < ir.args.length; di++) {
      const d = ir.args[di];
      if (!d || d.kind !== 'lit' || !Number.isInteger(d.value)) {
        return Promise.reject(new Error(
          'materialiseMeasureIR: iid dim must be integer literal'));
      }
      dims.push(d.value);
    }
    const k = dims.reduce((p: number, n: number) => p * n, 1);
    const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;
    if (inner && inner.kind === 'call' && SAMPLEABLE && SAMPLEABLE.has(inner.op)) {
      return shared.collectRefArrays(inner, ctx).then((refArrays: any) =>
        ctx.sendWorker({
          type: 'sampleN', ir: inner, count: ctx.sampleCount, repeat: k,
          refArrays: refArrays,
          seed: nameSeed('%materialise_ir:iid', ctx.rootSeed),
        })
      ).then((reply: any) => empirical.arrayMeasure(reply.samples, dims, null));
    }
    // Nested inner — recurse with count*k, then reshape.
    const innerCtx = Object.assign({}, ctx, { sampleCount: ctx.sampleCount * k });
    return materialiseMeasureIR(inner, innerCtx).then((innerM: any) =>
      empirical.arrayMeasure(innerM.samples, dims, null));
  }
  // joint / record → field-wise materialise + recordMeasure.
  if ((ir.op === 'joint' || ir.op === 'record') && Array.isArray(ir.fields)) {
    const fieldNames = ir.fields.map((f: any) => f.name);
    const fieldIRs   = ir.fields.map((f: any) => f.value);
    return Promise.all(fieldIRs.map((v: any, i: number) => {
      const subCtx = Object.assign({}, ctx, {
        rootSeed: ctx.rootSeed != null
          ? (ctx.rootSeed ^ ((i + 1) * 0x9e3779b1)) : ctx.rootSeed,
      });
      return materialiseMeasureIR(v, subCtx);
    })).then((subs: any[]) => {
      const fields: Record<string, any> = {};
      for (let i = 0; i < fieldNames.length; i++) fields[fieldNames[i]] = subs[i];
      return empirical.recordMeasure(fields, null);
    });
  }
  // broadcast(<Dist>, args…) → engine kernel-broadcast IR-direct entry.
  if (ir.op === 'broadcast' && Array.isArray(ir.args) && ir.args.length >= 1) {
    const head = ir.args[0];
    const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;
    if (head && head.kind === 'ref' && head.ns === 'self'
        && SAMPLEABLE && SAMPLEABLE.has(head.name)) {
      return materialiseKernelBroadcastIR(ir, ctx);
    }
  }
  // select(branches=…) → engine select IR-direct entry.
  if (ir.op === 'select' && Array.isArray(ir.branches) && ir.branches.length >= 1) {
    return materialiseSelectIR(ir, ctx);
  }
  // Leaf distribution (or any unrecognised call op — the worker's
  // sampleN throws if the op isn't in the sampler REGISTRY,
  // surfacing a clear diagnostic).
  return shared.collectRefArrays(ir, ctx).then((refArrays: any) =>
    ctx.sendWorker({
      type: 'sampleN', ir: ir, count: ctx.sampleCount,
      refArrays: refArrays,
      seed: nameSeed('%materialise_ir:leaf', ctx.rootSeed),
    })
  ).then((reply: any) => {
    const data = reply.samples;
    return {
      samples: data,
      value: { shape: [data.length], data: data },
      logWeights: reply.logWeights || null,
    };
  });
}

module.exports = {
  materialiseMeasure,
  materialiseKernelBroadcastIR,
  materialiseSelectIR,
  materialiseMeasureIR,
  // P3b — materialiser pipeline (engine-concepts §18.11 / §20.10.5
  // item 5). Stages get prepended to the default pipeline via
  // `registerMaterialiserStage(stage)`; they wrap the terminal
  // kindDispatch stage. Stages call `next(name, ctx)` to forward.
  registerMaterialiserStage,
  _resetMaterialiserPipeline,
  _clearMaterialiserPipeline,
  _installDefaultMaterialiserStages,
  kindDispatchStage,
  callableGuardStage,
  fixedPhaseStage,
  makeTracingStage,
  _runPipeline,
  // Helpers exposed for the viewer's plot-plan fallbacks (which
  // sometimes need to compose the same primitives outside the kind-
  // dispatch path — e.g., to render a kernel-applied measure that
  // doesn't have a binding-graph derivation).
  fixedValueToMeasure,
  collectRefArrays,
  classifyProfileSelfRefs,
  // Callable-layer predicates (engine-concepts §19.2). Exposed for
  // viewer-side filters that need to skip function-like / kernel-
  // typed source bindings before pulling samples[0] (the same array-
  // collapse failure mode fixed in render-profile.ts; flatppl-js
  // commit e9984f3).
  isFunctionLikeBinding,
  isCallableLayerBinding,
  nameSeed,
  makeMainThreadPrng,
  // Value ↔ Measure bridges.
  valueOf,
  measureFromValue,
};
