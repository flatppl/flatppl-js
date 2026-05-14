'use strict';

// Per-binding measure materialisation.
//
// Given a binding name and a context bundle (derivations, bindings,
// fixedValues, a recursive getMeasure callback, a worker handle, the
// global sample count, and a root seed for per-binding RNG streams),
// materialiseMeasure(name, ctx) returns a Promise<Measure> describing
// the binding's empirical distribution.
//
// The Measure record is the EmpiricalMeasure shape from empirical.js
// extended with two scalar metadata fields:
//
//   { samples:      Float64Array,             // per-atom values
//     logWeights:   Float64Array | null,      // null = uniform 1/N
//     logTotalmass: number,                   // default 0 (= totalmass 1)
//     n_eff:        number,                   // default = samples.length
//     fields?:      { name → Measure },       // record/joint shape
//     elems?:       Measure[],                // tuple shape
//     shape?:       'record' | 'tuple' | 'array',
//     ... }
//
// `logTotalmass` is on the log scale so deep compositions (iid^n,
// chain of weighted, …) stay representable when raw totalmass would
// overflow. `n_eff` is an effective-sample-size estimate that
// shrinks under reweighting / rejection / mixture-resampling; the
// default `samples.length` means "all atoms equally informative".
//
// The materialiser lives in the engine (not the viewer) because the
// per-kind dispatch is pure engine knowledge: given the derivation
// graph, the parents' materialised measures, and a primitive
// (worker.sampleN / evaluateN / logDensityN), how do we produce the
// derived measure? Viewer-specific concerns (caching, render flush,
// UI hooks) stay on the viewer side; the viewer's getMeasure shrinks
// to a thin wrapper over this function plus its own cache.
//
// `getMeasure` and `sendWorker` are injected via ctx rather than
// imported because each host (viewer, future Node CLI, integration
// tests) owns its own cache and worker dispatch strategy. The
// materialiser stays pure-async and side-effect-free apart from
// what those callbacks do.

const empirical    = require('./empirical');
const orchestrator = require('./orchestrator');
const rng          = require('./rng');

// =====================================================================
// Helpers
// =====================================================================

/**
 * FNV-1a 32-bit hash of `name` XOR'd against rootSeed. Gives each
 * binding its own deterministic RNG stream for sampleN — order-
 * independent so two unrelated bindings stay independent regardless
 * of which the user materialises first.
 */
function nameSeed(name, rootSeed) {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h = h ^ name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h ^ (rootSeed | 0)) >>> 0;
}

/**
 * Wrap a Philox state in a closure returning U(0,1) uniforms — the
 * shape empirical.systematicResample / multinomialResample expect.
 * One closure per call; the state mutates internally.
 */
function makeMainThreadPrng(seed) {
  let state = rng.stateFromKey(seed);
  return function () {
    const pair = rng.nextUniform(state);
    state = pair[1];
    return pair[0];
  };
}

/**
 * Resolve every value-position self-ref in `ir` to its parent's
 * Float64Array of samples, returning a refName→samples map for the
 * worker primitives' refArrays. Fixed-phase refs are dropped: they
 * flow through the worker's session env, NOT refArrays (which would
 * try to slice them per-atom and feed the worker an undefined entry).
 */
function collectRefArrays(ir, fixedValues, getMeasure) {
  const refs = orchestrator.collectSelfRefs(ir);
  const names = [];
  refs.forEach((n) => {
    if (fixedValues && fixedValues.has(n)) return;
    names.push(n);
  });
  return Promise.all(names.map(getMeasure)).then((measures) => {
    const out = {};
    for (let i = 0; i < names.length; i++) out[names[i]] = measures[i].samples;
    return out;
  });
}

/**
 * Convert a JS value (from the orchestrator's fixedValues map) to a
 * Measure record. Scalars broadcast across N atoms; numeric arrays
 * are flat samples; records and tuples recurse field-/element-wise.
 * Returns null for shapes we can't surface (rngstate, opaque
 * objects); the caller falls through to derivation-based dispatch.
 */
function fixedValueToMeasure(v, sampleCount) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    const arr = new Float64Array(sampleCount);
    arr.fill(v);
    return { samples: arr, logWeights: null,
             logTotalmass: 0, n_eff: sampleCount };
  }
  if (v instanceof Float64Array || v instanceof Int32Array || v instanceof Uint8Array) {
    const samples = Float64Array.from(v);
    return { samples: samples, logWeights: null,
             logTotalmass: 0, n_eff: samples.length };
  }
  if (Array.isArray(v)) {
    // Plain JS array: flat numeric vector OR mixed-shape tuple.
    let allNum = v.length > 0;
    for (let i = 0; allNum && i < v.length; i++) {
      if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) allNum = false;
    }
    if (allNum) {
      const samples = Float64Array.from(v);
      return { samples: samples, logWeights: null,
               logTotalmass: 0, n_eff: samples.length };
    }
    const elems = new Array(v.length);
    for (let ei = 0; ei < v.length; ei++) elems[ei] = fixedValueToMeasure(v[ei], sampleCount);
    return { elems: elems, logTotalmass: 0, n_eff: sampleCount };
  }
  if (v && typeof v === 'object') {
    if (v.key && Array.isArray(v.key) && v.counter) return null;   // rngstate
    const fields = {};
    let anyOk = false;
    for (const k in v) {
      if (!Object.prototype.hasOwnProperty.call(v, k)) continue;
      const sub = fixedValueToMeasure(v[k], sampleCount);
      if (sub) { fields[k] = sub; anyOk = true; }
    }
    if (anyOk) return { fields: fields, logTotalmass: 0, n_eff: sampleCount };
  }
  return null;
}

/**
 * Top-level N for any Measure record, regardless of shape — top-level
 * `samples` for scalar measures, the first field's length for records,
 * the first element's length for tuples, 0 for empty. Used by per-kind
 * handlers that need to allocate a sibling array of the same length.
 */
function measureN(m) {
  if (!m) return 0;
  if (m.samples) return m.samples.length;
  if (m.fields) {
    const k = Object.keys(m.fields)[0];
    if (k != null) return measureN(m.fields[k]);
  }
  if (m.elems && m.elems.length > 0) return measureN(m.elems[0]);
  return 0;
}

// =====================================================================
// Per-kind handlers
// =====================================================================

function matSample(name, d, ctx) {
  return collectRefArrays(d.distIR, ctx.fixedValues, ctx.getMeasure)
    .then((refArrays) => ctx.sendWorker({
      type: 'sampleN',
      ir: d.distIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      seed: nameSeed(name, ctx.rootSeed),
    }))
    .then((reply) => {
      // Worker reply: samples + logWeights (logWeights typically null
      // from a bare leaf-distribution sample). A leaf distribution is
      // a normalized probability measure, so logTotalmass = 0.
      const lw = reply.logWeights || null;
      return {
        samples: reply.samples,
        logWeights: lw,
        logTotalmass: 0,
        n_eff: reply.samples.length,
      };
    });
}

function matAlias(d, ctx) {
  // Alias: same measure record — reference equality is intentional so
  // click-flipping between a variate and its measure is free, and the
  // shared logWeights ref preserves propagateLogWeights's dedupe contract.
  return ctx.getMeasure(d.from);
}

function matEvaluate(d, ctx) {
  // Deterministic transform of variates. Per-atom: c_i = f(parents_i).
  // logWeights propagate via joint IS (sum independent events, dedupe
  // shared via reference identity). logTotalmass follows from the
  // resulting logWeights; n_eff is min(parents') as a fast bound.
  const refs = orchestrator.collectSelfRefs(d.ir);
  const parentNames = [];
  refs.forEach((n) => {
    if (ctx.fixedValues && ctx.fixedValues.has(n)) return;
    parentNames.push(n);
  });
  return Promise.all(parentNames.map(ctx.getMeasure)).then((parentMeasures) => {
    const refArrays = {};
    for (let i = 0; i < parentNames.length; i++) {
      refArrays[parentNames[i]] = parentMeasures[i].samples;
    }
    return ctx.sendWorker({
      type: 'evaluateN',
      ir: d.ir,
      count: ctx.sampleCount,
      refArrays: refArrays,
    }).then((reply) => {
      const lw = empirical.propagateLogWeights(parentMeasures);
      let n_eff = ctx.sampleCount;
      for (const p of parentMeasures) {
        if (typeof p.n_eff === 'number') n_eff = Math.min(n_eff, p.n_eff);
      }
      // logTotalmass: if the result has weights, it's the log-sum-exp;
      // otherwise (uniform) it's 0. The default `1` total mass is the
      // right answer for a deterministic transform of normalized
      // probability variates; weighted inputs propagate their mass.
      const logTotalmass = lw ? empirical.logSumExp(lw) : 0;
      return {
        samples: reply.samples,
        logWeights: lw,
        logTotalmass: logTotalmass,
        n_eff: n_eff,
      };
    });
  });
}

function matArray(d) {
  // Static array literal — values verbatim, no sampling, no worker
  // round-trip. Length equals the array's literal length, NOT the
  // sample count; downstream plot dispatches by mode for this kind.
  const samples = Float64Array.from(d.values);
  return Promise.resolve({
    samples: samples,
    logWeights: null,
    logTotalmass: 0,
    n_eff: samples.length,
  });
}

function matWeighted(d, ctx) {
  // weighted(w, base) / logweighted(lw, base): shift each parent
  // atom's logWeight by log(w_i) (or lw_i directly). totalmass scales
  // by the average weight; the empirical log-total-mass is
  // logSumExp(resulting logWeights).
  return ctx.getMeasure(d.from).then((parent) => {
    const lifted = empirical.materialiseUniform(parent);
    const N = lifted.logWeights.length;
    const w = new Float64Array(N);
    if (d.weightIR) {
      return collectRefArrays(d.weightIR, ctx.fixedValues, ctx.getMeasure).then((refArrays) =>
        ctx.sendWorker({
          type: 'evaluateN',
          ir: d.weightIR,
          count: ctx.sampleCount,
          refArrays: refArrays,
        })
      ).then((reply) => {
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
        return { samples: lifted.samples, logWeights: w,
                 logTotalmass: lTM, n_eff: nEff };
      });
    }
    // Constant fast path: orchestrator pre-computed d.logShift (a
    // uniform per-atom additive shift). totalmass simply scales.
    for (let i = 0; i < N; i++) w[i] = lifted.logWeights[i] + d.logShift;
    const parentLTM = (typeof parent.logTotalmass === 'number') ? parent.logTotalmass : 0;
    return {
      samples: lifted.samples,
      logWeights: w,
      logTotalmass: parentLTM + d.logShift,
      // Uniform constant-shift doesn't change relative weights → n_eff unchanged.
      n_eff: (typeof parent.n_eff === 'number') ? parent.n_eff : N,
    };
  });
}

function matNormalize(d, ctx) {
  // normalize(base): shift weights so they sum to 1 (logTotalmass = 0).
  return ctx.getMeasure(d.from).then((parent) => {
    const lifted = empirical.materialiseUniform(parent);
    const N = lifted.logWeights.length;
    const lse = empirical.logSumExp(lifted.logWeights);
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) w[i] = lifted.logWeights[i] - lse;
    const nEff = empirical.effectiveSampleSize({ samples: lifted.samples, logWeights: w });
    return { samples: lifted.samples, logWeights: w,
             logTotalmass: 0, n_eff: nEff };
  });
}

function matIid(name, d, ctx) {
  // iid(M, n, …): N atoms × k inner draws, atom-major packed into
  // one Float64Array. The worker's sampleN takes an optional repeat=k.
  // Parameters are pinned per atom (refArrays), so iid samples within
  // atom i share atom-i's parameter context.
  //
  // totalmass: M^k (in log: k · M.logTotalmass).
  // n_eff: inherits M's n_eff (the iid repeats are within-atom; the
  //   atom axis is what carries effective sample size).
  const distIR = orchestrator.leafSampleIR(d.from, ctx.derivations);
  if (!distIR) {
    return Promise.reject(new Error('iid: cannot resolve leaf sample IR for ' + d.from));
  }
  const k = d.dims.reduce((p, n) => p * n, 1);
  return collectRefArrays(distIR, ctx.fixedValues, ctx.getMeasure)
    .then((refArrays) => ctx.sendWorker({
      type: 'sampleN', ir: distIR, count: ctx.sampleCount, repeat: k,
      refArrays: refArrays,
      seed: nameSeed(name, ctx.rootSeed),
    }))
    .then((reply) => {
      // We could fetch the inner measure for n_eff / logTotalmass,
      // but for a leaf-distribution iid those defaults are 1 and N,
      // matching the leaf-sample handler. Stay simple.
      return Object.assign(
        empirical.arrayMeasure(reply.samples, d.dims, null),
        { logTotalmass: 0, n_eff: ctx.sampleCount },
      );
    });
}

function matTuple(d, ctx) {
  // Positional analogue of record. Each element materialises
  // independently; combine into a tuple Measure whose components live
  // in elems. Top-level logWeights is the join of components'
  // (propagateLogWeights handles dedupe + sum).
  return Promise.all(d.elems.map(ctx.getMeasure)).then((subs) => {
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

function matRecord(d, ctx) {
  // Multivariate (record / joint): each field's source binding gets
  // materialised; assembled into a record-shaped Measure (SoA — one
  // sub-measure per field). Top-level logWeights is the join across
  // fields; logTotalmass is the sum of fields' (independent product
  // measures multiply masses).
  const fieldNames = Object.keys(d.fields);
  const fieldDeps  = fieldNames.map((k) => d.fields[k]);
  return Promise.all(fieldDeps.map(ctx.getMeasure)).then((subs) => {
    const fields = {};
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

function matSuperpose(name, d, ctx) {
  // Superpose: concat parents' samples + logWeights, systematic-
  // resample to ctx.sampleCount. Mass-faithful: result's totalmass
  // equals the sum of parents' totalmasses (logSumExp of their
  // logTotalmasses); resampling produces equally-weighted atoms each
  // carrying (totalInputMass / N) of mass.
  return Promise.all(d.fromNames.map(ctx.getMeasure)).then((parents) => {
    let totalN = 0;
    for (const p of parents) totalN += p.samples.length;
    if (totalN === 0) {
      return { samples: new Float64Array(0), logWeights: null,
               logTotalmass: -Infinity, n_eff: 0 };
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
    return {
      samples: out,
      logWeights: outW,
      logTotalmass: totalLogMass,
      // After systematic resampling the atoms are uniform → n_eff = N.
      n_eff: ctx.sampleCount,
    };
  });
}

function matBayesupdate(d, ctx) {
  // Reweight the prior atoms by per-atom log-likelihood. Per spec:
  //   posterior = bayesupdate(L, prior),  L = likelihoodof(K, obs)
  // For each prior atom θ_i, logw_i = logdensityof(K(θ_i), obs).
  // The atoms are the prior's; logWeights = prior.logWeights + per-i logp.
  const bodyIR = d.bodyIR
    ? orchestrator.expandMeasureRefsInIR(d.bodyIR, ctx.derivations)
    : orchestrator.expandMeasureIR(d.bodyName, ctx.derivations);
  if (!bodyIR) {
    return Promise.reject(new Error('bayesupdate: cannot expand body into measure IR'));
  }
  const valueRefs = [];
  orchestrator.collectSelfRefs(bodyIR).forEach((n) => valueRefs.push(n));
  return Promise.all([ctx.getMeasure(d.from)].concat(valueRefs.map(ctx.getMeasure)))
    .then((arr) => {
      const parent = arr[0];
      const refMeasures = arr.slice(1);
      const refArrays = {};
      for (let i = 0; i < valueRefs.length; i++) {
        const rm = refMeasures[i];
        if (!rm || !rm.samples || !rm.samples.BYTES_PER_ELEMENT) {
          throw new Error('bayesupdate: ref "' + valueRefs[i] +
            '" did not materialise to a scalar EmpiricalMeasure');
        }
        refArrays[valueRefs[i]] = rm.samples;
      }
      const observed = orchestrator.resolveIRToValue(
        d.obsIR, ctx.bindings, ctx.fixedValues);
      return ctx.sendWorker({
        type: 'logDensityN',
        ir: bodyIR,
        count: ctx.sampleCount,
        refArrays: refArrays,
        observed: observed,
        tally: 'clamped',
      }).then((reply) => {
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
        return {
          samples: parent.samples,
          logWeights: newLW,
          logTotalmass: lTM,
          n_eff: nEff,
        };
      });
    });
}

/**
 * Map a structural set descriptor (orchestrator.parseSetIR shape) to
 * numeric [lo, hi] bounds. Mirrors worker.setBoundsFor for the
 * materialiser's filter-only fallback path; returns null for set
 * kinds the materialiser doesn't yet support (integers / booleans).
 */
function setBoundsForMat(setDescr) {
  if (!setDescr) return null;
  switch (setDescr.kind) {
    case 'interval':    return [+setDescr.lo, +setDescr.hi];
    case 'reals':       return [-Infinity, Infinity];
    case 'posreals':    return [0, Infinity];
    case 'nonnegreals': return [0, Infinity];
    case 'unitinterval': return [0, 1];
    default:            return null;
  }
}

/**
 * truncate(M, S) — restrict M's support to S per spec §06.
 *
 * Three paths, chosen at materialise time:
 *
 *   (A) CDF-inverse — when the parent expands to a self-contained
 *       call IR for a known stdlib distribution with static params
 *       (no value-refs in kwargs) and the set descriptor maps to
 *       numeric [lo, hi] bounds. Worker computes Q(F(lo)+u·ΔF) per
 *       atom. Exact; no NaNs; uniform weights. logTotalmass shifts
 *       by log(ΔF).
 *
 *   (B) Rejection-redraw — when the parent expands to a sampleable
 *       IR but isn't (or isn't suitable as) a built-in CDF target.
 *       Worker draws from the expanded IR up to ctx.rejectionBudget
 *       times per atom; budget-exhausted atoms become NaN. n_eff
 *       drops to the count of valid atoms; logTotalmass shifts by
 *       log(empirical-acceptance-probability).
 *
 *   (C) Filter-only fallback — when expandMeasureIR returns null
 *       (the parent's derivation chain hits a kind we can't lift to a
 *       self-contained IR, e.g. a chain through `normalize` or
 *       `superpose`). We keep the parent's atoms that lie in S and
 *       NaN the rest, with a matching -inf log-weight on those
 *       slots. No redraws, so N_valid may be substantially below the
 *       requested sample count.
 *
 * Per-atom rejection budget reads from ctx.rejectionBudget (defaults
 * to 1000). Configurable per host: the VS Code extension exposes it
 * as a setting; the web gallery uses the default.
 */
function matTruncate(d, ctx) {
  return ctx.getMeasure(d.from).then((parent) => {
    const N = ctx.sampleCount;
    const parentLTM = (typeof parent.logTotalmass === 'number') ? parent.logTotalmass : 0;
    const bounds = setBoundsForMat(d.setDescr);
    if (!bounds) {
      return Promise.reject(new Error(
        'truncate: unsupported set kind \'' + (d.setDescr && d.setDescr.kind) + '\''));
    }
    const [lo, hi] = bounds;

    // Try to lift the parent to a self-contained sample IR. If that
    // succeeds, we can take the worker-mediated paths (A or B).
    const expanded = orchestrator.expandMeasureIR(d.from, ctx.derivations);
    const parentUniform = !parent.logWeights;
    if (expanded && expanded.kind === 'call' && parentUniform) {
      const valueRefs = orchestrator.collectSelfRefs(expanded);
      const hasRefs = valueRefs.size > 0 || (Array.isArray(valueRefs) && valueRefs.length > 0);
      const cdfEligible = !hasRefs && orchestrator.SAMPLEABLE_DISTRIBUTIONS
        && orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(expanded.op);
      const seed = nameSeed(d.from + '|truncate', ctx.rootSeed);

      if (cdfEligible) {
        return ctx.sendWorker({
          type: 'truncateSampleN',
          ir: expanded,
          setDescr: d.setDescr,
          count: N,
          mode: 'cdf',
          seed: seed,
        }).then((reply) => ({
          samples: reply.samples,
          logWeights: null,
          logTotalmass: parentLTM + reply.logShift,
          n_eff: reply.n_eff,
        }));
      }

      // Rejection-redraw. Need refArrays for parametric params.
      return collectRefArrays(expanded, ctx.fixedValues, ctx.getMeasure)
        .then((refArrays) => ctx.sendWorker({
          type: 'truncateSampleN',
          ir: expanded,
          setDescr: d.setDescr,
          count: N,
          mode: 'rejection',
          budget: ctx.rejectionBudget != null ? ctx.rejectionBudget : 1000,
          refArrays: refArrays,
          seed: seed,
        }))
        .then((reply) => {
          // Map budget-exhausted NaN slots to -inf log-weights so
          // downstream evaluateN propagation can mask them out
          // mass-correctly. Successful atoms keep uniform weight
          // (logWeights = null on the result is fine when there were
          // no NaNs; otherwise we attach a 0/-inf mask).
          const samples = reply.samples;
          let anyNaN = false;
          for (let i = 0; i < samples.length; i++) {
            if (Number.isNaN(samples[i])) { anyNaN = true; break; }
          }
          let logWeights = null;
          if (anyNaN) {
            logWeights = new Float64Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
              logWeights[i] = Number.isNaN(samples[i]) ? -Infinity : 0;
            }
          }
          return {
            samples: samples,
            logWeights: logWeights,
            logTotalmass: parentLTM + reply.logShift,
            n_eff: reply.n_eff,
          };
        });
    }

    // Filter-only fallback. We have parent.samples (possibly weighted)
    // — keep atoms in S, NaN the rest. logTotalmass tracks the
    // empirical M(S) using the parent's weights.
    const parentSamples = parent.samples;
    const out = new Float64Array(parentSamples.length);
    const outW = parent.logWeights
      ? new Float64Array(parentSamples.length)
      : null;
    let n_eff = 0;
    if (parent.logWeights) {
      // Accumulate logSumExp of accepted weights for the truncated mass.
      let acceptedWeights = [];
      let totalWeights = [];
      for (let i = 0; i < parentSamples.length; i++) {
        const x = parentSamples[i];
        totalWeights.push(parent.logWeights[i]);
        if (x >= lo && x <= hi) {
          out[i] = x;
          outW[i] = parent.logWeights[i];
          acceptedWeights.push(parent.logWeights[i]);
          n_eff++;
        } else {
          out[i] = NaN;
          outW[i] = -Infinity;
        }
      }
      const lseA = acceptedWeights.length
        ? empirical.logSumExp(Float64Array.from(acceptedWeights))
        : -Infinity;
      const lseT = empirical.logSumExp(Float64Array.from(totalWeights));
      // log(M(S)/M(R)) — caller already has parent.logTotalmass; we
      // add the empirical conditional log-probability.
      const logShift = isFinite(lseA) && isFinite(lseT) ? (lseA - lseT) : -Infinity;
      return {
        samples: out,
        logWeights: outW,
        logTotalmass: parentLTM + logShift,
        n_eff: n_eff,
      };
    }
    // Uniform parent: simple count-based shift.
    let anyNaN = false;
    for (let i = 0; i < parentSamples.length; i++) {
      const x = parentSamples[i];
      if (x >= lo && x <= hi) { out[i] = x; n_eff++; }
      else { out[i] = NaN; anyNaN = true; }
    }
    let logWeights = null;
    if (anyNaN) {
      logWeights = new Float64Array(parentSamples.length);
      for (let i = 0; i < parentSamples.length; i++) {
        logWeights[i] = Number.isNaN(out[i]) ? -Infinity : 0;
      }
    }
    const logShift = n_eff > 0
      ? Math.log(n_eff / parentSamples.length)
      : -Infinity;
    return {
      samples: out,
      logWeights: logWeights,
      logTotalmass: parentLTM + logShift,
      n_eff: n_eff,
    };
  });
}

function matTotalmass(d, ctx) {
  // totalmass(M): per spec §06, scalar mass of a (possibly
  // unnormalized) measure. The orchestrator tracks each measure's
  // logTotalmass through every materialisation step (algebraic
  // propagation for joint / iid / weighted / superpose / normalize,
  // empirical logSumExp for reweighted measures). totalmass(M)
  // exposes that as a per-atom scalar value — broadcast since we
  // track a single ensemble logTotalmass per measure today; per-atom
  // tracking is a separate refinement.
  return ctx.getMeasure(d.measureName).then((m) => {
    const N = ctx.sampleCount;
    const tm = Math.exp(typeof m.logTotalmass === 'number' ? m.logTotalmass : 0);
    const samples = new Float64Array(N);
    samples.fill(tm);
    return {
      samples: samples,
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    };
  });
}

function matLogdensityof(d, ctx) {
  // Per spec §sec:posterior: broadcast logdensityof over prior atoms.
  // For each atom i of M, evaluate logp(obs | M_i). Produces a per-i
  // value (a scalar binding) — no logWeights, no totalmass mutation.
  //
  // chain MARGINALISATION: when the measure was originally a
  // `chain(prior, K)` (per spec §06 ν(B) = ∫ K(a, B) dμ(a)), the
  // per-atom log-likelihoods we compute below are exactly the
  // integrand evaluated at MC samples a_i ~ μ. The marginal
  // log-density of the chain at obs is:
  //
  //   log p_ν(obs) = log ∫ p_K(obs | a) dμ(a)
  //                ≈ logsumexp_i { log p_K(obs | a_i) } − log N
  //
  // We detect chain origin via the derivation's chainOrigin flag
  // (set by buildDerivations from the binding's pre-lift surface)
  // and reduce the per-atom output to this scalar, broadcast to N
  // for shape consistency with the per-atom convention. n_eff
  // collapses to 1 — there's only one estimator here, even though
  // it's built from N prior samples.
  const measureDeriv = ctx.derivations[d.measureName];
  const isChain = !!(measureDeriv && measureDeriv.chainOrigin);

  const measureIR = orchestrator.expandMeasureIR(d.measureName, ctx.derivations);
  if (!measureIR) {
    return Promise.reject(new Error('logdensityof: cannot expand measure "'
      + d.measureName + '" into a self-contained IR'));
  }
  const valueRefs = [];
  orchestrator.collectSelfRefs(measureIR).forEach((n) => valueRefs.push(n));
  return Promise.all(valueRefs.map(ctx.getMeasure)).then((refMeasures) => {
    const refArrays = {};
    for (let i = 0; i < valueRefs.length; i++) {
      const rm = refMeasures[i];
      if (!rm || !rm.samples || !rm.samples.BYTES_PER_ELEMENT) {
        throw new Error('logdensityof: ref "' + valueRefs[i] +
          '" did not materialise to a scalar EmpiricalMeasure');
      }
      refArrays[valueRefs[i]] = rm.samples;
    }
    const observed = orchestrator.resolveIRToValue(
      d.obsIR, ctx.bindings, ctx.fixedValues);
    return ctx.sendWorker({
      type: 'logDensityN',
      ir: measureIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      observed: observed,
      tally: 'clamped',
    }).then((reply) => {
      if (!isChain) {
        return {
          samples: reply.samples,
          logWeights: null,
          logTotalmass: 0,
          n_eff: reply.samples.length,
        };
      }
      // chain marginalisation reduction.
      const perAtom = reply.samples;
      const lse = empirical.logSumExp(perAtom);
      const margLogp = lse - Math.log(perAtom.length);
      const out = new Float64Array(perAtom.length);
      out.fill(margLogp);
      return {
        samples: out,
        logWeights: null,
        logTotalmass: 0,
        n_eff: 1,
      };
    });
  });
}

// =====================================================================
// Top-level dispatch
// =====================================================================

const KIND_HANDLERS = {
  alias:        (name, d, ctx) => matAlias(d, ctx),
  sample:       (name, d, ctx) => matSample(name, d, ctx),
  evaluate:     (name, d, ctx) => matEvaluate(d, ctx),
  array:        (name, d) =>      matArray(d),
  weighted:     (name, d, ctx) => matWeighted(d, ctx),
  normalize:    (name, d, ctx) => matNormalize(d, ctx),
  iid:          (name, d, ctx) => matIid(name, d, ctx),
  tuple:        (name, d, ctx) => matTuple(d, ctx),
  record:       (name, d, ctx) => matRecord(d, ctx),
  superpose:    (name, d, ctx) => matSuperpose(name, d, ctx),
  bayesupdate:  (name, d, ctx) => matBayesupdate(d, ctx),
  logdensityof: (name, d, ctx) => matLogdensityof(d, ctx),
  totalmass:    (name, d, ctx) => matTotalmass(d, ctx),
  truncate:     (name, d, ctx) => matTruncate(d, ctx),
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
function materialiseMeasure(name, ctx) {
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
  const handler = KIND_HANDLERS[d.kind];
  if (!handler) {
    return Promise.reject(new Error('unknown derivation kind: ' + d.kind));
  }
  return handler(name, d, ctx);
}

module.exports = {
  materialiseMeasure,
  // Helpers exposed for the viewer's plot-plan fallbacks (which
  // sometimes need to compose the same primitives outside the kind-
  // dispatch path — e.g., to render a kernel-applied measure that
  // doesn't have a binding-graph derivation).
  fixedValueToMeasure,
  collectRefArrays,
  nameSeed,
  makeMainThreadPrng,
};
