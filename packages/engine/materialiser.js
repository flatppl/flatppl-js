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
// extended with two scalar metadata fields, plus an internal Value
// view (shape-explicit refactor, Phase 4):
//
//   { samples:      Float64Array,             // per-atom values (legacy view)
//     value?:       Value,                    // shape-explicit view (Phase 4)
//     logWeights:   Float64Array | null,      // null = uniform 1/N
//     logTotalmass: number,                   // default 0 (= totalmass 1)
//     n_eff:        number,                   // default = samples.length
//     fields?:      { name → Measure },       // record/joint shape
//     elems?:       Measure[],                // tuple shape
//     shape?:       'record' | 'tuple' | 'array',  // kind discriminator (string)
//     dims?:        number[],                 // legacy intrinsic-dim suffix for
//                                             //   vector-atom measures (matIid)
//     ... }
//
// **Phase 4 contract.** The internal `.value` field carries the
// shape-explicit Value (see engine/value.js) for scalar-leaf measures
// — i.e. measures whose atoms are real-valued and not records/tuples.
// `.value.shape` includes the leading N (atom count) axis: shape=[N]
// for scalar atoms, shape=[N, k] for k-vector atoms (matIid), etc.
// `.value.data` shares storage with the legacy `.samples` Float64Array
// (no copy). The Klein-4 transpose tag is preserved through value-ops
// dispatch.
//
// Phase 4a (this commit) introduces the helpers `valueOf(m)` and
// `measureFromValue(v, extras)` plus documents the contract. The
// handlers themselves still produce the legacy shape; Phase 4b
// migrates each handler to populate `.value` and the consumers
// (density.js Phase 5, MvNormal Phase 6) start reading via `valueOf`.
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
const valueLib     = require('./value');

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
    for (let i = 0; i < names.length; i++) {
      // Phase 8: refArrays uniformly carry Values internally. Phase 4b
      // ensures every scalar-leaf Measure has a populated `.value`
      // (shape=[N] for scalar atoms, shape=[N, ...dims] for vector
      // atoms). Consumers — _perAtomFallback / walkLeaf accessors,
      // broadcast helpers — assume Value inputs and dispatch on shape.
      // The legacy "Float64Array for scalar-atom parents" path is gone;
      // the uniform Value type is the single internal contract.
      const m = measures[i];
      if (m.value) {
        out[names[i]] = m.value;
      } else if (m.samples) {
        // Defensive: pre-Phase-4b measures (shouldn't exist post-
        // migration, but guard for hand-crafted Measure inputs in
        // tests that bypass the materialiser).
        out[names[i]] = valueLib.batchedScalar(m.samples);
      } else {
        // No data at all — shouldn't happen for scalar-leaf measures.
        throw new Error('collectRefArrays: measure for "' + names[i]
          + '" has neither .value nor .samples');
      }
    }
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
    return scalarMeasureN(arr,
      { logWeights: null, logTotalmass: 0, n_eff: sampleCount });
  }
  if (v instanceof Float64Array || v instanceof Int32Array || v instanceof Uint8Array) {
    const samples = Float64Array.from(v);
    return scalarMeasureN(samples,
      { logWeights: null, logTotalmass: 0, n_eff: samples.length });
  }
  if (Array.isArray(v)) {
    // Plain JS array: flat scalar vector OR mixed-shape tuple. Scalars
    // are finite numbers or booleans — a deterministic boolean array
    // (e.g. an elementwise comparison / `.<` broadcast bound to a
    // name) is a value, not a 3-element tuple of nulls. Booleans
    // coerce to 1/0, matching how the engine surfaces booleans
    // everywhere else (a scalar `true` binding already samples as 1).
    let allScalar = v.length > 0;
    for (let i = 0; allScalar && i < v.length; i++) {
      const t = typeof v[i];
      if (t === 'boolean') continue;
      if (t !== 'number' || !Number.isFinite(v[i])) allScalar = false;
    }
    if (allScalar) {
      const samples = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) {
        samples[i] = v[i] === true ? 1 : v[i] === false ? 0 : v[i];
      }
      return scalarMeasureN(samples,
        { logWeights: null, logTotalmass: 0, n_eff: samples.length });
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

/**
 * Return a Value view of a scalar-leaf measure (Phase 4 helper). The
 * Value's `data` SHARES STORAGE with `m.samples` — no copy. Shape is
 * computed from the legacy `.dims` suffix:
 *
 *   m.dims = undefined         → shape = [N]            (scalar atoms)
 *   m.dims = [k]               → shape = [N, k]         (k-vector atoms)
 *   m.dims = [m, n]            → shape = [N, m, n]      (matrix atoms)
 *
 * Returns `null` for measures without a top-level `.samples` field
 * (records / tuples — call `valueOf` on a field or element of those
 * instead).
 *
 * Prefers `m.value` if already populated (Phase 4b migration); falls
 * back to building one from `.samples` and `.dims` (works for
 * pre-migration handlers).
 */
function valueOf(m) {
  if (!m) return null;
  if (m.value != null) return m.value;
  if (!m.samples) return null;
  const N = m.samples.length / (m.dims ? m.dims.reduce((a, b) => a * b, 1) : 1);
  const shape = m.dims ? [N | 0].concat(m.dims) : [m.samples.length];
  return { shape: shape, data: m.samples };
}

/**
 * Convenience: build a scalar-atom Measure from a Float64Array of
 * per-atom samples plus the metadata fields. Equivalent to
 * `measureFromValue(batchedScalar(samples), extras)` — the most common
 * call shape in the handler migrations.
 */
function scalarMeasureN(samples, extras) {
  return measureFromValue(valueLib.batchedScalar(samples), extras);
}

/**
 * Build a Measure record from a Value plus the measure-metadata fields.
 * `v.data` becomes both `.samples` (legacy view) and `.value.data`
 * (shared storage; no copy). `extras` carries logWeights /
 * logTotalmass / n_eff / dims; defaults are uniform weights and
 * full ESS. The Value's tail dimensions (shape after the leading N)
 * are written into `dims` so legacy consumers continue to work.
 *
 *   measureFromValue({shape: [N], data: …})           → scalar-atom Measure
 *   measureFromValue({shape: [N, k], data: …})        → k-vector-atom Measure
 *   measureFromValue({shape: [N, m, n], data: …})     → matrix-atom Measure
 */
function measureFromValue(v, extras) {
  if (!valueLib.isValue(v)) {
    throw new Error('measureFromValue: argument is not a Value');
  }
  if (v.shape.length === 0) {
    throw new Error(
      'measureFromValue: scalar Value (shape=[]) has no atom axis; ' +
      'wrap into a batched scalar (shape=[N]) first');
  }
  const N = v.shape[0];
  const dims = v.shape.length > 1 ? v.shape.slice(1) : undefined;
  extras = extras || {};
  const m = {
    samples:      v.data,
    value:        v,
    logWeights:   extras.logWeights != null ? extras.logWeights : null,
    logTotalmass: extras.logTotalmass != null ? extras.logTotalmass : 0,
    n_eff:        extras.n_eff != null ? extras.n_eff : N,
  };
  // Vector-atom measures carry the legacy `dims` + `shape: 'array'`
  // discriminator so downstream consumers (viewer plot dispatcher,
  // empirical.shapeOf, joint diagnostics) classify them uniformly with
  // matIid-produced array-shape measures. Without the `shape: 'array'`
  // marker, the viewer treats vector-atom data as scalar atoms and
  // mis-plots the result.
  if (dims) {
    m.dims = dims;
    m.shape = 'array';
  }
  // Complex-valued binding: `.samples` stays the real part (the legacy
  // scalar view); the imaginary buffer is exposed alongside as `.imag`
  // and the Measure is tagged `dtype: 'complex'` so the viewer can
  // render it as a complex quantity (modulus / Argand / re+im) rather
  // than silently plotting only the real part. Planar storage means
  // `.imag` shares the Value's `.im` buffer (no copy).
  if (valueLib.isComplexValue(v)) {
    m.imag = v.im;
    m.dtype = 'complex';
  }
  return m;
}

/**
 * Build a Measure from a worker `evaluateN` reply, transparently
 * handling the real / complex and scalar-atom / vector-atom cases.
 * `reply.imag` (present iff the per-atom result was complex) is paired
 * with `reply.samples` into a planar complex Value; `reply.dims`
 * (present for vector-output ops) becomes the per-atom shape.
 */
function measureFromReply(reply, count, extras) {
  const dims = reply.dims;
  const shape = dims ? [count | 0].concat(dims) : [reply.samples.length];
  const v = reply.imag
    ? valueLib.complexValue(reply.samples, reply.imag, shape)
    : { shape: shape, data: reply.samples };
  return measureFromValue(v, extras);
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
      return scalarMeasureN(reply.samples, {
        logWeights: lw,
        logTotalmass: 0,
        n_eff: reply.samples.length,
      });
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
      // Phase 8: uniform Value refArrays internally. Mirrors
      // collectRefArrays. Phase 4b populates `.value` on every
      // scalar-leaf parent; we fall back to wrapping `.samples`
      // defensively for pre-migration measures.
      const m = parentMeasures[i];
      if (m.value) {
        refArrays[parentNames[i]] = m.value;
      } else if (m.samples) {
        refArrays[parentNames[i]] = valueLib.batchedScalar(m.samples);
      } else {
        throw new Error('matEvaluate: parent "' + parentNames[i]
          + '" has neither .value nor .samples');
      }
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
      // Phase 7c: vector-output ops (softmax / l1unit / l2unit /
      // logsoftmax) produce reply.dims; complex-valued transforms
      // produce reply.imag. measureFromReply handles both (and the
      // plain scalar-real case) uniformly.
      return measureFromReply(reply, ctx.sampleCount, {
        logWeights: lw,
        logTotalmass: logTotalmass,
        n_eff: n_eff,
      });
    });
  });
}

// Function-typed bindings (fn / functionof / kernelof / bijection)
// aren't materialisable as values — they're consulted by name at
// density / sample dispatch time. matLogdensityof / matBayesupdate use
// this to filter such refs out of their "materialise the parents"
// pre-pass; the parents that ARE values still need materialising.
function isFunctionLikeBinding(binding) {
  if (!binding) return false;
  switch (binding.type) {
    case 'fn':
    case 'functionof':
    case 'kernelof':
    case 'bijection':
      return true;
    default:
      return false;
  }
}

function matPushfwd(name, d, ctx) {
  // pushfwd(f, M): the pushforward of M through function f. Per spec
  // §06, samples are { f(x) : x ~ M }. We get M's samples via the
  // recursive getMeasure, then run one batched evaluateN over f's
  // body with refArrays binding f's param name to M's samples — same
  // mass / logWeights / n_eff propagate through unchanged (the
  // pushforward map preserves total mass; bijection or not).
  //
  // For density evaluation, density.js consults f's bijection
  // annotation via opts.resolveBijection (set up by matLogdensityof /
  // matBayesupdate when they encounter pushfwd in the expanded IR).
  const fBinding = ctx.bindings && ctx.bindings.get(d.fnRef);
  if (!fBinding) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' not found`));
  }
  const fnInfo = resolveFnBody(fBinding, ctx.bindings);
  if (!fnInfo) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' has no callable body`
      + ` (type=${fBinding.type})`));
  }
  return ctx.getMeasure(d.from).then((M) => {
    if (!M.samples) {
      return Promise.reject(new Error(`pushfwd: base measure '${d.from}' is not scalar `
        + `(record/tuple/iid not yet supported for first-class pushfwd materialisation)`));
    }
    return ctx.sendWorker({
      type: 'evaluateN',
      ir: fnInfo.body,
      count: M.samples.length,
      refArrays: { [fnInfo.paramName]: M.samples },
    }).then((reply) => {
      return measureFromReply(reply, M.samples.length, {
        logWeights: M.logWeights,
        logTotalmass: M.logTotalmass,
        n_eff: M.n_eff,
      });
    });
  });
}

// Resolve a function binding (fn / functionof / kernelof / bijection)
// to its callable body + single-param name. Bijection bindings carry
// their `f` component in `binding.ir.args[0]` (the bijection
// annotation classifier stores it via a ref so the orchestrator
// preserves the inner function); we follow that ref to the actual
// function binding's body. Returns { body, paramName } or null when
// the binding isn't function-shaped.
function resolveFnBody(binding, bindings) {
  if (!binding) return null;
  if (binding.type === 'bijection') {
    // The bijection's first arg is the forward function f. Follow
    // through to its body. Stored as binding.bijection = { fName,
    // fInvName, logVolume } by the classifier (see orchestrator
    // classifyBijection).
    if (!binding.bijection || !binding.bijection.fName) return null;
    const fwd = bindings.get(binding.bijection.fName);
    return resolveFnBody(fwd, bindings);
  }
  if (binding.type !== 'fn' && binding.type !== 'functionof'
      && binding.type !== 'kernelof') {
    return null;
  }
  if (!binding.ir || binding.ir.kind !== 'call'
      || binding.ir.op !== 'functionof' || !binding.ir.body) {
    return null;
  }
  const params = binding.ir.params || [];
  if (params.length !== 1) return null;
  return { body: binding.ir.body, paramName: params[0] };
}

function matArray(d) {
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
      // Uniform constant-shift doesn't change relative weights → n_eff unchanged.
      n_eff: (typeof parent.n_eff === 'number') ? parent.n_eff : N,
    });
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
    return scalarMeasureN(lifted.samples,
      { logWeights: w, logTotalmass: 0, n_eff: nEff });
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
      //
      // Vector-atom Value: shape=[N, ...dims]. data is atom-major.
      const value = { shape: [ctx.sampleCount | 0].concat(d.dims), data: reply.samples };
      return Object.assign(
        empirical.arrayMeasure(reply.samples, d.dims, null),
        { value: value, logTotalmass: 0, n_eff: ctx.sampleCount },
      );
    });
}

function matKernelBroadcast(name, d, ctx) {
  // broadcast(Dist, c1, c2, …) — array-valued independent-product
  // measure (spec §04). Element j of every atom is drawn from
  // Dist(params_j), where params_j is the j-th element of each
  // collection arg (rank-1) or the held-constant scalar (rank-0 /
  // length-1 singleton — the 87c9be1 shape rules, v1 = 1-D). Result
  // is a vector-atom measure shape=[N, K], atom-major, mirroring
  // matIid. Probability kernel ⇒ independent product is a probability
  // measure: logTotalmass 0, n_eff N. Closed-form logdensity is a
  // documented follow-up (TODO §04), exactly as matIid defers it.
  const sampler = require('./sampler');
  const params = (sampler._internal.REGISTRY[d.distOp] || {}).params;
  if (!params) {
    return Promise.reject(new Error(
      'broadcast: unknown distribution kernel ' + d.distOp));
  }
  // Map parameter name → source Value (resolved atom-indep, like
  // matMvNormal does for mu/cov).
  const srcByParam = {};
  try {
    if (d.kwargIRs && Object.keys(d.kwargIRs).length > 0) {
      for (const pn of Object.keys(d.kwargIRs)) {
        srcByParam[pn] = valueLib.asValue(orchestrator.resolveIRToValue(
          d.kwargIRs[pn], ctx.bindings, ctx.fixedValues));
      }
    } else {
      if (d.argIRs.length !== params.length) {
        throw new Error('broadcast(' + d.distOp + '): expected '
          + params.length + ' parameter args (' + params.join(', ')
          + '), got ' + d.argIRs.length);
      }
      for (let i = 0; i < params.length; i++) {
        srcByParam[params[i]] = valueLib.asValue(orchestrator.resolveIRToValue(
          d.argIRs[i], ctx.bindings, ctx.fixedValues));
      }
    }
  } catch (err) {
    return Promise.reject(err instanceof Error ? err : new Error(String(err)));
  }
  // Broadcast length K: the common length of the rank-1 collection
  // args; scalars / length-1 are singletons held constant.
  let K = 1;
  const pnames = Object.keys(srcByParam);
  for (const pn of pnames) {
    const v = srcByParam[pn];
    if (v.shape.length > 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp + '): parameter '
        + pn + ' must be a scalar or 1-D array (multi-axis is a follow-up),'
        + ' got shape=' + JSON.stringify(v.shape)));
    }
    const len = v.shape.length === 0 ? 1 : v.shape[0];
    if (len !== 1) {
      if (K !== 1 && K !== len) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): incompatible collection lengths (' + K + ' vs ' + len + ')'));
      }
      K = len;
    }
  }
  if (K < 1) {
    return Promise.reject(new Error('broadcast(' + d.distOp
      + '): empty collection argument'));
  }
  const N = ctx.sampleCount;
  // j-th element of a collection arg (scalar / length-1 → broadcast).
  const elemAt = (v, j) => {
    const len = v.shape.length === 0 ? 1 : v.shape[0];
    return v.data[len === 1 ? 0 : j];
  };

  // Closed-form specialization: broadcast(Normal, mu, sigma) is the
  // independent product of N(mu_j, sigma_j²) — i.e. MvNormal(mu,
  // diag(sigma²)). With the structured-matrix diag fast-paths this is
  // O(N·n), exact, and a single worker draw: lower_cholesky(diag) =
  // diag(sigma), then mu + L·z via the diag mulN/addN fast-paths
  // (exactly matMvNormal's pipeline, but the cov never densifies and
  // there is no O(n³) Cholesky). Scalar / held-constant mu or sigma
  // broadcast into the length-K vectors, so this also covers
  // broadcast(Normal, scalarMu, sigmas) etc.
  if (d.distOp === 'Normal' && srcByParam.mu && srcByParam.sigma) {
    const valueOps = require('./value-ops');
    const muVec  = new Float64Array(K);
    const sigSq  = new Float64Array(K);
    for (let j = 0; j < K; j++) {
      muVec[j] = elemAt(srcByParam.mu, j);
      const s = elemAt(srcByParam.sigma, j);
      sigSq[j] = s * s;
    }
    const cov = valueLib.diagMatrix(sigSq);                 // diag Value
    let L;
    try {
      L = sampler._internal.ARITH_OPS.lower_cholesky(cov);  // diag(σ), O(n)
    } catch (err) {
      return Promise.reject(new Error('broadcast(Normal): ' + err.message));
    }
    const stdNormalIR = {
      kind: 'call', op: 'Normal',
      kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
    };
    return ctx.sendWorker({
      type: 'sampleN', ir: stdNormalIR, count: N, repeat: K,
      refArrays: {}, seed: nameSeed(name, ctx.rootSeed),
    }).then((reply) => {
      const z = { shape: [N, K], data: reply.samples };     // [N,K] atom-major
      const Lz = valueOps.mulN(L, z, N);                     // diag·z, O(N·n)
      const result = valueOps.addN({ shape: [K], data: muVec }, Lz, N);
      return measureFromValue(result, {
        logWeights: null, logTotalmass: 0, n_eff: N,
      });
    });
  }

  // Per element j: build Dist(params_j) and draw N atoms. K small
  // (model dimension), N large — K leaf-sample calls is fine for v1.
  const cols = new Array(K);
  let chain = Promise.resolve();
  for (let j = 0; j < K; j++) {
    const jj = j;
    chain = chain.then(() => {
      const kwargs = {};
      for (const pn of pnames) {
        kwargs[pn] = { kind: 'lit', value: elemAt(srcByParam[pn], jj) };
      }
      const distIR = { kind: 'call', op: d.distOp, kwargs: kwargs };
      return ctx.sendWorker({
        type: 'sampleN', ir: distIR, count: N,
        refArrays: {},
        seed: nameSeed(name + ':' + jj, ctx.rootSeed),
      }).then((reply) => { cols[jj] = reply.samples; });
    });
  }
  return chain.then(() => {
    // Atom-major pack into shape=[N, K]: atom i occupies [i*K, (i+1)*K).
    const out = new Float64Array(N * K);
    for (let j = 0; j < K; j++) {
      const col = cols[j];
      for (let i = 0; i < N; i++) out[i * K + j] = col[i];
    }
    const value = { shape: [N | 0, K], data: out };
    return Object.assign(
      empirical.arrayMeasure(out, [K], null),
      { value: value, logTotalmass: 0, n_eff: N },
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
      // shape=[0] empty batched scalar. Use measureFromValue directly
      // because batchedScalar would also produce shape=[0] but going
      // through the helper documents the empty case.
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

function matBayesupdate(d, ctx) {
  // Reweight the prior atoms by per-atom log-likelihood. Per spec:
  //   posterior = bayesupdate(L, prior),  L = likelihoodof(K, obs)
  // For each prior atom θ_i, logw_i = logdensityof(K(θ_i), obs).
  // The atoms are the prior's; logWeights = prior.logWeights + per-i logp.
  const bodyIR = d.bodyIR
    ? orchestrator.expandMeasureRefsInIR(d.bodyIR, ctx.derivations)
    : orchestrator.expandMeasureIR(d.bodyName, ctx.derivations, undefined, ctx.bindings);
  if (!bodyIR) {
    return Promise.reject(new Error('bayesupdate: cannot expand body into measure IR'));
  }
  const valueRefs = [];
  orchestrator.collectSelfRefs(bodyIR).forEach((n) => {
    if (isFunctionLikeBinding(ctx.bindings && ctx.bindings.get(n))) return;
    valueRefs.push(n);
  });
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
        return scalarMeasureN(parent.samples, {
          logWeights: newLW,
          logTotalmass: lTM,
          n_eff: nEff,
        });
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
        }).then((reply) => scalarMeasureN(reply.samples, {
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
          return scalarMeasureN(samples, {
            logWeights: logWeights,
            logTotalmass: parentLTM + reply.logShift,
            n_eff: reply.n_eff,
          });
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
      return scalarMeasureN(out, {
        logWeights: outW,
        logTotalmass: parentLTM + logShift,
        n_eff: n_eff,
      });
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
    return scalarMeasureN(out, {
      logWeights: logWeights,
      logTotalmass: parentLTM + logShift,
      n_eff: n_eff,
    });
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
    return scalarMeasureN(samples, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

function matMvNormal(name, d, ctx) {
  // Phase 6 of the shape-explicit refactor.
  //
  // MvNormal(mu, cov) — per spec §08, samples are n-vectors with
  // x ~ Normal_n(mu, cov). Implementation routes through the spec
  // equivalence  pushfwd(fn(mu + L*_), iid(Normal(0,1), n))  but
  // without the AST rewrite — we do the matvec / vector-add directly
  // via value-ops so the per-atom batched form short-circuits.
  //
  // Pipeline:
  //   1. Resolve mu (atom-indep vector, shape=[n]) and cov (shape=[n, n]).
  //   2. L = lower_cholesky(cov)         — one O(n³) call.
  //   3. Draw N atoms of n standard normals → shape=[N, n].
  //   4. result = mu + L * z (value-ops.mulN + addN)  → shape=[N, n].
  //
  // logTotalmass = 0 (normalized probability measure); n_eff = N
  // (independent atoms). The output Measure is vector-atom shape
  // (samples + dims=[n]) via measureFromValue.
  const valueOps = require('./value-ops');
  const sampler  = require('./sampler');
  const distIR = d.distIR;
  if (!distIR || !distIR.kwargs || !distIR.kwargs.mu || !distIR.kwargs.cov) {
    return Promise.reject(new Error('MvNormal: requires mu and cov kwargs'));
  }
  // Evaluate mu / cov: atom-indep right now (per-atom params deferred).
  // We use the orchestrator's resolveIRToValue helper which threads
  // through fixedValues + bindings.
  const muVal = orchestrator.resolveIRToValue(
    distIR.kwargs.mu, ctx.bindings, ctx.fixedValues);
  const covVal = orchestrator.resolveIRToValue(
    distIR.kwargs.cov, ctx.bindings, ctx.fixedValues);
  if (muVal == null) {
    return Promise.reject(new Error('MvNormal: cannot resolve mu (per-atom params deferred)'));
  }
  if (covVal == null) {
    return Promise.reject(new Error('MvNormal: cannot resolve cov (per-atom params deferred)'));
  }
  // mu may arrive as nested JS array / flat array / Value. Normalise
  // to a Value with shape=[n].
  const muValue = valueLib.asValue(muVal);
  if (muValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'MvNormal: mu must be a vector, got shape=' + JSON.stringify(muValue.shape)));
  }
  const n = muValue.shape[0];
  // cov: accept nested JS array or shape=[n, n] Value.
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n || covValue.shape[1] !== n) {
    return Promise.reject(new Error(
      'MvNormal: cov must be ' + n + 'x' + n + ', got shape='
      + JSON.stringify(covValue.shape)));
  }
  // Cholesky factor (Value shape=[n, n]).
  let L;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('MvNormal: ' + err.message));
  }
  // Draw N × n standard normals. Use the worker to get a Float64Array
  // of length N*n; we treat it as the atom-major shape=[N, n] buffer.
  const N = ctx.sampleCount;
  const stdNormalIR = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
  };
  return ctx.sendWorker({
    type: 'sampleN', ir: stdNormalIR, count: N, repeat: n,
    refArrays: {},
    seed: nameSeed(name, ctx.rootSeed),
  }).then((reply) => {
    // z is shape=[N, n] atom-major. Build the Value.
    const z = { shape: [N, n], data: reply.samples };
    // L * z: shape=[N, n] (per-atom matvec via mulN). Then add mu
    // (atom-indep vector) via addN.
    const Lz = valueOps.mulN(L, z, N);
    const result = valueOps.addN(muValue, Lz, N);
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

function matLogdensityof(d, ctx) {
  // Per spec §sec:posterior: broadcast logdensityof over prior atoms.
  // For each atom i of M, evaluate logp(obs | M_i). Produces a per-i
  // value (a scalar binding) — no logWeights, no totalmass mutation.
  //
  // chain MARGINALISATION: when the measure was originally a
  // `kchain(prior, K)` (per spec §06 ν(B) = ∫ K(a, B) dμ(a)), the
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
  // chainOrigin: legacy inlineChainOps kchain tag. The first-class
  // path expresses the same MC marginal as kind:'jointchain' with
  // marginalize:true (kchain) — expandMeasureIR returns just the last
  // step's measure (prior var as a per-atom ref), so the SAME
  // logsumexp−logN reduction marginalizes the prior out.
  const isChain = !!(measureDeriv && (measureDeriv.chainOrigin
    || (measureDeriv.kind === 'jointchain' && measureDeriv.marginalize)));

  const measureIR = orchestrator.expandMeasureIR(d.measureName, ctx.derivations, undefined, ctx.bindings);
  if (!measureIR) {
    return Promise.reject(new Error('logdensityof: cannot expand measure "'
      + d.measureName + '" into a self-contained IR'));
  }
  const valueRefs = [];
  const fixedRefs = [];
  orchestrator.collectSelfRefs(measureIR).forEach((n) => {
    if (isFunctionLikeBinding(ctx.bindings && ctx.bindings.get(n))) return;
    // Walker-threaded names: a ref that is neither a binding nor a
    // fixed value is a synthetic joint step-variate name introduced by
    // expandMeasureIR's first-class jointchain canonicalisation (e.g.
    // the intermediate variates of an N-ary chain). The density
    // walker's `fields`/`source` overlay env-threading resolves these
    // from the consumed observation — they are NOT per-atom prior
    // refs, so they must not be materialised here (getMeasure would
    // fail with "no derivation"). Overlay precedence (overlay >
    // refArrays > baseEnv) makes this correct even when a name
    // coincides with a binding.
    if (!(ctx.bindings && ctx.bindings.has(n))
        && !(ctx.fixedValues && ctx.fixedValues.has(n))) {
      return;
    }
    // Fixed-phase refs (literal values, arrays/matrices, etc.) flow
    // through the worker's session env via setEnv — they're atom-
    // independent, so the density walker resolves them once per call
    // rather than once per atom. valueRefs holds only the per-atom
    // refs that need materialised refArrays.
    if (ctx.fixedValues && ctx.fixedValues.has(n)) {
      fixedRefs.push(n);
      return;
    }
    valueRefs.push(n);
  });
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
    // Push fixed-phase refs to the worker session env so the density
    // walker (which evaluates leaf params against baseEnv) can resolve
    // them. setEnv with merge=true so we don't clobber any
    // host-provided session env.
    let setEnvP = Promise.resolve();
    if (fixedRefs.length > 0) {
      const fixedEnv = {};
      for (const n of fixedRefs) fixedEnv[n] = ctx.fixedValues.get(n);
      setEnvP = ctx.sendWorker({ type: 'setEnv', env: fixedEnv, merge: true });
    }
    return setEnvP.then(() => ctx.sendWorker({
      type: 'logDensityN',
      ir: measureIR,
      count: ctx.sampleCount,
      refArrays: refArrays,
      observed: observed,
      tally: 'clamped',
    })).then((reply) => {
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
  kernelbroadcast: (name, d, ctx) => matKernelBroadcast(name, d, ctx),
  tuple:        (name, d, ctx) => matTuple(d, ctx),
  record:       (name, d, ctx) => matRecord(d, ctx),
  superpose:    (name, d, ctx) => matSuperpose(name, d, ctx),
  bayesupdate:  (name, d, ctx) => matBayesupdate(d, ctx),
  logdensityof: (name, d, ctx) => matLogdensityof(d, ctx),
  totalmass:    (name, d, ctx) => matTotalmass(d, ctx),
  truncate:     (name, d, ctx) => matTruncate(d, ctx),
  pushfwd:      (name, d, ctx) => matPushfwd(name, d, ctx),
  mvnormal:     (name, d, ctx) => matMvNormal(name, d, ctx),
  // jointchain/kchain first-class kind. Step-1 stub: the kind is only
  // produced when orchestrator JOINTCHAIN_STATE.firstClass is on (off by
  // default), and step-1 tests exercise classification only. Real
  // sampling lands in step 2 (sample base; per step apply kernelRef to
  // the accumulated prior atoms; marginalize ⇒ keep last step's
  // variates). Throws clearly if reached early.
  jointchain:   (name, d, ctx) => matJointchain(name, d, ctx),
};

function matJointchain(name, d, ctx) {
  // First-class jointchain/kchain materialisation (consume/rest
  // consolidation step 2b). Mirrors the spec §06 stochastic-node
  // equivalence WITHOUT the inlineChainOps AST rewrite:
  //   a ~ M           (step 0 base measure)
  //   b ~ K(a)        (step 1 kernel applied to the prior variate)
  //   jointchain ⇒ variate = cat(a, b)  (tuple, or record if labelled)
  //   kchain     ⇒ variate = b          (a marginalized; the MC
  //                                       density integral is step 2c)
  // Kernel application reuses the same per-atom primitive as
  // matPushfwd: resolve {body, paramName}, then one worker draw of K's
  // body measure with refArrays binding the param to the prior atoms
  // (structural — never by surface-kwarg-name matching).
  //
  // Scope (2b/2b-ext): N steps, left-associative; scalar base measure
  // and scalar-variate single-param kernels; the i-th kernel takes the
  // cat of all prior step variates (spec §06 `c ~ K3([a,b])`),
  // realised by rewiring the kernel param to ref(prior_0) / vector(ref
  // prior_0, …) + scalar refArrays (no worker vector-refArray ext).
  // Kernel-first base, record/tuple/iid base, and vector/record kernel
  // output remain explicit clear deferrals (flag OFF in production so
  // legacy inlineChainOps still owns uncovered shapes — no
  // regression). Density parity (expandMeasureIR): N-ary labelled
  // jointchain ✓; 2-step kchain MC ✓; N-ary kchain + positional
  // jointchain density are clean deferrals (null ⇒ reject; sampling
  // unaffected).
  // Structural sampler mirroring the density side (expandMeasureIR is
  // the "closure walk"): materialise the base (scalar ⇒ one variate;
  // record ⇒ its fields); for each kernel step expand the body
  // (following body→ref via expandMeasureIR into the self-contained
  // measure IR where the boundary params surface as leaf refs), then
  // sample each LEAF field with refArrays binding params to prior
  // variates — NAMED params auto-splat to like-named prior fields
  // (spec §04), a lone HOLE param rewires to the prior cat (spec §06
  // `c~K3([a,b])`). jointchain ⇒ record/tuple of all variates; kchain
  // ⇒ only the last step's variates (prior marginalised; density MC
  // via matLogdensityof isChain).
  const steps = (d && d.steps) || [];
  if (steps.length < 2) {
    return Promise.reject(new Error('jointchain: need at least 2 steps'));
  }
  const base = steps[0];
  if (base.kernel) {
    return Promise.reject(new Error(
      'jointchain: kernel-first base is itself a kernel, not a closed '
      + 'measure (spec §06 kernel-first chain) — not materialisable; '
      + 'disintegrate handles it structurally (step 3).'));
  }
  const baseP = base.ref != null
    ? ctx.getMeasure(base.ref)
    : ctx.sendWorker({
        type: 'sampleN', ir: base.measureIR, count: ctx.sampleCount,
        refArrays: {}, seed: nameSeed(name + ':jc0', ctx.rootSeed),
      }).then((r) => measureFromReply(r, ctx.sampleCount,
        { logTotalmass: 0, n_eff: ctx.sampleCount }));

  return Promise.resolve(baseP).then((M0) => {
    // priorVars: ordered [{ name, m }] of every scalar variate so far;
    // byName for auto-splat refArray binding. baseRefName lets a hole
    // param bind to the single base prior (scalar base).
    const priorVars = [];
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

    // Resolve a kernel step → { params, leaves:[{ name, ir }] }. Body
    // is expanded through refs (the closure walk); a record/joint body
    // contributes one leaf per field, a leaf-dist body one unnamed
    // leaf. One nesting level (the obs_dist=joint(y=Normal) shape);
    // deeper composites are a clean follow-up.
    const kernelLeaves = (kstep, fallbackName) => {
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
        body = orchestrator.expandMeasureIR(
          body.name, ctx.derivations, undefined, ctx.bindings);
        if (!body) return null;
      }
      let leaves;
      if (body.kind === 'call'
          && (body.op === 'joint' || body.op === 'record')
          && Array.isArray(body.fields)) {
        leaves = body.fields.map((fl) => ({ name: fl.name, ir: fl.value }));
      } else {
        leaves = [{ name: fallbackName, ir: body }];
      }
      return { params: f.params, leaves };
    };

    // Bind a leaf-dist IR's kernel params to prior variates and return
    // { ir, refArrays }. NAMED param matching a prior var ⇒ refArray
    // by that name (auto-splat). A lone HOLE param (single, unmatched)
    // ⇒ rewire to the prior cat: ref(p0) for one prior, vector(p0,…)
    // for ≥2; bind each via a synthetic scalar refArray.
    const PRIOR = (j) => '__jc$' + j;
    const bindLeaf = (leafIR, params) => {
      const refArrays = {};
      const named = params.filter((p) =>
        priorVars.some((v) => v.name === p));
      const holes = params.filter((p) => named.indexOf(p) === -1);
      for (const p of named) {
        const v = priorVars.find((q) => q.name === p);
        refArrays[p] = v.m.samples;
      }
      let ir = leafIR;
      if (holes.length > 0) {
        if (params.length !== 1) return null;     // multi-param + hole
        // Hole binds to the whole prior cat (all variates so far; for
        // a scalar base that's the base itself).
        const catVars = priorVars.length > 0
          ? priorVars
          : [{ name: baseRefName, m: M0 }];
        const k = catVars.length;
        for (let j = 0; j < k; j++) refArrays[PRIOR(j)] = catVars[j].m.samples;
        const param = holes[0];
        const sub = (node) => {
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
          const out = {};
          for (const key in node) out[key] = sub(node[key]);
          return out;
        };
        ir = sub(leafIR);
      }
      return { ir, refArrays };
    };

    // Fold the kernel steps left-associatively. Each step appends its
    // produced variate(s) to `produced` (and to priorVars so later
    // steps can splat them).
    let chainP = Promise.resolve([]);
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
        let p = Promise.resolve(produced);
        for (let li = 0; li < ke.leaves.length; li++) {
          const leaf = ke.leaves[li];
          p = p.then((acc) => {
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
            }).then((reply) => {
              const Mi = measureFromReply(reply, N, {
                logWeights: priorVars[0].m.logWeights,
                logTotalmass: 0, n_eff: priorVars[0].m.n_eff,
              });
              if (!Mi.samples) {
                return Promise.reject(new Error(
                  `jointchain: step ${i} kernel leaf produced a `
                  + 'non-scalar variate (follow-up)'));
              }
              const vn = leaf.name != null ? leaf.name : ('s' + i);
              priorVars.push({ name: vn, m: Mi });
              return acc.concat([{ name: vn, m: Mi }]);
            });
          });
        }
        return p;
      });
    }

    return chainP.then((produced) => {
      // kchain: only the last step's produced variate(s); prior
      // marginalised (density MC handled by matLogdensityof isChain).
      // jointchain: all variates (base + every produced).
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
      const subs = outPairs.map((x) => x.m);
      const lw = empirical.propagateLogWeights(subs);
      let nEff = N;
      for (const s of subs) {
        if (s.n_eff != null) nEff = Math.min(nEff, s.n_eff);
      }
      // Single scalar variate (e.g. scalar-base kchain) ⇒ a scalar
      // measure; multiple / named ⇒ a record; positional unlabelled
      // all-scalar jointchain ⇒ a tuple.
      if (outPairs.length === 1) {
        return Object.assign({}, outPairs[0].m,
          { logTotalmass: 0, n_eff: nEff });
      }
      const named = !!d.labels || (M0 && M0.fields)
        || outPairs.some((x) => !/^s\d+$/.test(x.name));
      if (named) {
        const fields = {};
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
  // Phase 4 helpers — Value ↔ Measure bridges.
  valueOf,
  measureFromValue,
  measureN,
};
