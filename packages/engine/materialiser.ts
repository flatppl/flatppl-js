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
// `.value` carries the shape-explicit Value (see engine/value.ts) for
// scalar-leaf measures — atoms are real-valued and not records/tuples.
// `.value.shape` includes the leading N (atom count) axis: shape=[N]
// for scalar atoms, shape=[N, k] for k-vector atoms (matIid), etc.
// `.value.data` shares storage with `.samples` (no copy). The Klein-4
// transpose tag is preserved through value-ops dispatch.
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

import type {
  DerivationAlias,
  DerivationArray,
  DerivationBase,
  DerivationBayesupdate,
  DerivationBinnedPoissonProcess,
  DerivationBroadcastLogdensity,
  DerivationDirichlet,
  DerivationEvaluate,
  DerivationIid,
  DerivationInverseWishart,
  DerivationLKJ,
  DerivationLKJCholesky,
  DerivationWishart,
  DerivationJointchain,
  DerivationKernelBroadcast,
  DerivationLogdensityof,
  DerivationMultinomial,
  DerivationMvNormal,
  DerivationNormalize,
  DerivationPushfwd,
  DerivationRecord,
  DerivationSample,
  DerivationSelect,
  DerivationSuperpose,
  DerivationTotalmass,
  DerivationTruncate,
  DerivationTuple,
  DerivationWeighted,
  EmpiricalMeasure,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const rng          = require('./rng.ts');
const valueLib     = require('./value.ts');

// =====================================================================
// Helpers
// =====================================================================

/**
 * FNV-1a 32-bit hash of `name` XOR'd against rootSeed. Gives each
 * binding its own deterministic RNG stream for sampleN — order-
 * independent so two unrelated bindings stay independent regardless
 * of which the user materialises first.
 */
function nameSeed(name: any, rootSeed: any) {
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
function makeMainThreadPrng(seed: any) {
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
function collectRefArrays(ir: any, fixedValues: any, getMeasure: any) {
  const refs = orchestrator.collectSelfRefs(ir);
  const names: string[] = [];
  refs.forEach((n: string) => {
    if (fixedValues && fixedValues.has(n)) return;
    names.push(n);
  });
  return Promise.all(names.map(getMeasure)).then((measures: any[]) => {
    const out: any = {};
    for (let i = 0; i < names.length; i++) {
      // refArrays uniformly carry Values internally. Every scalar-leaf
      // Measure has a populated `.value` (shape=[N] for scalar atoms,
      // shape=[N, ...dims] for vector atoms). Consumers
      // (_perAtomFallback / walkLeaf accessors, broadcast helpers)
      // assume Value inputs and dispatch on shape.
      const m = measures[i];
      if (m.value) {
        out[names[i]] = m.value;
      } else if (m.samples) {
        // Defensive: hand-crafted Measure inputs in tests that bypass
        // the materialiser may omit `.value`.
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
// Classify a JS array as a rectangular nested numeric array (rank ≥ 2).
// Returns { shape, size } if every leaf is a finite number / boolean and
// every level has consistent length; null otherwise. Rank-1 numeric
// arrays return null (handled by the dedicated allScalar branch above).
function _classifyNestedNumeric(v: any): any {
  if (!Array.isArray(v) || v.length === 0) return null;
  const shape: number[] = [];
  let cur: any = v;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    cur = cur.length > 0 ? cur[0] : null;
  }
  if (shape.length < 2) return null;
  // Verify rectangularity + numeric leaves.
  let size = 1;
  for (const d of shape) size *= d;
  if (size === 0) return null;
  function _check(node: any, depth: number): boolean {
    if (depth === shape.length) {
      const t = typeof node;
      return (t === 'number' && !Number.isNaN(node)) || t === 'boolean';
    }
    if (!Array.isArray(node) || node.length !== shape[depth]) return false;
    for (let i = 0; i < node.length; i++) {
      if (!_check(node[i], depth + 1)) return false;
    }
    return true;
  }
  return _check(v, 0) ? { shape, size } : null;
}

function _flattenInto(v: any, out: Float64Array, pos: number): number {
  if (Array.isArray(v)) {
    for (let i = 0; i < v.length; i++) pos = _flattenInto(v[i], out, pos);
    return pos;
  }
  out[pos] = v === true ? 1 : v === false ? 0 : +v;
  return pos + 1;
}

function fixedValueToMeasure(v: any, sampleCount: any) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    const arr = new Float64Array(sampleCount);
    arr.fill(v);
    return scalarMeasureN(arr,
      { logWeights: null, logTotalmass: 0, n_eff: sampleCount });
  }
  // Shape-explicit Value (e.g. produced by `eye(n)`, `diagmat`,
  // transpose / lower_cholesky chains). A fixed-phase Value is ONE
  // value, not an atom-batched measure: shape=[3, 3] means "a single
  // 3x3 matrix", not "3 atoms of 3-vector". If we passed it through
  // measureFromValue with its native rank, the viewer would dispatch
  // to the multivariate corner-plot path (renderRecordMarginals)
  // which would mis-label the matrix entries as "samples".
  // Flatten to a 1D scalar measure (length = prod(shape)) so the
  // viewer's array-mode step plot renders the values. A proper
  // rank-2+ heatmap renderer would preserve the matrix structure
  // visually — tracked as a follow-up.
  if (valueLib.isValue(v)) {
    const dense = valueLib.densify(v);
    const flat = dense.data instanceof Float64Array
      ? dense.data
      : Float64Array.from(dense.data);
    return scalarMeasureN(flat, {
      logWeights: null, logTotalmass: 0, n_eff: flat.length,
    });
  }
  // Complex scalar `z = complex(re, im)` materialises as { re, im }.
  // Bridge to a planar complex Value of shape=[1] so the viewer's
  // complex-aware paths (.samples=re, .imag=im, dtype='complex') fire.
  if (v && typeof v === 'object'
      && typeof v.re === 'number' && typeof v.im === 'number'
      && Object.keys(v).length === 2) {
    const re = new Float64Array([v.re]);
    const im = new Float64Array([v.im]);
    const cv = valueLib.complexValue(re, im, [1]);
    return measureFromValue(cv,
      { logWeights: null, logTotalmass: 0, n_eff: 1 });
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
      // Accept finite numbers and ±Infinity (extlinspace spec §07
      // emits ±Infinity as endpoints; NaN remains a real bug-signal).
      if (t !== 'number' || Number.isNaN(v[i])) allScalar = false;
    }
    if (allScalar) {
      const samples = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) {
        samples[i] = v[i] === true ? 1 : v[i] === false ? 0 : v[i];
      }
      return scalarMeasureN(samples,
        { logWeights: null, logTotalmass: 0, n_eff: samples.length });
    }
    // Rank-≥2 numeric array (e.g. fixed-phase 2D matrix from
    // `rand(state, iid(N, [3, 3]))`). A single fixed multi-dim value
    // is NOT an atom-batched measure (shape=[3, 3] means "a 3x3
    // matrix", not "3 atoms of 3-vector"). Flatten to a 1D scalar
    // measure of length prod(shape) so the viewer renders it as a
    // step plot of the values, not as a misleading "N samples of
    // a k-vector" corner plot. The multi-dim structure is lost in
    // the plot — a heatmap renderer for rank≥2 fixed arrays is a
    // tracked follow-up.
    const nested = _classifyNestedNumeric(v);
    if (nested) {
      const flat = new Float64Array(nested.size);
      _flattenInto(v, flat, 0);
      return scalarMeasureN(flat, {
        logWeights: null, logTotalmass: 0, n_eff: nested.size,
      });
    }
    const elems = new Array(v.length);
    for (let ei = 0; ei < v.length; ei++) elems[ei] = fixedValueToMeasure(v[ei], sampleCount);
    return { elems: elems, logTotalmass: 0, n_eff: sampleCount };
  }
  if (v && typeof v === 'object') {
    if (v.key && Array.isArray(v.key) && v.counter) return null;   // rngstate
    const fields: any = {};
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
function measureN(m: any) {
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
 * Return a Value view of a scalar-leaf measure. The Value's `data`
 * SHARES STORAGE with `m.samples` — no copy. Shape is computed from
 * the `.dims` suffix:
 *
 *   m.dims = undefined         → shape = [N]            (scalar atoms)
 *   m.dims = [k]               → shape = [N, k]         (k-vector atoms)
 *   m.dims = [m, n]            → shape = [N, m, n]      (matrix atoms)
 *
 * Returns `null` for measures without a top-level `.samples` field
 * (records / tuples — call `valueOf` on a field or element of those
 * instead). Prefers a pre-populated `m.value` if present.
 */
function valueOf(m: any) {
  if (!m) return null;
  if (m.value != null) return m.value;
  if (!m.samples) return null;
  const N = m.samples.length / (m.dims ? m.dims.reduce((a: any, b: any) => a * b, 1) : 1);
  const shape = m.dims ? [N | 0].concat(m.dims) : [m.samples.length];
  return { shape: shape, data: m.samples };
}

/**
 * Convenience: build a scalar-atom Measure from a Float64Array of
 * per-atom samples plus the metadata fields. Equivalent to
 * `measureFromValue(batchedScalar(samples), extras)` — the most common
 * call shape in the handler migrations.
 */
function scalarMeasureN(samples: any, extras: any) {
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
function measureFromValue(v: any, extras: any) {
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
  const m: any = {
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
function measureFromReply(reply: any, count: any, extras: any) {
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

function matSample(name: string, d: DerivationSample, ctx: any) {
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
  const refs = orchestrator.collectSelfRefs(d.ir);
  const parentNames: string[] = [];
  refs.forEach((n: any) => {
    if (ctx.fixedValues && ctx.fixedValues.has(n)) return;
    parentNames.push(n);
  });
  return Promise.all(parentNames.map(ctx.getMeasure)).then((parentMeasures: any[]) => {
    const refArrays: any = {};
    for (let i = 0; i < parentNames.length; i++) {
      // Uniform Value refArrays internally — mirror collectRefArrays.
      // Wrap `.samples` defensively for hand-crafted measures.
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
    }).then((reply: any) => {
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
      // Vector-output ops (softmax / l1unit / l2unit / logsoftmax)
      // produce reply.dims; complex-valued transforms produce
      // reply.imag. measureFromReply handles both (and the plain
      // scalar-real case) uniformly.
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
function isFunctionLikeBinding(binding: any) {
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

function matPushfwd(name: string, d: DerivationPushfwd, ctx: any) {
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
  return ctx.getMeasure(d.from).then((M: any) => {
    if (!M.samples) {
      return Promise.reject(new Error(`pushfwd: base measure '${d.from}' is not scalar `
        + `(record/tuple/iid not yet supported for first-class pushfwd materialisation)`));
    }
    return ctx.sendWorker({
      type: 'evaluateN',
      ir: fnInfo.body,
      count: M.samples.length,
      refArrays: { [fnInfo.paramName]: M.samples },
    }).then((reply: any) => {
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
function resolveFnBody(binding: any, bindings: any) {
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
  const k = d.dims.reduce((p: any, n: any) => p * n, 1);
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

function matKernelBroadcast(name: string, d: DerivationKernelBroadcast, ctx: any) {
  // broadcast(Dist, c1, c2, …) — array-valued independent-product
  // measure (spec §04). Element j of every atom is drawn from
  // Dist(params_j), where params_j is the j-th element of each
  // collection arg (rank-1) or the held-constant scalar (rank-0 /
  // length-1 singleton — the 87c9be1 shape rules, v1 = 1-D). Result
  // is a vector-atom measure shape=[N, K], atom-major, mirroring
  // matIid. Probability kernel ⇒ independent product is a probability
  // measure: logTotalmass 0, n_eff N. Closed-form logdensity is a
  // documented follow-up (TODO §04), exactly as matIid defers it.
  const sampler = require('./sampler.ts');
  const params = (sampler._internal.REGISTRY[d.distOp] || {}).params;
  if (!params) {
    return Promise.reject(new Error(
      'broadcast: unknown distribution kernel ' + d.distOp));
  }
  // Map parameter name → source Value (resolved atom-indep, like
  // matMvNormal does for mu/cov).
  const srcByParam: any = {};
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
  const elemAt = (v: any, j: any) => {
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
    const valueOps = require('./value-ops.ts');
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
      return Promise.reject(new Error('broadcast(Normal): ' + (err as any).message));
    }
    const stdNormalIR = {
      kind: 'call', op: 'Normal',
      kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
    };
    return ctx.sendWorker({
      type: 'sampleN', ir: stdNormalIR, count: N, repeat: K,
      refArrays: {}, seed: nameSeed(name, ctx.rootSeed),
    }).then((reply: any) => {
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
      const kwargs: Record<string, any> = {};
      for (const pn of pnames) {
        kwargs[pn] = { kind: 'lit', value: elemAt(srcByParam[pn], jj) };
      }
      const distIR = { kind: 'call', op: d.distOp, kwargs: kwargs };
      return ctx.sendWorker({
        type: 'sampleN', ir: distIR, count: N,
        refArrays: {},
        seed: nameSeed(name + ':' + jj, ctx.rootSeed),
      }).then((reply: any) => { cols[jj] = reply.samples; });
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

function matTuple(d: DerivationTuple, ctx: any) {
  // Positional analogue of record. Each element materialises
  // independently; combine into a tuple Measure whose components live
  // in elems. Top-level logWeights is the join of components'
  // (propagateLogWeights handles dedupe + sum).
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
  // equals the sum of parents' totalmasses (logSumExp of their
  // logTotalmasses); resampling produces equally-weighted atoms each
  // carrying (totalInputMass / N) of mass.
  return Promise.all(d.fromNames.map(ctx.getMeasure)).then((parents: any[]) => {
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
  // selector is always realised (never marginalised); that is exactly
  // the kchain-sample vs kchain-density asymmetry.
  //
  // ifelse(c, a, b): branches=[a, b], selector c ∈ {0,1}; c truthy
  // (Bernoulli success) ⇒ branch 0 (a), else branch 1 (b). For K>2
  // (explicit Categorical mixture, later) the selector is a 1-based
  // index; clamp defensively.
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
  // Each branch is a named measure binding ({ ref }) or an inline
  // sampleable-distribution leaf ({ ir }); sample the latter directly
  // via the worker (the matSample path) — no binding to fetch.
  const branchP = branchEntries.map((b: any, bi: any) => {
    if (b && b.ref != null) return ctx.getMeasure(b.ref);
    return collectRefArrays(b.ir, ctx.fixedValues, ctx.getMeasure)
      .then((refArrays) => ctx.sendWorker({
        type: 'sampleN', ir: b.ir, count: ctx.sampleCount,
        refArrays: refArrays, seed: nameSeed(name + ':b' + bi, ctx.rootSeed),
      }))
      .then((reply) => scalarMeasureN(reply.samples, {
        logWeights: reply.logWeights || null, logTotalmass: 0,
        n_eff: reply.samples.length }));
  });
  const want = branchP.concat([ctx.getMeasure(d.selectorRef)]);
  return Promise.all(want).then((ms) => {
    const sel = empirical.materialiseUniform(ms[ms.length - 1]).samples;
    const branches = ms.slice(0, ms.length - 1)
      .map((m) => empirical.materialiseUniform(m).samples);
    const K = branches.length;
    const N = ctx.sampleCount;
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let k;
      if (K === 2) {
        // ifelse: success/true (selector ≠ 0) ⇒ branch 0.
        k = sel[i] ? 0 : 1;
      } else {
        // Categorical index into the branch list. Base is 1 for
        // Categorical (spec 1-based), 0 for Categorical0.
        const base = (d.selectorBase != null) ? d.selectorBase : 1;
        k = (sel[i] | 0) - base;
      }
      if (k < 0) k = 0; else if (k >= K) k = K - 1;
      out[i] = branches[k][i];
    }
    // ifelse / mixture of probability measures is a probability
    // measure (uniform atoms, unit mass).
    return scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

function matBayesupdate(d: DerivationBayesupdate, ctx: any) {
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
  const valueRefs: string[] = [];
  orchestrator.collectSelfRefs(bodyIR).forEach((n: any) => {
    if (isFunctionLikeBinding(ctx.bindings && ctx.bindings.get(n))) return;
    valueRefs.push(n);
  });
  return Promise.all([ctx.getMeasure(d.from)].concat(valueRefs.map(ctx.getMeasure)))
    .then((arr) => {
      const parent = arr[0];
      const refMeasures = arr.slice(1);
      const refArrays: any = {};
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
      }).then((reply: any) => {
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
function setBoundsForMat(setDescr: any) {
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
function matTruncate(d: DerivationTruncate, ctx: any) {
  return ctx.getMeasure(d.from).then((parent: any) => {
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
        }).then((reply: any) => scalarMeasureN(reply.samples, {
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
          let logWeights: Float64Array | null = null;
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
      const acceptedWeights: number[] = [];
      const totalWeights: number[] = [];
      const outWnn = outW!;
      for (let i = 0; i < parentSamples.length; i++) {
        const x = parentSamples[i];
        totalWeights.push(parent.logWeights[i]);
        if (x >= lo && x <= hi) {
          out[i] = x;
          outWnn[i] = parent.logWeights[i];
          acceptedWeights.push(parent.logWeights[i]);
          n_eff++;
        } else {
          out[i] = NaN;
          outWnn[i] = -Infinity;
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
    let logWeights: Float64Array | null = null;
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

function matTotalmass(d: DerivationTotalmass, ctx: any) {
  // totalmass(M): per spec §06, scalar mass of a (possibly
  // unnormalized) measure. The orchestrator tracks each measure's
  // logTotalmass through every materialisation step (algebraic
  // propagation for joint / iid / weighted / superpose / normalize,
  // empirical logSumExp for reweighted measures). totalmass(M)
  // exposes that as a per-atom scalar value — broadcast since we
  // track a single ensemble logTotalmass per measure today; per-atom
  // tracking is a separate refinement.
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

function matMvNormal(name: string, d: DerivationMvNormal, ctx: any) {
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
  const valueOps = require('./value-ops.ts');
  const sampler  = require('./sampler.ts');
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
    return Promise.reject(new Error('MvNormal: ' + (err as any).message));
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
  }).then((reply: any) => {
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

// =====================================================================
// matDirichlet — atom-batched probability-simplex draws
// =====================================================================
//
// Spec §08: Dirichlet(alpha = [a_1, ..., a_K]). Atom is a length-K
// probability vector summing to 1. Per-coord rep: g_k ~ Gamma(a_k, 1)
// independently across coords, then x = g / sum(g) (l1-normalize).
//
// Implementation: K worker calls (one per coord) each drawing N gamma
// variates; combine into atom-major shape=[N, K] + l1-normalize per
// atom. Per-coord seeds are derived from `name + '|' + k` so a
// resample of just one coord doesn't shift the others.
//
// alpha must be atom-independent (literal vector or fixed-phase
// binding). Per-atom alpha (a vector that varies across N atoms) is
// deferred — it requires per-atom Gamma parameters, which the worker's
// sampleN doesn't broadcast through.
function matDirichlet(name: string, d: DerivationDirichlet, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const alphaIR = (distIR.kwargs && distIR.kwargs.alpha)
    || (distIR.args && distIR.args[0]);
  if (!alphaIR) {
    return Promise.reject(new Error('Dirichlet: requires alpha argument'));
  }
  const alphaVal = orchestrator.resolveIRToValue(alphaIR, ctx.bindings, ctx.fixedValues);
  if (alphaVal == null) {
    return Promise.reject(new Error('Dirichlet: cannot resolve alpha (per-atom alpha deferred)'));
  }
  const alphaValue = valueLib.asValue(alphaVal);
  if (alphaValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'Dirichlet: alpha must be a vector, got shape=' + JSON.stringify(alphaValue.shape)));
  }
  const K = alphaValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('Dirichlet: alpha must be non-empty'));
  }
  for (let k = 0; k < K; k++) {
    if (!(alphaValue.data[k] > 0)) {
      return Promise.reject(new Error(
        'Dirichlet: alpha[' + k + '] = ' + alphaValue.data[k] + ' must be positive'));
    }
  }
  const N = ctx.sampleCount;
  // K parallel worker calls — one Gamma(alpha_k, 1) per coord. Each
  // returns Float64Array(N).
  const promises: Promise<any>[] = [];
  for (let k = 0; k < K; k++) {
    const gammaIR = {
      kind: 'call', op: 'Gamma',
      kwargs: {
        shape: { kind: 'lit', value: alphaValue.data[k] },
        rate:  { kind: 'lit', value: 1 },
      },
    };
    promises.push(ctx.sendWorker({
      type: 'sampleN', ir: gammaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|' + k, ctx.rootSeed),
    }));
  }
  return Promise.all(promises).then((replies: any[]) => {
    // Stitch K Float64Array(N)s into atom-major shape=[N, K] + l1-norm.
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const v = replies[k].samples[i];
        data[i * K + k] = v;
        sum += v;
      }
      // sum > 0 with probability 1 for positive-shape Gammas, but be
      // defensive against numerical edge cases (tiny shape parameters
      // can return exact 0).
      const inv = sum > 0 ? 1 / sum : 0;
      for (let k = 0; k < K; k++) data[i * K + k] *= inv;
    }
    const result: any = { shape: [N, K], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matMultinomial — atom-batched K-vector count draws
// =====================================================================
//
// Spec §08: Multinomial(n, p) — n iid Categorical(p) draws aggregated
// into a length-K count vector (K = length(p)). Atom is shape=[K]
// non-negative integers summing to n.
//
// Implementation: per atom, draw n Categorical1(p) samples and count
// per category. The Categorical1 sampler is in REGISTRY (the spec-
// shape 1-indexed Categorical); we evaluate the IR once per atom
// because n + p are atom-independent. Per-atom shared rng state via
// nameSeed + per-atom interleaving.
//
// n and p must be atom-independent (literal / fixed-phase). Per-atom
// parameters deferred — same constraint as MvNormal / Dirichlet.
function matMultinomial(name: string, d: DerivationMultinomial, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const nIR = (distIR.kwargs && distIR.kwargs.n)
    || (distIR.args && distIR.args[0]);
  const pIR = (distIR.kwargs && distIR.kwargs.p)
    || (distIR.args && distIR.args[1]);
  if (!nIR || !pIR) {
    return Promise.reject(new Error('Multinomial: requires n and p arguments'));
  }
  const nVal = orchestrator.resolveIRToValue(nIR, ctx.bindings, ctx.fixedValues);
  const pVal = orchestrator.resolveIRToValue(pIR, ctx.bindings, ctx.fixedValues);
  if (typeof nVal !== 'number' || !Number.isInteger(nVal) || nVal < 0) {
    return Promise.reject(new Error('Multinomial: n must be a non-negative integer'));
  }
  const pValue = valueLib.asValue(pVal);
  if (pValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'Multinomial: p must be a vector, got shape=' + JSON.stringify(pValue.shape)));
  }
  const K = pValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('Multinomial: p must be non-empty'));
  }
  // Validate p: non-negative, normalises to 1.
  let pSum = 0;
  for (let k = 0; k < K; k++) {
    if (!(pValue.data[k] >= 0)) {
      return Promise.reject(new Error(
        'Multinomial: p[' + k + '] = ' + pValue.data[k] + ' must be non-negative'));
    }
    pSum += pValue.data[k];
  }
  if (!(pSum > 0)) {
    return Promise.reject(new Error('Multinomial: p must sum to > 0'));
  }
  const N = ctx.sampleCount;
  const n = nVal | 0;
  // Issue n*N Categorical draws (one worker call yielding a flat
  // Float64Array of length n*N), then aggregate per atom into K-vector
  // counts. Using Categorical (1-indexed) per spec — we subtract 1 for
  // the 0-based count-vector slot. Worker stays uniform: one large
  // sampleN call rather than N separate per-atom calls.
  const catIR: any = {
    kind: 'call', op: 'Categorical',
    kwargs: { p: { kind: 'lit-vec', value: Array.from(pValue.data) } },
  };
  // Re-inject p as a literal array of lits the worker can read.
  catIR.kwargs.p = {
    kind: 'call', op: 'vector',
    args: Array.from(pValue.data, (v) => ({ kind: 'lit', value: v })),
  };
  if (n === 0) {
    // Zero-count atom: every K-slot is 0. Skip the sample round-trip.
    const data = new Float64Array(N * K);
    const result: any = { shape: [N, K], data };
    return Promise.resolve(measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    }));
  }
  return ctx.sendWorker({
    type: 'sampleN', ir: catIR, count: N * n,
    refArrays: {},
    seed: nameSeed(name, ctx.rootSeed),
  }).then((reply: any) => {
    const draws = reply.samples; // Float64Array(N*n), each in 1..K
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      const base = i * n;
      for (let j = 0; j < n; j++) {
        const cat = (draws[base + j] | 0) - 1;  // 1-indexed → 0-indexed
        if (cat >= 0 && cat < K) data[i * K + cat] += 1;
      }
    }
    const result: any = { shape: [N, K], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matWishart — atom-batched n×n SPD matrix draws via Bartlett
// =====================================================================
//
// Spec §08: Wishart(nu, scale) with scale ∈ SPD(n) and nu > n - 1.
// Bartlett decomposition: A is n×n lower-triangular with
//   A[i,i] ~ sqrt(Chi²_{nu - i})   (1-indexed: chi² with nu - i + 1 dof
//                                   for slot i in 1..n; equiv. nu - i dof
//                                   for 0-indexed slot i in 0..n-1)
//   A[i,j] ~ Normal(0, 1)  for i > j
//   A[i,j] = 0             for i < j
// L = lower_cholesky(scale); W = (L A)(L A)^T.
//
// Storage: shape=[N, n, n] atom-major (row-major within each atom).
// Atom-independent nu and scale; per-atom params deferred (same
// constraint as MvNormal / Dirichlet).
//
// Chi²_k ≡ Gamma(shape=k/2, rate=1/2), so the diag entries are
// drawn via Gamma((nu - i)/2, 1/2) and then sqrt-ed.
function matWishart(name: string, d: DerivationWishart, ctx: any): Promise<EmpiricalMeasure> {
  return wishartCore(name, d.distIR, ctx, /*invert=*/false);
}

// InverseWishart(nu, scale): pushforward of Wishart(nu, inv(scale))
// through inv (the spec equivalence in TODO §08). Implementation
// runs the same Bartlett pipeline but inverts the cholesky factor +
// the final matrix; the inverse computation is a per-atom forward/
// back substitution.
function matInverseWishart(name: string, d: DerivationInverseWishart, ctx: any): Promise<EmpiricalMeasure> {
  return wishartCore(name, d.distIR, ctx, /*invert=*/true);
}

function wishartCore(name: string, distIR: any, ctx: any, invert: boolean): Promise<EmpiricalMeasure> {
  const valueOps = require('./value-ops.ts');
  const sampler  = require('./sampler.ts');
  const opLabel = invert ? 'InverseWishart' : 'Wishart';
  const nuIR = (distIR.kwargs && distIR.kwargs.nu)
    || (distIR.args && distIR.args[0]);
  const scaleIR = (distIR.kwargs && distIR.kwargs.scale)
    || (distIR.args && distIR.args[1]);
  if (!nuIR || !scaleIR) {
    return Promise.reject(new Error(`${opLabel}: requires nu and scale arguments`));
  }
  const nuVal = orchestrator.resolveIRToValue(nuIR, ctx.bindings, ctx.fixedValues);
  if (typeof nuVal !== 'number') {
    return Promise.reject(new Error(`${opLabel}: nu must be a number`));
  }
  const scaleVal = orchestrator.resolveIRToValue(scaleIR, ctx.bindings, ctx.fixedValues);
  if (scaleVal == null) {
    return Promise.reject(new Error(`${opLabel}: cannot resolve scale (per-atom scale deferred)`));
  }
  const scaleValue = Array.isArray(scaleVal) && scaleVal.length > 0 && Array.isArray(scaleVal[0])
    ? valueOps._nestedToValue(scaleVal)
    : valueLib.asValue(scaleVal);
  if (scaleValue.shape.length !== 2 || scaleValue.shape[0] !== scaleValue.shape[1]) {
    return Promise.reject(new Error(
      `${opLabel}: scale must be a square matrix, got shape=` + JSON.stringify(scaleValue.shape)));
  }
  const n = scaleValue.shape[0];
  if (!(nuVal > n - 1)) {
    return Promise.reject(new Error(
      `${opLabel}: nu (${nuVal}) must be > n - 1 = ${n - 1}`));
  }
  // For InverseWishart, the spec equivalence is pushfwd(inv, Wishart(nu,
  // inv(scale))). We invert `scale` up front to feed the same Bartlett
  // sampler, then invert the final matrix per atom (one back-substitution
  // each via the Cholesky factor of the sampled W).
  let scaleForBartlett: any;
  let LS: any;
  try {
    if (invert) {
      // L_inv_S = cholesky(inv(scale)). Numerically: compute cholesky of
      // scale then invert that L. inv(scale) = inv(L L^T) = (L^-T)(L^-1)
      // so cholesky(inv(scale)) is NOT L^-T (which is upper-triangular);
      // we instead use scale^-1 = (L^-T L^-1), reflect the equivalence as
      // running Bartlett with cholesky(inv(scale)) which gives a fresh
      // lower-triangular factor of the inverse. Cheapest: invert the
      // matrix and rechol it.
      scaleForBartlett = sampler._internal.ARITH_OPS.inv(scaleValue);
      LS = sampler._internal.ARITH_OPS.lower_cholesky(scaleForBartlett);
    } else {
      scaleForBartlett = scaleValue;
      LS = sampler._internal.ARITH_OPS.lower_cholesky(scaleValue);
    }
  } catch (err) {
    return Promise.reject(new Error(`${opLabel}: ` + (err as any).message));
  }
  const N = ctx.sampleCount;
  // Draw n batched Chi²_{nu - i} streams for the Bartlett diagonal.
  const diagPromises: Promise<any>[] = [];
  for (let i = 0; i < n; i++) {
    const gammaIR = {
      kind: 'call', op: 'Gamma',
      kwargs: {
        shape: { kind: 'lit', value: (nuVal - i) / 2 },
        rate:  { kind: 'lit', value: 0.5 },
      },
    };
    diagPromises.push(ctx.sendWorker({
      type: 'sampleN', ir: gammaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|diag|' + i, ctx.rootSeed),
    }));
  }
  // Off-diag below: n*(n-1)/2 standard normals per atom.
  const offDiagPerAtom = n * (n - 1) / 2;
  const offDiagP = offDiagPerAtom > 0 ? ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
    count: N * offDiagPerAtom,
    refArrays: {},
    seed: nameSeed(name + '|offdiag', ctx.rootSeed),
  }) : Promise.resolve({ samples: new Float64Array(0) });
  return Promise.all([Promise.all(diagPromises), offDiagP]).then(([diagReplies, offDiagReply]: any[]) => {
    const data = new Float64Array(N * n * n);
    const A = new Float64Array(n * n);
    const T = new Float64Array(n * n);
    const W = new Float64Array(n * n);
    const offDiagData = offDiagReply.samples;
    const LSdata = LS.data;
    for (let atom = 0; atom < N; atom++) {
      // Build A (lower-tri Bartlett factor).
      for (let i = 0; i < n * n; i++) A[i] = 0;
      for (let i = 0; i < n; i++) {
        A[i * n + i] = Math.sqrt(diagReplies[i].samples[atom]);
      }
      let oIdx = atom * offDiagPerAtom;
      for (let i = 1; i < n; i++) {
        for (let j = 0; j < i; j++) A[i * n + j] = offDiagData[oIdx++];
      }
      // T = LS * A.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += LSdata[i * n + k] * A[k * n + j];
          T[i * n + j] = s;
        }
      }
      // W = T * T^T.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += T[i * n + k] * T[j * n + k];
          W[i * n + j] = s;
        }
      }
      const wBase = atom * n * n;
      if (invert) {
        // For InverseWishart we need inv(W). Cholesky of W exists (W
        // is SPD by construction), so invert via back-substitution.
        // Cheaper: invert the triangular T (W = T T^T ⇒ inv(W) =
        // inv(T)^T inv(T) = inv(T^T) inv(T)). T is lower-triangular,
        // so inv(T) is lower-triangular and computable by O(n²) back-
        // substitution per atom.
        // tinv = inv(T) (lower-triangular).
        const tinv = new Float64Array(n * n);
        for (let j = 0; j < n; j++) {
          // Solve T x = e_j → x is column j of tinv.
          for (let i = 0; i < n; i++) {
            if (i < j) { tinv[i * n + j] = 0; continue; }
            let s = (i === j) ? 1 : 0;
            for (let k = j; k < i; k++) s -= T[i * n + k] * tinv[k * n + j];
            tinv[i * n + j] = s / T[i * n + i];
          }
        }
        // W_inv = tinv^T * tinv (n×n symmetric).
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let k = 0; k < n; k++) s += tinv[k * n + i] * tinv[k * n + j];
            data[wBase + i * n + j] = s;
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) data[wBase + i * n + j] = W[i * n + j];
        }
      }
    }
    const result: any = { shape: [N, n, n], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matLKJCholesky — atom-batched n×n lower-triangular Cholesky factor
//                  of a correlation matrix
// =====================================================================
//
// Spec §08: LKJCholesky(n, eta) — onion-procedure sampler. eta > 0
// concentration parameter; eta=1 is the uniform distribution over
// correlation matrices. Returns L lower-triangular with unit-norm
// rows; L L^T is a correlation matrix.
//
// Onion sampling (Lewandowski-Kurowicka-Joe 2009): row-by-row growth
//   row 0 → (1)                                  // n=1: trivial
//   row 1 → (sin θ, cos θ) where sin θ ~ Beta-2  // n=2: 2D rotation
//   row i → (y, x · prev_unit) with
//             y² ~ Beta(α_i, α_i)·…  (concentration param ramp);
//             x = sqrt(1 - y²); prev_unit is uniform on S^{i-1}.
//
// Reference impl: Stan's onion sampler. Each row i (1-indexed) needs
//   - one beta draw: B_i ~ Beta(η + (n - i - 1)/2, (i)/2)        // for the radial squared distance
//   - i standard normals (then normalised → uniform on S^{i-1})
// then L[i, 0..i-1] = sqrt(B_i) · u_unit; L[i, i] = sqrt(1 - B_i);
// L[0, 0] = 1.
function matLKJCholesky(name: string, d: DerivationLKJCholesky, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const nIR   = (distIR.kwargs && distIR.kwargs.n)   || (distIR.args && distIR.args[0]);
  const etaIR = (distIR.kwargs && distIR.kwargs.eta) || (distIR.args && distIR.args[1]);
  if (!nIR || !etaIR) {
    return Promise.reject(new Error('LKJCholesky: requires n and eta arguments'));
  }
  const nVal   = orchestrator.resolveIRToValue(nIR,   ctx.bindings, ctx.fixedValues);
  const etaVal = orchestrator.resolveIRToValue(etaIR, ctx.bindings, ctx.fixedValues);
  if (typeof nVal !== 'number' || !Number.isInteger(nVal) || nVal < 1) {
    return Promise.reject(new Error('LKJCholesky: n must be a positive integer'));
  }
  if (typeof etaVal !== 'number' || !(etaVal > 0)) {
    return Promise.reject(new Error('LKJCholesky: eta must be a positive number'));
  }
  const n   = nVal | 0;
  const eta = etaVal;
  const N   = ctx.sampleCount;
  if (n === 1) {
    // Trivial: every atom is the 1×1 identity matrix.
    const data = new Float64Array(N);
    for (let i = 0; i < N; i++) data[i] = 1;
    return Promise.resolve(measureFromValue(
      { shape: [N, 1, 1], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N }));
  }
  // Per row i (1..n-1):
  //   beta_i ~ Beta(eta + (n - i - 1)/2, i/2)  [Stan parameterisation]
  //   z_{i,k} ~ Normal(0, 1) for k=0..i-1, then unit-normalise → u_i.
  // Build all beta and normal draws as separate worker batches.
  const betaPromises: Promise<any>[] = [];
  for (let i = 1; i < n; i++) {
    const a = eta + (n - i - 1) / 2;
    const b = i / 2;
    const betaIR = {
      kind: 'call', op: 'Beta',
      kwargs: { alpha: { kind: 'lit', value: a },
                beta:  { kind: 'lit', value: b } },
    };
    betaPromises.push(ctx.sendWorker({
      type: 'sampleN', ir: betaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|beta|' + i, ctx.rootSeed),
    }));
  }
  // Normals: for each row i, we need N*i standard normals. Bundle into
  // one big call (N * sum_{i=1..n-1} i = N * n*(n-1)/2 normals).
  const totalNormals = N * (n * (n - 1) / 2);
  const normalP = ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
    count: totalNormals,
    refArrays: {},
    seed: nameSeed(name + '|norm', ctx.rootSeed),
  });
  return Promise.all([Promise.all(betaPromises), normalP]).then(([betaReplies, normalReply]: any[]) => {
    const data = new Float64Array(N * n * n);
    const normals = normalReply.samples;
    let nIdx = 0;
    for (let atom = 0; atom < N; atom++) {
      const base = atom * n * n;
      // L[0, 0] = 1.
      data[base] = 1;
      for (let i = 1; i < n; i++) {
        const Bi = betaReplies[i - 1].samples[atom];
        const ri = Math.sqrt(Bi);              // radial: ‖row[0..i-1]‖
        const diag = Math.sqrt(1 - Bi);        // self-coord on the unit ball
        // Read i standard normals from the bundled stream, normalise.
        let zsq = 0;
        const z = new Float64Array(i);
        for (let k = 0; k < i; k++) {
          const v = normals[nIdx++];
          z[k] = v;
          zsq += v * v;
        }
        const invNorm = zsq > 0 ? 1 / Math.sqrt(zsq) : 0;
        for (let k = 0; k < i; k++) {
          data[base + i * n + k] = ri * z[k] * invNorm;
        }
        data[base + i * n + i] = diag;
      }
    }
    return measureFromValue(
      { shape: [N, n, n], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// matLKJ — atom-batched n×n correlation matrix (LKJCholesky · LKJCholesky^T)
// =====================================================================
//
// Pushforward of LKJCholesky through R = L L^T.
function matLKJ(name: string, d: DerivationLKJ, ctx: any): Promise<EmpiricalMeasure> {
  // Reuse matLKJCholesky for the L samples, then square per atom.
  const choleskyD: DerivationLKJCholesky = { kind: 'lkjcholesky', distIR: d.distIR };
  return matLKJCholesky(name, choleskyD, ctx).then((Lmeasure: any) => {
    const dims = Lmeasure.dims;
    if (!dims || dims.length !== 2 || dims[0] !== dims[1]) {
      return Promise.reject(new Error('LKJ: unexpected LKJCholesky output shape'));
    }
    const n = dims[0];
    const N = Lmeasure.samples.length / (n * n);
    const data = new Float64Array(N * n * n);
    const L = Lmeasure.samples;
    for (let atom = 0; atom < N; atom++) {
      const base = atom * n * n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          // L is lower-triangular: L[i, k] = 0 for k > i.
          const maxK = Math.min(i, j);
          for (let k = 0; k <= maxK; k++) s += L[base + i * n + k] * L[base + j * n + k];
          data[base + i * n + j] = s;
        }
      }
    }
    return measureFromValue(
      { shape: [N, n, n], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// matBinnedPoissonProcess — atom-batched K-vector of independent Poisson counts
// =====================================================================
//
// Spec §08: per-atom shape=[K] non-negative integers, each entry an
// independent Poisson draw with its own rate. Supports two surface
// forms today: rates as a K-vector (direct) or rates as a measure
// argument (deferred — would require integration of intensity over
// bins).
//
// Direct form: BinnedPoissonProcess(rates = [r_1, …, r_K]).
// One worker call per coord (K calls of Poisson(r_k), each yielding
// N integer counts); stitch into atom-major shape=[N, K].
function matBinnedPoissonProcess(name: string, d: DerivationBinnedPoissonProcess, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const ratesIR = (distIR.kwargs && distIR.kwargs.rates)
    || (distIR.args && distIR.args[0]);
  if (!ratesIR) {
    return Promise.reject(new Error(
      'BinnedPoissonProcess: requires rates argument (per-bin Poisson means)'));
  }
  const ratesVal = orchestrator.resolveIRToValue(ratesIR, ctx.bindings, ctx.fixedValues);
  if (ratesVal == null) {
    return Promise.reject(new Error('BinnedPoissonProcess: cannot resolve rates (per-atom rates deferred)'));
  }
  const ratesValue = valueLib.asValue(ratesVal);
  if (ratesValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'BinnedPoissonProcess: rates must be a vector, got shape=' + JSON.stringify(ratesValue.shape)));
  }
  const K = ratesValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('BinnedPoissonProcess: rates must be non-empty'));
  }
  for (let k = 0; k < K; k++) {
    if (!(ratesValue.data[k] >= 0)) {
      return Promise.reject(new Error(
        'BinnedPoissonProcess: rates[' + k + '] = ' + ratesValue.data[k] + ' must be non-negative'));
    }
  }
  const N = ctx.sampleCount;
  const promises: Promise<any>[] = [];
  // Per coord k: zero rate is the degenerate Poisson(0) — all atoms
  // are 0 a.s. Short-circuit so the leaf sampler doesn't reject the
  // boundary (stdlib's Poisson constructor disallows rate == 0).
  for (let k = 0; k < K; k++) {
    const rk = ratesValue.data[k];
    if (rk === 0) {
      promises.push(Promise.resolve({ samples: new Float64Array(N) }));
      continue;
    }
    const poissonIR = {
      kind: 'call', op: 'Poisson',
      kwargs: { rate: { kind: 'lit', value: rk } },
    };
    promises.push(ctx.sendWorker({
      type: 'sampleN', ir: poissonIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|bin|' + k, ctx.rootSeed),
    }));
  }
  return Promise.all(promises).then((replies: any[]) => {
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < K; k++) data[i * K + k] = replies[k].samples[i];
    }
    return measureFromValue(
      { shape: [N, K], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// Walk an expanded measure IR collecting every `select` node that
// still carries an unresolved runtime-weight spec (`weightsFrom`).
// engine-concepts §11: a non-closed-form selector condition yields a
// structurally-exact mixture whose per-branch weights must be
// estimated ONCE from the materialised selector ensemble. expandMeasureIR
// is pure and cannot reduce an ensemble, so it threads the spec
// through; the materialiser resolves it here. The estimate is a
// single scalar per branch (the selector is marginalised in the
// density), hence constant in the observation point — it fits the
// existing IR `logweights` slot exactly, just supplied at
// materialisation instead of by classify. Generic deep walk over all
// object/array child slots so it finds selects nested anywhere
// (branches, weighted/normalize wrappers, jointchain steps, …).
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

// Resolve every runtime-weight spec in `measureIR` into literal
// logweights, in place (the IR is freshly expanded per call). Each
// distinct selector ref is materialised once; the per-branch weight
// is the empirical frequency p̂_k of the selector landing in branch k
// (K==2 boolean ifelse: branch 0 = condition TRUE = selector truthy,
// matching matSelect's `sel?0:1` gather, so density and sampling
// agree). log 0 = −∞ is the correct weight for an unobserved branch
// (it drops out of the logsumexp mixture). No-op (and synchronous-
// equivalent) when there are no specs, so existing fast paths are
// unaffected. This is the general "materialiser resolves
// ensemble-reduction weights" capability — MC-weight ifelse is its
// first consumer; future ensemble-weighted selects reuse it
// unchanged.
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
        // Boolean ifelse: branch 0 when the {0,1} selector is truthy.
        let t = 0;
        for (let i = 0; i < Nn; i++) if (sel[i]) t++;
        cnt[0] = t; cnt[1] = Nn - t;
      } else {
        // General K-way: branch (sel − base), clamped (matSelect uses
        // the same clamped gather, so weights and gather stay aligned).
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

function matLogdensityof(d: DerivationLogdensityof, ctx: any) {
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
  // We detect a kchain (marginalising) measure via the first-class
  // derivation (kind:'jointchain' with marginalize:true) and reduce
  // the per-atom output to this scalar, broadcast to N for shape
  // consistency with the per-atom convention. n_eff collapses to 1 —
  // there's only one estimator here, even though it's built from N
  // prior samples. expandMeasureIR returns just the last step's
  // measure (the prior variate as a per-atom ref), so the
  // logsumexp−logN reduction marginalizes the prior out.
  const measureDeriv = ctx.derivations[d.measureName];
  const isChain = !!(measureDeriv
    && measureDeriv.kind === 'jointchain' && measureDeriv.marginalize);

  const measureIR0 = orchestrator.expandMeasureIR(d.measureName, ctx.derivations, undefined, ctx.bindings);
  if (!measureIR0) {
    return Promise.reject(new Error('logdensityof: cannot expand measure "'
      + d.measureName + '" into a self-contained IR'));
  }
  // Resolve any non-closed-form selector weights (engine-concepts §11
  // MC-weight selector) from their materialised ensembles before
  // density evaluation. No-op for closed-form / weight-free measures.
  return resolveRuntimeWeights(measureIR0, ctx).then((measureIR) => {
  const valueRefs: string[] = [];
  const fixedRefs: any[] = [];
  orchestrator.collectSelfRefs(measureIR).forEach((n: any) => {
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
  // N-ary kchain density (engine-concepts §6 chain-associativity).
  // expandMeasureIR returned ONLY the final kernel body with its hole
  // rewired to the cat over the (n−1)-joint history's variate names
  // (the deterministic positional `s{i}` convention shared with
  // derivations.js). Materialise that retained joint history ONCE via
  // matJointchain (N-ary retain SAMPLING is already complete) and
  // bind its variate columns as the per-atom refArrays the cat
  // consumes — the joint draw is preserved (same atom index across
  // columns), so the existing logsumexp−logN reduction below is the
  // exact MC marginal, just over a joint prior instead of a scalar.
  const naryKchain = isChain && measureDeriv
    && Array.isArray(measureDeriv.steps) && measureDeriv.steps.length > 2
    && !measureDeriv.labels;
  const innerJointP = naryKchain
    ? matJointchain(d.measureName + '$jchist',
        { kind: 'jointchain', marginalize: false, labels: null,
          steps: measureDeriv.steps.slice(0, -1) }, ctx)
    : Promise.resolve(null);
  return Promise.all([
    Promise.all(valueRefs.map(ctx.getMeasure)),
    innerJointP,
  ]).then(([refMeasures, innerJoint]: [any[], any]) => {
    const refArrays: any = {};
    for (let i = 0; i < valueRefs.length; i++) {
      const rm = refMeasures[i];
      if (!rm || !rm.samples || !rm.samples.BYTES_PER_ELEMENT) {
        throw new Error('logdensityof: ref "' + valueRefs[i] +
          '" did not materialise to a scalar EmpiricalMeasure');
      }
      refArrays[valueRefs[i]] = rm.samples;
    }
    if (innerJoint) {
      // Positional retain history ⇒ a tuple measure (.elems) in step
      // order; column i is variate `s{i}` (matches expandMeasureIR's
      // vname). Record history (.fields) ordered the same way.
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
    // Push fixed-phase refs to the worker session env so the density
    // walker (which evaluates leaf params against baseEnv) can resolve
    // them. setEnv with merge=true so we don't clobber any
    // host-provided session env.
    let setEnvP = Promise.resolve();
    if (fixedRefs.length > 0) {
      const fixedEnv: Record<string, any> = {};
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
  });
}

function matBroadcastLogdensity(d: DerivationBroadcastLogdensity, ctx: any) {
  // broadcast(logdensityof, M, pts) = [logdensityof(M, p) ∀ p∈pts].
  //
  // FAST PATH (engine-concepts §11/§12; flatppl-js is the EAGER
  // engine): when M is a self-contained tractable measure — no
  // per-atom prior refs, not a kchain MC-marginal — its log-density
  // is a closed-form function of the point, so evaluate it at ALL
  // points in ONE batched, sampling-free pass: the density spine's
  // N-axis indexes the evaluation points (`pointsBatched`). This
  // reuses walkSelect/walkWeighted/walkNormalize/walkLeaf verbatim and
  // is bit-identical to the per-point reference (same logpdf
  // catalogue + params) — the closed-form broadcast(logdensityof,…)
  // tests are the regression oracle. Otherwise (kchain marginal, or
  // M with per-atom prior refs where the points axis would collide
  // with the prior-atom axis) fall back to the per-point reference
  // route (correctness over speed). Result is a value array
  // (length = #points), same shape as a static `array` binding — the
  // plot panel renders it as an index/value curve.
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
  const measureIR0 = orchestrator.expandMeasureIR(
    d.measureName, ctx.derivations, undefined, ctx.bindings);
  // Resolve non-closed-form selector weights (engine-concepts §11)
  // from their materialised ensembles. The resulting literal weights
  // are constants, so a runtime-weight select stays closed-form-
  // batchable (it then takes the fast points-batched pass below).
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
// Top-level dispatch
// =====================================================================

const KIND_HANDLERS = {
  alias:        (name: any, d: any, ctx: any) => matAlias(d, ctx),
  sample:       (name: any, d: any, ctx: any) => matSample(name, d, ctx),
  evaluate:     (name: any, d: any, ctx: any) => matEvaluate(d, ctx),
  array:        (name: any, d: any) =>      matArray(d),
  weighted:     (name: any, d: any, ctx: any) => matWeighted(d, ctx),
  normalize:    (name: any, d: any, ctx: any) => matNormalize(d, ctx),
  iid:          (name: any, d: any, ctx: any) => matIid(name, d, ctx),
  kernelbroadcast: (name: any, d: any, ctx: any) => matKernelBroadcast(name, d, ctx),
  broadcast_logdensity: (name: any, d: any, ctx: any) => matBroadcastLogdensity(d, ctx),
  tuple:        (name: any, d: any, ctx: any) => matTuple(d, ctx),
  record:       (name: any, d: any, ctx: any) => matRecord(d, ctx),
  superpose:    (name: any, d: any, ctx: any) => matSuperpose(name, d, ctx),
  select:       (name: any, d: any, ctx: any) => matSelect(name, d, ctx),
  bayesupdate:  (name: any, d: any, ctx: any) => matBayesupdate(d, ctx),
  logdensityof: (name: any, d: any, ctx: any) => matLogdensityof(d, ctx),
  totalmass:    (name: any, d: any, ctx: any) => matTotalmass(d, ctx),
  truncate:     (name: any, d: any, ctx: any) => matTruncate(d, ctx),
  pushfwd:      (name: any, d: any, ctx: any) => matPushfwd(name, d, ctx),
  mvnormal:     (name: any, d: any, ctx: any) => matMvNormal(name, d, ctx),
  dirichlet:    (name: any, d: any, ctx: any) => matDirichlet(name, d, ctx),
  multinomial:  (name: any, d: any, ctx: any) => matMultinomial(name, d, ctx),
  wishart:           (name: any, d: any, ctx: any) => matWishart(name, d, ctx),
  inversewishart:    (name: any, d: any, ctx: any) => matInverseWishart(name, d, ctx),
  lkjcholesky:       (name: any, d: any, ctx: any) => matLKJCholesky(name, d, ctx),
  lkj:               (name: any, d: any, ctx: any) => matLKJ(name, d, ctx),
  binnedpoissonprocess: (name: any, d: any, ctx: any) => matBinnedPoissonProcess(name, d, ctx),
  // jointchain/kchain first-class kind (the only path; legacy
  // inlineChainOps deleted). matJointchain samples the base then
  // applies each kernel step structurally; marginalize ⇒ keep only
  // the last step's variate(s).
  jointchain:   (name: any, d: any, ctx: any) => matJointchain(name, d, ctx),
};

function matJointchain(name: string, d: DerivationJointchain, ctx: any) {
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
      }).then((r: any) => measureFromReply(r, ctx.sampleCount,
        { logTotalmass: 0, n_eff: ctx.sampleCount }));

  return Promise.resolve(baseP).then((M0) => {
    // priorVars: ordered [{ name, m }] of every scalar variate so far;
    // byName for auto-splat refArray binding. baseRefName lets a hole
    // param bind to the single base prior (scalar base).
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

    // Resolve a kernel step → { params, leaves:[{ name, ir }] }. Body
    // is expanded through refs (the closure walk); a record/joint body
    // contributes one leaf per field, a leaf-dist body one unnamed
    // leaf. One nesting level (the obs_dist=joint(y=Normal) shape);
    // deeper composites are a clean follow-up.
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
        body = orchestrator.expandMeasureIR(
          body.name, ctx.derivations, undefined, ctx.bindings);
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

    // Bind a leaf-dist IR's kernel params to prior variates and return
    // { ir, refArrays }. NAMED param matching a prior var ⇒ refArray
    // by that name (auto-splat). A lone HOLE param (single, unmatched)
    // ⇒ rewire to the prior cat: ref(p0) for one prior, vector(p0,…)
    // for ≥2; bind each via a synthetic scalar refArray.
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
        if (params.length !== 1) return null;     // multi-param + hole
        // Hole binds to the whole prior cat (all variates so far; for
        // a scalar base that's the base itself).
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
          p = p.then((acc: any[]) => {
            const vn = leaf.name != null ? leaf.name : ('s' + i);
            // Composite kernel-body field: an `iid(D, n)` field (e.g.
            // obs_dist = joint(obs = iid(Normal(a, b), 10))). Reuse
            // matIid's worker mechanism — sampleN of the inner leaf
            // with repeat=n — so the iid draws share the per-atom
            // parameter context (a, b are closure ancestors of the
            // prior; collectRefArrays threads them via getMeasure,
            // consistently with the cached prior draws since prior
            // field names == kernel param names == draw binding
            // names). Worker-RNG path, same as every other jointchain
            // leaf — no in-process traceeval / RNG split.
            if (leaf.ir && leaf.ir.kind === 'call'
                && leaf.ir.op === 'iid'
                && Array.isArray(leaf.ir.args) && leaf.ir.args.length === 2) {
              let inner = leaf.ir.args[0];
              if (inner && inner.kind === 'ref' && inner.ns === 'self') {
                inner = orchestrator.leafSampleIR(inner.name, ctx.derivations)
                  || inner;
              }
              // iid count: a fixed-phase scalar (literal, or a
              // fixed-binding expr the deterministic evaluator can
              // compute against fixedValues).
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
                .then((extra) => ctx.sendWorker({
                  type: 'sampleN', ir: bnd.ir, count: N, repeat: reps,
                  refArrays: Object.assign({}, extra, bnd.refArrays),
                  seed: nameSeed(name + ':jc' + i + '$' + li, ctx.rootSeed),
                })).then((reply) => {
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
      // Variate shape mirrors the kernel/base structure. Unwrap to a
      // bare scalar measure ONLY for a lone SYNTHETIC-named scalar
      // (scalar-base kchain / hole-kernel scalar body). A lone NAMED
      // field (e.g. kchain whose kernel body is joint(obs=…)) keeps
      // its record shape {obs} — the kernel's variate is that record
      // (spec: kchain keeps the last kernel's variate). Multiple /
      // named ⇒ record; positional all-synthetic-scalar ⇒ tuple.
      if (outPairs.length === 1
          && /^s\d+$/.test(outPairs[0].name)
          && outPairs[0].m && outPairs[0].m.samples
          && !outPairs[0].m.fields) {
        return Object.assign({}, outPairs[0].m,
          { logTotalmass: 0, n_eff: nEff });
      }
      const named = !!d.labels || (M0 && M0.fields)
        || outPairs.some((x) => !/^s\d+$/.test(x.name));
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
  // Value ↔ Measure bridges.
  valueOf,
  measureFromValue,
};
