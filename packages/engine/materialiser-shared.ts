'use strict';

// Shared helpers for the per-binding materialiser. Owns the
// EmpiricalMeasure shape builders (scalarMeasureN, measureFromValue,
// measureFromReply), the refArrays plumbing trio
// (measureToRefValue / collectRefArrays / prepareDensityRefs /
// pushFixedEnv — engine-concepts §17.1), the fixed-phase →
// EmpiricalMeasure adapter (fixedValueToMeasure), the structured-set
// → numeric-bounds map (setBoundsForMat), and the small RNG /
// function-binding utilities every mat* handler uses.
//
// See engine/materialiser.ts for the per-kind handlers and
// KIND_HANDLERS dispatch.

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

/**
 * Convert a single materialised parent measure into the Value the
 * worker's refArrays / per-atom evaluator expects. Shared by every
 * site that builds a refArrays entry from a parent's Measure — keeps
 * the (scalar-leaf vs vector-leaf) dispatch in one place so it can't
 * drift between callers.
 *
 * Scalar-leaf measures (`.samples` Float64Array(N)) wrap as a Value
 * shape=[N]; vector-leaf measures already carry the Value shape=[N, k]
 * (or higher rank) in `.value`. Hand-crafted Measure inputs in tests
 * that omit `.value` fall through the Float64Array branch via
 * batchedScalar so they keep working unchanged.
 */
function measureToRefValue(m: any, name: string, label: string) {
  if (m == null) {
    throw new Error(label + ': measure for "' + name + '" is null/undefined');
  }
  if (m.value) return m.value;
  if (m.samples && m.samples.BYTES_PER_ELEMENT !== undefined) {
    return valueLib.batchedScalar(m.samples);
  }
  throw new Error(label + ': measure for "' + name +
    '" has neither .value nor .samples');
}

/**
 * Resolve every value-position self-ref in `ir` to its parent's
 * Value, returning a refName→Value map for the worker primitives'
 * refArrays. Fixed-phase refs are dropped: they flow through the
 * worker's session env, NOT refArrays (which would try to slice them
 * per-atom and feed the worker an undefined entry).
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
      out[names[i]] = measureToRefValue(measures[i], names[i], 'collectRefArrays');
    }
    return out;
  });
}

/**
 * Single owner of the (collectSelfRefs → filter → split → materialise
 * → build refArrays + fixedEnv) plumbing the worker-bound density /
 * evaluate paths need. Replaces the bespoke per-call-site logic in
 * matBayesupdate / matLogdensityof / matEvaluate that all drifted
 * subtly (scalar-only guards, missing setEnv for fixedRefs, missing
 * built-in-name gates).
 *
 * Filters:
 *  - function-like bindings (fn / functionof / kernelof / bijection) —
 *    these are consulted by name at density / sample dispatch time;
 *    they don't materialise as values.
 *  - refs that name neither a binding nor a fixed value — built-in
 *    distribution kernel names (`Normal`, `MvNormal`, …) and synthetic
 *    joint step-variate names introduced by expandMeasureIR's
 *    jointchain canonicalisation. Density-walker env-threading or
 *    closed-form classification owns those; they must not become
 *    refArrays entries (getMeasure would fail with 'no derivation').
 *
 * Returns:
 *  - `refArrays`   — { name → Value } for per-atom parent measures.
 *  - `fixedEnv`    — { name → JS value } for fixed-phase refs. The
 *    caller pushes this via `setEnv merge:true` before its
 *    logDensityN / evaluateN message; helpers fixedRefsToSetEnv /
 *    pushFixedEnv expose the canonical pattern.
 */
function prepareDensityRefs(ir: any, ctx: any, label: string) {
  const refs = orchestrator.collectSelfRefs(ir);
  const perAtomNames: string[] = [];
  const fixedEnv: Record<string, any> = {};
  refs.forEach((n: any) => {
    if (isFunctionLikeBinding(ctx.bindings && ctx.bindings.get(n))) return;
    const isBinding = !!(ctx.bindings && ctx.bindings.has(n));
    const isFixed   = !!(ctx.fixedValues && ctx.fixedValues.has(n));
    if (!isBinding && !isFixed) return;
    if (isFixed) {
      fixedEnv[n] = ctx.fixedValues.get(n);
      return;
    }
    perAtomNames.push(n);
  });
  return Promise.all(perAtomNames.map(ctx.getMeasure)).then((measures: any[]) => {
    const refArrays: Record<string, any> = {};
    for (let i = 0; i < perAtomNames.length; i++) {
      refArrays[perAtomNames[i]] =
        measureToRefValue(measures[i], perAtomNames[i], label);
    }
    return { refArrays, fixedEnv, perAtomNames };
  });
}

/**
 * Push a `fixedEnv` map onto the worker session env via setEnv merge.
 * Returns a Promise that resolves once the merge has been applied;
 * empty maps short-circuit to Promise.resolve(). Companion to
 * prepareDensityRefs.
 */
function pushFixedEnv(ctx: any, fixedEnv: Record<string, any>) {
  for (const _k in fixedEnv) {
    return ctx.sendWorker({ type: 'setEnv', env: fixedEnv, merge: true });
  }
  return Promise.resolve();
}

/**
 * Convert a JS value (from the orchestrator's fixedValues map) to a
 * Measure record. Scalars broadcast across N atoms; numeric arrays
 * are flat samples; records and tuples recurse field-/element-wise.
 * Returns null for shapes we can't surface (rngstate, opaque
 * objects); the caller falls through to derivation-based dispatch.
 */
function fixedValueToMeasure(v: any, sampleCount: any): any {
  if (typeof v === 'number' && Number.isFinite(v)) {
    const arr = new Float64Array(sampleCount);
    arr.fill(v);
    return scalarMeasureN(arr,
      { logWeights: null, logTotalmass: 0, n_eff: sampleCount });
  }
  // Shape-explicit Value (e.g. produced by `eye(n)`, `diagmat`,
  // transpose / lower_cholesky chains). A fixed-phase Value is ONE
  // value, not an atom-batched measure: shape=[3, 3] means "a single
  // 3x3 matrix", not "3 atoms of 3-vector". Flatten to a 1D scalar
  // measure (length = prod(shape)) so the viewer's array-mode step
  // plot renders the values. A proper rank-2+ heatmap renderer would
  // preserve the matrix structure visually — tracked as a follow-up.
  if (valueLib.isValue(v)) {
    const dense = valueLib.densify(v);
    const flat = dense.data instanceof Float64Array
      ? dense.data
      : Float64Array.from(dense.data);
    const m: any = scalarMeasureN(flat, {
      logWeights: null, logTotalmass: 0, n_eff: flat.length,
    });
    if (dense.shape.length >= 2) m.intrinsicShape = dense.shape.slice();
    return m;
  }
  // Complex scalar `z = complex(re, im)` → planar complex Value of shape=[1].
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
    let allScalar = v.length > 0;
    for (let i = 0; allScalar && i < v.length; i++) {
      const t = typeof v[i];
      if (t === 'boolean') continue;
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
    // Per spec §03: nested JS array reaching this branch is genuinely
    // vector-of-vectors. Recurse element-wise into the elems tuple.
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
function measureN(m: any): number {
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
 * per-atom samples plus the metadata fields.
 */
function scalarMeasureN(samples: any, extras: any) {
  return measureFromValue(valueLib.batchedScalar(samples), extras);
}

/**
 * Build a Measure record from a Value plus the measure-metadata fields.
 * `v.data` becomes both `.samples` (legacy view) and `.value.data`
 * (shared storage; no copy).
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
  if (dims) {
    m.dims = dims;
    m.shape = 'array';
  }
  if (valueLib.isComplexValue(v)) {
    m.imag = v.im;
    m.dtype = 'complex';
  }
  return m;
}

/**
 * Build a Measure from a worker `evaluateN` reply, transparently
 * handling the real / complex and scalar-atom / vector-atom cases.
 */
function measureFromReply(reply: any, count: any, extras: any) {
  const dims = reply.dims;
  const shape = dims ? [count | 0].concat(dims) : [reply.samples.length];
  const v = reply.imag
    ? valueLib.complexValue(reply.samples, reply.imag, shape)
    : { shape: shape, data: reply.samples };
  return measureFromValue(v, extras);
}

/**
 * Map a structural set descriptor (orchestrator.parseSetIR shape) to
 * numeric [lo, hi] bounds. Returns null for set kinds the materialiser
 * doesn't yet support (integers / booleans).
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
 * Resolve a function binding (fn / functionof / kernelof / bijection)
 * to its callable body + single-param name. Bijection bindings carry
 * their `f` component in `binding.bijection.fName`; we follow that ref
 * to the actual function binding's body. Returns { body, paramName }
 * or null when the binding isn't function-shaped.
 */
function resolveFnBody(binding: any, bindings: any): any {
  if (!binding) return null;
  if (binding.type === 'bijection') {
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

module.exports = {
  nameSeed,
  makeMainThreadPrng,
  isFunctionLikeBinding,
  measureToRefValue,
  collectRefArrays,
  prepareDensityRefs,
  pushFixedEnv,
  fixedValueToMeasure,
  measureN,
  valueOf,
  scalarMeasureN,
  measureFromValue,
  measureFromReply,
  setBoundsForMat,
  resolveFnBody,
};
