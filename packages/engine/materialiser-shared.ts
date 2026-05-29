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

// Callable-layer bindings (fn / functionof / kernelof / bijection /
// fchain — engine-concepts §19.3) aren't materialisable as values —
// they're consulted by name at density / sample dispatch time.
// matLogdensityof / matBayesupdate / classifyProfileSelfRefs use this
// to filter such refs out of their "materialise the parents" pre-pass;
// the parents that ARE values still need materialising.
//
// Phase 2 will add `'kernel-chain'` (kernel-first jointchain/kchain)
// to this set; at that point the by-tag enumeration here is
// supplemented (not replaced) by the type-driven predicate
// `isCallableLayerBinding(b)` which reads `b.inferredType.kind ∈
// {'func', 'kernel'}` — same answer, two source-of-truth angles.
function isFunctionLikeBinding(binding: any) {
  if (!binding) return false;
  switch (binding.type) {
    case 'fn':
    case 'functionof':
    case 'kernelof':
    case 'bijection':
    case 'fchain':
      return true;
    default:
      return false;
  }
}

/**
 * Type-driven callable-layer predicate (engine-concepts §19.2). Reads
 * `inferredType.kind ∈ {'function', 'kernel'}` — function-layer and
 * kernel-layer bindings sit in the "callable layer" (consulted by
 * name at dispatch time, not materialised as a value or measure).
 *
 * Use this when the caller has `inferredType` available (post-
 * typeinfer, which is all materialiser / sampler / viewer paths).
 * Use `isFunctionLikeBinding(binding)` when only the producer-tag is
 * available (pre-typeinfer in the analyzer). Both predicates should
 * agree on every binding by construction; this is a property test
 * worth adding.
 */
function isCallableLayerBinding(binding: any): boolean {
  if (!binding || !binding.inferredType) return false;
  const k = binding.inferredType.kind;
  return k === 'function' || k === 'kernel';
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
function collectRefArrays(ir: any, fixedValuesOrCtx: any, getMeasure?: any) {
  // Two call shapes supported:
  //  - (ir, ctx)                                — preferred: auto-pushes
  //    fixed-phase refs into the worker session env via setEnv merge.
  //    The fixed-phase refs the IR mentions (e.g. max_r, N, K) reach
  //    the worker's eval context automatically — no caller needs to
  //    remember to pushFixedEnv first.
  //  - (ir, fixedValues, getMeasure)            — legacy: returns
  //    refArrays only; fixed-phase refs are silently dropped on the
  //    assumption the caller has already pushed them. Retained for
  //    back-compat; new callers should pass `ctx`.
  const isCtxForm = (fixedValuesOrCtx
    && typeof fixedValuesOrCtx === 'object'
    && typeof fixedValuesOrCtx.getMeasure === 'function');
  const fixedValues = isCtxForm ? fixedValuesOrCtx.fixedValues : fixedValuesOrCtx;
  const _getMeasure = isCtxForm ? fixedValuesOrCtx.getMeasure : getMeasure;
  const ctx        = isCtxForm ? fixedValuesOrCtx : null;

  const refs = orchestrator.collectSelfRefs(ir);
  const names: string[] = [];
  const fixedEnv: Record<string, any> = {};
  let anyFixed = false;
  refs.forEach((n: string) => {
    if (fixedValues && fixedValues.has(n)) {
      fixedEnv[n] = fixedValues.get(n);
      anyFixed = true;
      return;
    }
    names.push(n);
  });
  // ctx-form: pushFixedEnv first so the worker has these refs before
  // the caller's next sendWorker. Legacy form: skip the push (caller
  // is responsible for setEnv).
  const preFlight = (ctx && anyFixed)
    ? pushFixedEnv(ctx, fixedEnv)
    : Promise.resolve();
  return preFlight
    .then(() => Promise.all(names.map(_getMeasure)))
    .then((measures: any[]) => {
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
// Walk an IR expression and substitute every `self`-ref to a
// callable binding (fn / functionof / kernelof / bijection — see
// `isFunctionLikeBinding`) with the binding's inline `functionof` IR.
// Used at materialiser → worker hand-off so the worker's `_resolveFn`
// sees inline `functionof` shapes directly (the
// `_resolveFn` `self`-ref branch requires `env.__resolveFnBody`,
// which the worker's session env doesn't carry).
//
// Returns the (possibly new) IR; mutates nothing on the input.
// Bounded recursion — callable bindings can't form cycles per
// FlatPPL spec §04 (modules are DAGs).
function inlineCallableRefs(ir: any, bindings: any): any {
  if (!ir || typeof ir !== 'object') return ir;
  // Self-ref to a callable binding → splice in its `functionof` IR.
  if (ir.kind === 'ref' && ir.ns === 'self' && bindings && bindings.get) {
    const b = bindings.get(ir.name);
    if (isFunctionLikeBinding(b) && b.ir
        && b.ir.kind === 'call' && b.ir.op === 'functionof') {
      // Recurse into the inlined body too — a callable can reference
      // other callables. Cycles can't form (spec §04 DAG).
      return inlineCallableRefs(b.ir, bindings);
    }
    return ir;
  }
  if (ir.kind !== 'call') return ir;
  let changed = false;
  let newArgs: any[] | null = null;
  if (Array.isArray(ir.args)) {
    newArgs = new Array(ir.args.length);
    for (let i = 0; i < ir.args.length; i++) {
      const w = inlineCallableRefs(ir.args[i], bindings);
      newArgs[i] = w;
      if (w !== ir.args[i]) changed = true;
    }
  }
  let newKwargs: Record<string, any> | null = null;
  if (ir.kwargs && typeof ir.kwargs === 'object') {
    newKwargs = {};
    for (const k in ir.kwargs) {
      const w = inlineCallableRefs(ir.kwargs[k], bindings);
      newKwargs[k] = w;
      if (w !== ir.kwargs[k]) changed = true;
    }
  }
  let newBody = ir.body;
  if (ir.body) {
    newBody = inlineCallableRefs(ir.body, bindings);
    if (newBody !== ir.body) changed = true;
  }
  let newFields: any[] | null = null;
  if (Array.isArray(ir.fields)) {
    newFields = new Array(ir.fields.length);
    for (let i = 0; i < ir.fields.length; i++) {
      const f = ir.fields[i];
      const wv = inlineCallableRefs(f && f.value, bindings);
      newFields[i] = (wv === f.value) ? f : { ...f, value: wv };
      if (wv !== f.value) changed = true;
    }
  }
  if (!changed) return ir;
  const walked: any = { ...ir };
  if (newArgs) walked.args = newArgs;
  if (newKwargs) walked.kwargs = newKwargs;
  if (newBody !== ir.body) walked.body = newBody;
  if (newFields) walked.fields = newFields;
  return walked;
}

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
 * Pure / synchronous classifier sibling of `prepareDensityRefs` for the
 * profile-plot path. Returns the names of value-typed, random-phase
 * self-refs whose materialised parent measure the caller should fetch
 * to construct a single representative value for `fixedEnv`.
 *
 * The viewer's profile-plot path (`render-profile.ts`) needs the same
 * filtering as `prepareDensityRefs` — drop function-like bindings, drop
 * built-in distribution names and other unbound refs, drop fixed-phase
 * refs (already in the worker session env via setEnv) — but NOT the
 * async per-atom measure-fetch. `profileN` doesn't accept refArrays;
 * non-swept random-phase parents need a single first-cut value, not a
 * per-atom slice.
 *
 * Centralising the classification here keeps the filter rules in one
 * place. Without this, render-profile.ts duplicated the loop by hand
 * and got it wrong: feeding built-in names through `tryGetMeasure`,
 * and clobbering fixed-phase array values (`x_data = [...]`) with the
 * `samples[0]` of a materialised constant-atom measure.
 */
function classifyProfileSelfRefs(
  ir: any, bindings: any, fixedValues: any
): { perAtomNames: string[] } {
  const refs = orchestrator.collectSelfRefs(ir);
  const perAtomNames: string[] = [];
  refs.forEach((n: any) => {
    if (isFunctionLikeBinding(bindings && bindings.get(n))) return;
    if (fixedValues && fixedValues.has(n)) return;
    if (!bindings || !bindings.has(n)) return;
    perAtomNames.push(n);
  });
  return { perAtomNames };
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
    // engine-concepts §20.2 / TODO Phase 1: a fixed-phase scalar is a
    // rank-0 Value (`{shape: [], data: Float64Array([v])}`); broadcasting
    // against atom-batched operands happens lazily in valueOps via
    // stride-0 reads on the rank-0 side. The canonical `.value` field
    // is what consumers should read; `.samples` is kept as a length-N
    // buffer transitionally so legacy histogram / viewer / per-atom
    // consumers continue to see the broadcast form until they migrate
    // to `valueOf(m)` reads.
    const arr = new Float64Array(sampleCount);
    arr.fill(v);
    const m = scalarMeasureN(arr,
      { logWeights: null, logTotalmass: 0, n_eff: sampleCount });
    m.value = valueLib.scalar(v);
    m.rank0 = true;
    return m;
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
// Per-atom parameter-shape access for broadcast(K|Dist, …) and
// related N-atom × K-element materialiser paths.
//
// The shape contract in the engine's batched scalar/array storage
// (§2.1 leading-axis batch) admits four input shapes for a kernel
// parameter that's broadcast over K elements with N atoms each:
//
//   scalar             — same value at every (i, j)
//   shape=[K]          — varies along j (K elements), constant in i
//   shape=[N]          — varies along i (N atoms), constant in j
//   shape=[N, K]       — varies along both, atom-major row-major
//
// `elemScalarAtJ(v, j, N)` returns the value at element index j
// (the K-axis). Used when a parameter is atom-independent and the
// per-element scalar is enough to drive a single worker `sampleN`
// call. Size-1 axes broadcast by repetition.
//
// `perAtomColumnAtJ(v, j, N)` returns a length-N Float64Array
// holding the per-atom column at element index j. Used when a
// parameter genuinely varies per atom and needs to flow through
// the worker as a `refArrays` entry. Throws if the shape isn't
// one of the four supported above.
//
// These utilities used to live as closures inside `matKernelBroadcast`
// (mat-broadcast.ts); pulled here so other per-atom materialiser
// paths can share them and so the param-shape contract is in one
// documented place.
function elemScalarAtJ(v: any, j: number, _N: number): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (valueLib.isValue(v)) {
    const shape = v.shape;
    if (shape.length === 0) return v.data[0];
    if (shape.length === 1) {
      return v.data[shape[0] === 1 ? 0 : j];
    }
  }
  if (v && v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number') {
    return v[v.length === 1 ? 0 : j];
  }
  if (Array.isArray(v)) return +v[v.length === 1 ? 0 : j];
  throw new Error('broadcast: cannot scalarise atom-indep value at j=' + j);
}

function perAtomColumnAtJ(v: any, j: number, N: number): Float64Array {
  const col = new Float64Array(N);
  if (valueLib.isValue(v)) {
    const shape = v.shape;
    const data = v.data;
    if (shape.length === 1 && shape[0] === N) {
      col.set(data);
    } else if (shape.length === 2 && shape[0] === N) {
      const stride = shape[1];
      const off = stride === 1 ? 0 : j;
      for (let i = 0; i < N; i++) col[i] = data[i * stride + off];
    } else {
      throw new Error('broadcast: unexpected per-atom shape '
        + JSON.stringify(shape));
    }
  } else if (v && v.BYTES_PER_ELEMENT !== undefined && v.length === N) {
    col.set(v);
  } else {
    throw new Error('broadcast: per-atom param has unexpected type '
      + (typeof v));
  }
  return col;
}

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
  inlineCallableRefs,
  nameSeed,
  makeMainThreadPrng,
  isFunctionLikeBinding,
  isCallableLayerBinding,
  measureToRefValue,
  collectRefArrays,
  prepareDensityRefs,
  classifyProfileSelfRefs,
  pushFixedEnv,
  fixedValueToMeasure,
  measureN,
  valueOf,
  scalarMeasureN,
  measureFromValue,
  measureFromReply,
  setBoundsForMat,
  elemScalarAtJ,
  perAtomColumnAtJ,
  resolveFnBody,
};
