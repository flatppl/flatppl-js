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
 * Derive a per-binding Philox key from `rootKey` and the binding's
 * name. Gives each binding its own deterministic, well-mixed RNG
 * stream for sampleN — order-independent so two unrelated bindings
 * stay independent regardless of which the user materialises first.
 *
 * Migration history (commit 2, 2026-05-30): the old shape was an
 * FNV-1a-32 hash of `name` XOR'd against a single-uint32 rootSeed,
 * returning one uint32 the worker fed into `stateFromKey(seed, 0)` —
 * an unmixed key with all entropy in lane 0. We now thread a
 * full 64-bit Philox key (`[k0, k1]`) end-to-end: rootKey is built
 * once at the orchestrator boundary via `rng.keyFromSeed(rootSeed)`,
 * and per-binding keys come from `foldIn(rootKey, xxhash32(name))`.
 * xxhash32 has much better avalanche than FNV-1a-32 on short
 * structured names (the bulk of our `name` domain — '%select_ir:b'+bi,
 * 'name:jc'+i+'$'+li, etc.), and `foldIn` keeps the two-lane key
 * fully mixed.
 */
function nameSeed(name: string, rootKey: [number, number]): [number, number] {
  return rng.foldIn(rootKey, rng.xxhash32(name));
}

/**
 * Wrap a Philox state in a closure returning U(0,1) uniforms — the
 * shape empirical.systematicResample / multinomialResample expect.
 * One closure per call; the state mutates internally.
 *
 * Post-commit-2: takes a full `PhiloxKey` (`[k0, k1]`) rather than a
 * single uint32. The old single-arg `stateFromKey(seed)` call passed
 * `k1=undefined` (silently coerced to 0), producing a one-lane key.
 * We now spread both lanes explicitly so both halves of the Philox
 * key carry the caller's entropy.
 */
function makeMainThreadPrng(key: [number, number]) {
  let state = rng.stateFromKey(key[0], key[1]);
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
 * Convert a materialised measure to an atom-batched `[N, D]` Value for
 * use as a per-atom parameter (Phase 5.1 Session 5f-2). Handles three
 * shapes a vector-valued binding can take:
 *   - `m.value` (already a shape-tagged Value, e.g. an iid / pushfwd
 *     output `[N, D]`) — returned as-is.
 *   - composite `m.elems` (a tuple / array measure whose components are
 *     scalar measures, e.g. `muv = [m1, m2]`) — the per-atom component
 *     samples are STACKED into an atom-major `[N, D]` buffer
 *     (`data[n*D + d] = elems[d].samples[n]`), tagged `outerRank: 1`.
 *   - scalar `m.samples` ([N] batched scalar) — wrapped as `[N]`.
 *
 * Unlike `measureToRefValue`, this knows how to flatten a composite
 * vector variate into the contiguous `[N, D]` storage the bijection
 * registry's atom-batched param contract expects. Throws for shapes it
 * can't flatten (ragged / nested-composite elems).
 */
function measureToParamValue(m: any, name: string, label: string) {
  if (m == null) {
    throw new Error(label + ': param measure for "' + name + '" is null/undefined');
  }
  if (m.value && Array.isArray(m.value.shape)) return m.value;
  if (Array.isArray(m.elems) && m.elems.length > 0
      && m.elems.every((e: any) => e && e.samples
        && e.samples.BYTES_PER_ELEMENT !== undefined)) {
    const D = m.elems.length;
    const N = m.elems[0].samples.length;
    const data = new Float64Array(N * D);
    for (let d = 0; d < D; d++) {
      const s = m.elems[d].samples;
      if (s.length !== N) {
        throw new Error(label + ': ragged composite param measure for "'
          + name + '" (component ' + d + ' length ' + s.length + ' ≠ ' + N + ')');
      }
      for (let n = 0; n < N; n++) data[n * D + d] = s[n];
    }
    return { shape: [N, D], data, outerRank: 1 };
  }
  if (m.samples && m.samples.BYTES_PER_ELEMENT !== undefined) {
    return valueLib.batchedScalar(m.samples);
  }
  throw new Error(label + ': param measure for "' + name
    + '" is not a vector-valued shape this path can flatten to [N, D] '
    + '(needs .value, scalar-component .elems, or scalar .samples)');
}

/**
 * Convert a RECORD-shaped measure (`m.fields = {a, b, …}`, each a per-atom
 * sub-measure) into a plain JS array of `N` per-atom record objects:
 * `out[i] = { a: <atom-i of field a>, b: …, … }`.
 *
 * This is the density-side feeding shape for a reified kernel whose SINGLE
 * parametric input is a whole record (the transport `pars` case): the prior
 * is `record(a, b, mu)` and the kernel body field-accesses `pars.a` / `.b` /
 * `.mu`. Fed as `refArrays[<param>]`, the density per-atom accessor returns
 * `arr[i]` — the record object — so `get_field(pars, "a")` resolves per atom
 * (audit §3 / H1 record-param case). Unlike `measureToRefValue`, which only
 * handles scalar/Value leaves, this descends the record structure:
 *   - scalar field (`.samples`)         → `samples[i]` (a number)
 *   - nested record field (`.fields`)   → recurse (a per-atom object)
 *   - vector field (`.value`, rank ≥ 1) → the atom-`i` leading-axis slice
 *
 * Per-atom objects (not typed-array columns + IR substitution like the
 * sampling-side `matJointchain.bindLeaf`) are chosen for correctness and
 * generality: a whole-record reference, not just `record.field`, just works,
 * and structuredClone carries plain number objects across the worker
 * boundary. The cost is `N` small allocations — acceptable for a one-shot
 * posterior materialisation.
 */
function measureToPerAtomRecords(m: any, name: string, label: string): any[] {
  if (m == null || !m.fields) {
    throw new Error(label + ': record measure for "' + name
      + '" has no .fields to expand into per-atom objects');
  }
  const fnames = Object.keys(m.fields);
  // Per-field per-atom accessor, precomputed so the N-loop stays tight.
  const access = fnames.map((f) => {
    const fm = m.fields[f];
    if (fm && fm.samples && fm.samples.BYTES_PER_ELEMENT !== undefined) {
      const s = fm.samples;
      return (i: number) => s[i];
    }
    if (fm && fm.fields) {
      const sub = measureToPerAtomRecords(fm, name + '.' + f, label);
      return (i: number) => sub[i];
    }
    if (fm && fm.value && Array.isArray(fm.value.shape)) {
      const v = fm.value, shape = v.shape, data = v.data;
      if (shape.length === 1) return (i: number) => data[i];
      const tail = shape.slice(1);
      const tailLen = tail.reduce((a: number, b: number) => a * b, 1);
      return (i: number) => ({ shape: tail, data: data.subarray(i * tailLen, (i + 1) * tailLen) });
    }
    throw new Error(label + ': record field "' + name + '.' + f
      + '" is neither scalar (.samples), record (.fields), nor vector (.value)');
  });
  // N from the first scalar/Value field that exposes a length.
  let N = 0;
  for (const f of fnames) {
    const fm = m.fields[f];
    if (fm && fm.samples && typeof fm.samples.length === 'number') { N = fm.samples.length; break; }
    if (fm && fm.value && Array.isArray(fm.value.shape)) { N = fm.value.shape[0]; break; }
  }
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const o: Record<string, any> = {};
    for (let j = 0; j < fnames.length; j++) o[fnames[j]] = access[j](i);
    out[i] = o;
  }
  return out;
}

// Repeat each per-atom block `k` times into a fresh contiguous buffer:
// `dst[(i*k + j)*blockLen + l] = src[i*blockLen + l]`. The constructor
// is reused so dtype is preserved (Float64Array / Int32Array / …).
function _tileAtomMajor(src: any, N: number, k: number, blockLen: number): any {
  const dst = new src.constructor(N * k * blockLen);
  for (let i = 0; i < N; i++) {
    const sOff = i * blockLen;
    for (let j = 0; j < k; j++) {
      const dOff = (i * k + j) * blockLen;
      for (let l = 0; l < blockLen; l++) dst[dOff + l] = src[sOff + l];
    }
  }
  return dst;
}

/**
 * Repeat each atom's draw `k` times, atom-major, producing an
 * `N·k`-atom measure where atom `i·k + j == atom i` for every
 * `j ∈ [0, k)`. Copies all other fields through unchanged.
 *
 * This realises the **repeat axis** for the matIid composite fallback
 * (engine-concepts §20.1 / §22.4 — the `iid` entry that
 * `dissolver.propagateAxisStack` records). When `iid(M, k)` over a
 * composite `M` is materialised by inflating `M` to `N·k` atoms, the
 * atom-level VALUE draws `M` conditions on (e.g. a per-atom `psi`) must
 * stay CONSTANT across the `k` inner draws — spec §06's within-atom
 * conditional independence: "draw `psi_i` per atom, then `k` iid draws
 * from `M` *at that `psi_i`*". Tiling the parent's `N`-atom value to
 * `N·k` makes the inflated dispatch reuse `psi_i` for atom `i`'s `k`
 * inner positions, while `M`'s measure structure (branch selection,
 * leaf draws) redraws freshly per position. The caller's `[N, k]`
 * reshape is row-major, so the repeat is atom-major (block `i·k+j` ←
 * block `i`).
 *
 * Only the per-atom payloads a refArray param can carry are tiled
 * (`.samples` / `.value` (+ `.im`) / `.logWeights` / `.elems`); a no-op
 * when `k <= 1` or the leading axis isn't `N` (rank-0 constants ride
 * the fixedValues path and never reach here).
 */
function tileMeasureAtomMajor(m: any, N: number, k: number): any {
  if (!m || k <= 1) return m;
  const out = Object.assign({}, m);
  if (m.samples && m.samples.BYTES_PER_ELEMENT !== undefined
      && m.samples.length === N) {
    out.samples = _tileAtomMajor(m.samples, N, k, 1);
  }
  if (m.value && Array.isArray(m.value.shape) && m.value.shape[0] === N
      && m.value.data && m.value.data.BYTES_PER_ELEMENT !== undefined) {
    const inner = m.value.shape.slice(1);
    const blockLen = inner.reduce((a: number, b: number) => a * b, 1);
    const nv = Object.assign({}, m.value);
    nv.shape = [N * k].concat(inner);
    nv.data = _tileAtomMajor(m.value.data, N, k, blockLen);
    if (m.value.im && m.value.im.BYTES_PER_ELEMENT !== undefined) {
      nv.im = _tileAtomMajor(m.value.im, N, k, blockLen);
    }
    out.value = nv;
  }
  if (m.logWeights && m.logWeights.BYTES_PER_ELEMENT !== undefined
      && m.logWeights.length === N) {
    out.logWeights = _tileAtomMajor(m.logWeights, N, k, 1);
  }
  if (Array.isArray(m.elems)) {
    out.elems = m.elems.map((e: any) => tileMeasureAtomMajor(e, N, k));
  }
  return out;
}

/**
 * Resolve every value-position self-ref in `ir` to its parent's
 * Value, returning a refName→Value map for the worker primitives'
 * refArrays. Fixed-phase refs are auto-pushed into the worker
 * session env via `pushFixedEnv` (setEnv merge), so each caller's
 * subsequent sendWorker call sees them automatically — no caller
 * needs to remember the setEnv step.
 *
 * Signature: `collectRefArrays(ir, ctx)`. `ctx` is the materialiser
 * context — must carry `fixedValues` (the orchestrator's pre-eval
 * map), `getMeasure(name)` (parent-measure lookup), and `sendWorker`
 * (for the fixedEnv push). The legacy 3-arg form
 * `(ir, fixedValues, getMeasure)` was retired 2026-05-29 after every
 * in-engine caller migrated to ctx-form; eight callers each
 * remembering setEnv-merge was exactly the distributed-responsibility
 * anti-pattern engine-concepts §11 calls out.
 */
function collectRefArrays(ir: any, ctx: any) {
  const refs = orchestrator.collectSelfRefs(ir);
  const names: string[] = [];
  const fixedEnv: Record<string, any> = {};
  let anyFixed = false;
  const fixedValues = ctx && ctx.fixedValues;
  const extra = ctx && ctx._extraRefArrays;
  refs.forEach((n: string) => {
    // A ref the CLM boundary feed already supplies (ctx._extraRefArrays) is
    // taken from the overlay, NOT re-materialised via getMeasure — this both
    // avoids the boundary-conflation re-draw (audit §3) and lets SYNTHETIC
    // history-variate names (s0, s1, … — not bindings) resolve without a
    // "no derivation" throw.
    if (extra && extra[n] != null) return;
    if (fixedValues && fixedValues.has(n)) {
      fixedEnv[n] = fixedValues.get(n);
      anyFixed = true;
      return;
    }
    names.push(n);
  });
  const preFlight = anyFixed ? pushFixedEnv(ctx, fixedEnv) : Promise.resolve();
  return preFlight
    .then(() => Promise.all(names.map(ctx.getMeasure)))
    .then((measures: any[]) => {
      const out: any = {};
      for (let i = 0; i < names.length; i++) {
        out[names[i]] = measureToRefValue(measures[i], names[i], 'collectRefArrays');
      }
      // CLM boundary feed (measure-lowering unification Phase 4). matClm feeds
      // a lowered body's boundary inputs via feedInputs (the ONE feeding
      // contract sample + density share) and threads the columns through
      // `ctx._extraRefArrays`. These cover refs `collectSelfRefs` does NOT
      // see — crucially the kernel's `%local`-namespaced param refs (a NAMED
      // record-base kchain kernel keeps `ref(%local, t1)`, which the worker
      // resolves by bare name from refArrays). They override the getMeasure
      // results so the body conditions on the FED prior, not a re-materialised
      // like-named binding (the boundary-conflation bug, audit §3).
      if (ctx && ctx._extraRefArrays) {
        for (const k in ctx._extraRefArrays) {
          if (ctx._extraRefArrays[k] != null) out[k] = ctx._extraRefArrays[k];
        }
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

// Transitive boundary substitution (audit §3 #4, fixes H5 + H3's
// derived-parameter case). A reified kernel's distribution parameters often
// reach the boundary inputs through DERIVED value bindings — e.g. a kernel
// with boundaries `theta1`/`theta2` whose body is `iid(Normal(mu = a,
// sigma = b), 10)` with `a = 5*theta2`, `b = abs(theta1)*theta2`. The density
// side must score against the FED boundary atoms, but `a`/`b` are value
// bindings (no measure derivation), so prepareDensityRefs would try to
// getMeasure them → "no derivation for 'a'".
//
// This walk inlines every evaluable VALUE binding (recursively) so the
// dist-params end up expressed in the boundary inputs, which it keeps as
// `self` refs (NOT %local) so the caller's boundRefArrays feeding + the
// per-atom accessor — both keyed by bare self-ref name — still resolve them.
// Measures / draws carry a derivation and are left as refs for the density
// walker; boundary inputs are stopped before inlining. Cycle-guarded.
function inlineBoundaryDerivations(ir: any, boundarySet: Set<string>, ctx: any): any {
  const bindings = ctx && ctx.bindings;
  const derivations = ctx && ctx.derivations;
  const visiting = new Set<string>();
  function walk(node: any): any {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(walk);
    if (node.kind === 'ref' && node.ns === 'self' && node.name) {
      const name = node.name;
      if (boundarySet.has(name)) return node;     // fed boundary input — keep
      if (visiting.has(name)) return node;
      const drv = derivations && Object.prototype.hasOwnProperty.call(derivations, name)
        ? derivations[name] : null;
      const target = bindings && bindings.get(name);
      // Evaluable value binding: an 'evaluate' derivation, or no derivation
      // with a call-shaped ir (a pruned computed value like `a = 5*theta2`).
      // Measures / draws carry a sample/iid/record/… derivation and are NOT
      // inlined — they stay refs the density walker resolves per-atom.
      const isEvaluableValue =
        (drv && drv.kind === 'evaluate')
        || (!drv && target && target.ir && target.ir.kind === 'call');
      if (isEvaluableValue && target && target.ir) {
        visiting.add(name);
        const sub = walk(target.ir);
        visiting.delete(name);
        return sub;
      }
      return node;
    }
    const out: Record<string, any> = {};
    for (const k in node) out[k] = walk(node[k]);
    return out;
  }
  return walk(ir);
}

function prepareDensityRefs(ir: any, ctx: any, label: string, boundRefArrays?: any) {
  const refs = orchestrator.collectSelfRefs(ir);
  // Reified-kernel boundary inputs are FED by the caller (the prior's atoms
  // in bayesupdate, the integration variable in kchain), per the spec
  // lowering — never re-materialised via getMeasure (audit §3 / H1). When
  // `boundRefArrays` supplies a name, use that column and skip getMeasure.
  const bound = boundRefArrays && typeof boundRefArrays === 'object' ? boundRefArrays : null;
  // CLM boundary feed (measure-lowering unification Phase 4): matClm threads the
  // fed boundary columns through `ctx._extraRefArrays`, the overlay
  // `collectRefArrays` already honours (above). Honour it here too so the
  // density / evaluate ref-prep reached from a CLM-fed child ctx (the Smell A
  // materialiser merge) conditions on the FED prior, not a re-materialised
  // like-named binding (audit §3). Treated exactly like `bound`: skip
  // getMeasure for a supplied ref, then overlay the columns at the end.
  const extra = ctx && ctx._extraRefArrays && typeof ctx._extraRefArrays === 'object'
    ? ctx._extraRefArrays : null;
  // Bijection bindings reachable from `ir` carry their registry
  // parameters (affine's mu/cov → {b, L} paramIRs) in a side-channel on
  // `binding.bijection`, NOT in the walked IR tree — so collectSelfRefs
  // can't see them, and the bijection binding itself is skipped below as
  // function-like. The density walker's registry fast path (walkPushfwd)
  // evaluates those paramIRs and needs their upstream refs in the eval
  // env. Without this, a lifted MvNormal whose cov/mu is a NAMED REF
  // (e.g. `cov = sigma` where `sigma = rowstack(...)`, enabled by the
  // 5f-1 gate widening) throws "unbound self reference 'sigma'" at
  // density time. Gather those refs so they flow into fixedEnv /
  // perAtom alongside the rest. (The sample side avoids this because
  // matPushfwd resolves paramIRs on the main thread via
  // resolveIRToValue, which has the binding map.)
  const paramIRRefs = new Set<string>();   // refs sourced from bijection paramIRs
  for (const n of Array.from(refs)) {
    const b = ctx.bindings && ctx.bindings.get(n);
    if (b && b.bijection && b.bijection.paramIRs) {
      for (const k of Object.keys(b.bijection.paramIRs)) {
        for (const r of orchestrator.collectSelfRefs(b.bijection.paramIRs[k])) {
          refs.add(r);
          paramIRRefs.add(r);
        }
      }
    }
  }
  // 5f-2 / adversarial-verify Issue 2: a per-atom (stochastic) ref
  // sourced from a bijection's paramIRs means a marginal density of a
  // hierarchical pushfwd (per-atom mean/scale) — a Monte Carlo estimate
  // the closed-form single-observation density walker can't compute.
  // Detect it HERE and emit the actionable deferred diagnostic, rather
  // than letting the composite param measure crash `measureToRefValue`
  // below with a confusing "neither .value nor .samples". (Sampling such
  // models works via matPushfwd; only the closed-form density defers.)
  for (const n of paramIRRefs) {
    const isFixed = !!(ctx.fixedValues && ctx.fixedValues.has(n));
    const isBinding = !!(ctx.bindings && ctx.bindings.has(n));
    if (isBinding && !isFixed
        && !isFunctionLikeBinding(ctx.bindings.get(n))) {
      throw new Error(label + ": pushfwd bijection has an atom-dependent "
        + "registry param (per-atom mean/scale via '" + n + "'); the "
        + "marginal density of a hierarchical pushfwd is a Monte Carlo "
        + "estimate not supported on the closed-form path (deferred — "
        + "sampling such models works via matPushfwd)");
    }
  }
  const perAtomNames: string[] = [];
  const fixedEnv: Record<string, any> = {};
  refs.forEach((n: any) => {
    // Boundary inputs the caller fed take precedence over any like-named
    // module binding — do NOT getMeasure them (that is the conflation bug).
    if (bound && Object.prototype.hasOwnProperty.call(bound, n)) return;
    if (extra && extra[n] != null) return;
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
    if (bound) for (const k in bound) refArrays[k] = bound[k];
    if (extra) for (const k in extra) { if (extra[k] != null) refArrays[k] = extra[k]; }
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
 * refs — but NOT the async per-atom measure-fetch. (Demand-driven, §17.4:
 * fixed-phase refs are no longer pre-loaded by a bulk setEnv push;
 * render-profile now pushes THIS profile body's own fixed refs to the
 * worker session env on demand before the profileN call. The filter
 * logic here is unchanged — still correct, just for that new reason.)
 * `profileN` doesn't accept refArrays;
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
 * companion to prepareDensityRefs. Empty maps short-circuit.
 *
 * Single responsibility: this helper covers ONLY fixed-phase
 * pre-evaluated bindings (the `fixedValues` channel). The module
 * registry is pushed separately via `pushModuleRegistry(ctx)` —
 * see that helper for the rationale.
 */
function pushFixedEnv(ctx: any, fixedEnv: Record<string, any>) {
  for (const _k in fixedEnv) {
    return ctx.sendWorker({ type: 'setEnv', env: fixedEnv, merge: true });
  }
  return Promise.resolve();
}

/**
 * Push `ctx.moduleRegistry` (alias → {stdName, stdCompat}) onto the
 * worker session env as `__moduleRegistry`, idempotent per ctx.
 *
 * Cross-module call dispatch (sampler.ts `_evaluateStandardModuleCall`)
 * reads `env.__moduleRegistry` to resolve `(call target=({ns:
 * <alias>, name: X}) ...)` to the standard-modules.ts registry's
 * descriptor. The push happens ONCE per materialiser ctx — guarded
 * by a `_moduleRegistryPushed` flag on the ctx itself so repeated
 * calls short-circuit. Hosts that maintain their own setEnv plumbing
 * (the viewer's `rebuildDerivations`) push it directly; this helper
 * covers tests and any path that materialises through `matSession`-
 * style entry points.
 *
 * Empty registries short-circuit. Idempotent: safe to call from
 * every materialiser entry point as a defensive precondition.
 */
function pushModuleRegistry(ctx: any) {
  if (!ctx || !ctx.moduleRegistry || ctx._moduleRegistryPushed) {
    return Promise.resolve();
  }
  const reg = ctx.moduleRegistry;
  // Empty registry → nothing to push.
  let empty = true;
  for (const _k in reg) { empty = false; break; }
  if (empty) {
    ctx._moduleRegistryPushed = true;
    return Promise.resolve();
  }
  ctx._moduleRegistryPushed = true;
  return ctx.sendWorker({
    type: 'setEnv',
    env: { __moduleRegistry: reg },
    merge: true,
  });
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
    // Set `intrinsicShape` (the viewer's matrix-mode trigger) ONLY for
    // true matrices / tensors, NOT for nested-vector values. Spec §03:
    // a vec-of-vec is not a matrix; the viewer must not render it as
    // a heatmap. The user-facing type system already routes vec-of-vec
    // to the array-mode step plot via `inferredType.kind === 'array'`
    // with rank-1 elem-array; this is the runtime defense-in-depth.
    if (dense.shape.length >= 2 && !valueLib.isNestedVectorValue(dense)) {
      m.intrinsicShape = dense.shape.slice();
    }
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
    if (valueLib.isAtomBatched(v, N) && shape.length === 1) {
      col.set(data);
    } else if (valueLib.isAtomBatched(v, N) && shape.length === 2) {
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

// Phase 4.1 extension — per-atom slice at cell (j, r) for the
// vector-per-cell broadcast shape contract.
//
// In matKernelBroadcast's iid-composite path, a broadcast arg may
// itself be rank-2 atom-indep `[K, D]` (e.g. `x_per_group` in the
// random-intercepts regression) or rank-3 atom-batched `[N, K, D]`.
// The K axis is the outer kernel-broadcast cell axis; the D axis is
// the INNER iid axis (matches the composite kernel body's `iid(M, n)`
// count). Iterating (j, r) reduces such a value to a per-atom scalar.
//
// `elemScalarAtJR` returns the atom-indep scalar at (j, r). Handles
// scalar / [K] / [K, D]; size-1 axes broadcast by repetition.
//
// `perAtomColumnAtJR` returns the per-atom column at (j, r). Handles
// atom-batched [N] / [N, K] / [N, K, D]; size-1 inner axes broadcast.
function elemScalarAtJR(
  v: any, j: number, r: number, _N: number,
): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (valueLib.isValue(v)) {
    const shape = v.shape;
    if (shape.length === 0) return v.data[0];
    if (shape.length === 1) {
      // [K] axis only; index j (size-1 broadcasts).
      return v.data[shape[0] === 1 ? 0 : j];
    }
    if (shape.length === 2) {
      // [K, D] atom-indep, row-major.
      const K = shape[0], D = shape[1];
      const jj = K === 1 ? 0 : j;
      const rr = D === 1 ? 0 : r;
      return v.data[jj * D + rr];
    }
  }
  if (v && v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number') {
    return v[v.length === 1 ? 0 : j];
  }
  if (Array.isArray(v)) return +v[v.length === 1 ? 0 : j];
  throw new Error('broadcast: cannot scalarise atom-indep value at (j='
    + j + ',r=' + r + ')');
}

function perAtomColumnAtJR(
  v: any, j: number, r: number, N: number,
): Float64Array {
  const col = new Float64Array(N);
  if (valueLib.isValue(v)) {
    const shape = v.shape;
    const data = v.data;
    if (valueLib.isAtomBatched(v, N) && shape.length === 1) {
      // [N] — atom-batched scalar; same per-atom for all (j, r).
      col.set(data);
    } else if (valueLib.isAtomBatched(v, N) && shape.length === 2) {
      // [N, K] — atom-batched scalar-per-cell; index j.
      const K = shape[1];
      const off = K === 1 ? 0 : j;
      for (let i = 0; i < N; i++) col[i] = data[i * K + off];
    } else if (valueLib.isAtomBatched(v, N) && shape.length === 3) {
      // [N, K, D] — atom-batched vector-per-cell; index (j, r).
      const K = shape[1], D = shape[2];
      const jj = K === 1 ? 0 : j;
      const rr = D === 1 ? 0 : r;
      const stride2 = K * D;
      for (let i = 0; i < N; i++) col[i] = data[i * stride2 + jj * D + rr];
    } else {
      throw new Error('broadcast: unexpected per-atom shape '
        + JSON.stringify(shape) + ' at (j=' + j + ',r=' + r + ')');
    }
  } else if (v && v.BYTES_PER_ELEMENT !== undefined && v.length === N) {
    col.set(v);
  } else {
    throw new Error('broadcast: per-atom param has unexpected type '
      + (typeof v) + ' at (j=' + j + ',r=' + r + ')');
  }
  return col;
}

// Classify a broadcast-arg or param Value's shape relative to atom N
// AND a known set of atom-batched ref names. Returns one of:
//
//   { kind: 'scalar' }                  — rank-0 / number / boolean
//   { kind: 'K',   K }                  — atom-indep, rank-1 [K]
//   { kind: 'KD',  K, D }               — atom-indep, rank-2 [K, D]
//   { kind: 'N' }                       — atom-batched scalar [N]
//   { kind: 'NK',  K }                  — atom-batched scalar-per-cell [N, K]
//   { kind: 'NKD', K, D }               — atom-batched vec-per-cell [N, K, D]
//
// `usesAtom` is the disambiguator for rank-1 [N]: a stand-alone
// length-N vector that DOESN'T reference any atom-batched ref is
// treated as `K = N` (a real length-N collection arg), not as
// `kind: 'N'`. Mirrors mat-broadcast's existing per-param classifier.
function classifyBroadcastArg(
  v: any, N: number, usesAtom: boolean,
): { kind: string; K?: number; D?: number } {
  if (typeof v === 'number' || typeof v === 'boolean') return { kind: 'scalar' };
  if (valueLib.isValue(v)) {
    const shape = v.shape;
    if (shape.length === 0) return { kind: 'scalar' };
    if (shape.length === 1) {
      if (usesAtom && valueLib.isAtomBatched(v, N)) return { kind: 'N' };
      return { kind: 'K', K: shape[0] };
    }
    if (shape.length === 2) {
      if (valueLib.isAtomBatched(v, N)) return { kind: 'NK', K: shape[1] };
      return { kind: 'KD', K: shape[0], D: shape[1] };
    }
    if (shape.length === 3) {
      if (valueLib.isAtomBatched(v, N)) {
        return { kind: 'NKD', K: shape[1], D: shape[2] };
      }
    }
    return { kind: 'unsupported' };
  }
  if (v && v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number') {
    if (usesAtom && v.length === N) return { kind: 'N' };
    return { kind: 'K', K: v.length };
  }
  if (Array.isArray(v)) return { kind: 'K', K: v.length };
  return { kind: 'unsupported' };
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
  measureToParamValue,
  measureToPerAtomRecords,
  inlineBoundaryDerivations,
  tileMeasureAtomMajor,
  collectRefArrays,
  prepareDensityRefs,
  classifyProfileSelfRefs,
  pushFixedEnv,
  pushModuleRegistry,
  fixedValueToMeasure,
  measureN,
  valueOf,
  scalarMeasureN,
  measureFromValue,
  measureFromReply,
  setBoundsForMat,
  elemScalarAtJ,
  perAtomColumnAtJ,
  elemScalarAtJR,
  perAtomColumnAtJR,
  classifyBroadcastArg,
  resolveFnBody,
};
