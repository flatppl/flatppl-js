'use strict';

// =====================================================================
// kernel-broadcast-shape.ts — recognise kernel-broadcast IR shapes
// =====================================================================
//
// **Background.** Before P7 of the broadcast/aggregate/batching
// consolidation (TODO-flatppl-js.md "In-flight P1-P9"), THREE places
// recognised the iid-composite kernel-binding shape (used by
// `broadcast(<user-kernel>, …)` where the user kernel is
// `kernelof(iid(<Dist>, n), kw)`):
//
//   1. `derivations.ts:_isIidCompositeKernelBinding` — classify-time
//      recognition for kernel-broadcast derivation building.
//   2. `mat-broadcast.ts:_detectIidKernelBody` — runtime unpacking to
//      drive the per-cell sampleN loop with repeat=n.
//
// Both walked the same IR shape:
//
//     functionof(
//       lawof(iid(<BuiltinDistCall_or_ref>, <n_literal>)),
//       <kernel_kwargs>
//     )
//
// — with one-level anon-deref through the lift's hoisting and the
// same gate that the inner builtin must be in
// `SAMPLEABLE_DISTRIBUTIONS`. The two walks drifted at the
// fixedValues lookup (mat-broadcast.ts:347-351 accepts a fixed-phase
// integer binding via ctx.fixedValues; derivations.ts didn't).
//
// **P7 contract.** ONE module hoists the structural recognition;
// classify-time and runtime consumers share it. The classifier takes
// the bindings map and an optional `fixedValues` lookup; with
// fixedValues, ref-to-integer-binding form is admissible (matches
// mat-broadcast.ts's runtime extension); without, only literal n is
// admitted (matches classify-time conservatism).
//
// SOTA alignment: Pyro's `plate` is the single object recognised by
// every downstream system (SVI, MCMC, predictive) — no parallel
// recognisers in TraceEnum / NUTS. Our kernel-broadcast shape is the
// analogous "one IR shape, many consumers" pattern.

interface IidKernelDescriptor {
  /** The user-kernel binding's IR (the functionof node). */
  binding: any;
  /** The kernel's parameter names. */
  params: string[];
  /** The kernel's kwarg names (mirror of params for canonical kernels). */
  paramKwargs: string[];
  /** The inner builtin distribution opcode (e.g. 'Normal', 'Bernoulli'). */
  distOp: string;
  /** The inner builtin's parameter names (from sampler.REGISTRY). */
  distParams: string[];
  /** The inner builtin's kwargs IR (placeholder-substituted at call-site). */
  distKwargs: Record<string, any>;
  /** The iid axis size — literal positive integer. */
  n: number;
}

/**
 * Recognise an iid-composite user-kernel binding. Returns an
 * `IidKernelDescriptor` on match, null otherwise.
 *
 * Accepts the following IR shape (post-lowering):
 *
 *     functionof(
 *       lawof(iid(<BuiltinDistCall>, n_literal | n_fixed_ref)),
 *       kernel_kwargs...
 *     )
 *
 * - `<BuiltinDistCall>` may be inline or one-level anon-deref via
 *   the lift pass's hoisting.
 * - `n` is admitted as a literal positive integer; with
 *   `fixedValues` supplied, also as a self-ref to a fixed-phase
 *   integer binding (runtime use). Without `fixedValues`, only
 *   literal n is admitted (classify-time conservatism — defers
 *   ref resolution to materialise time).
 * - The inner builtin must be in `SAMPLEABLE_DISTRIBUTIONS`.
 *
 * Single source of truth for the classify-time +
 * runtime-unpacking recognition; both `derivations.classify-
 * KernelBroadcast`'s sub-helper and `mat-broadcast._detectIid-
 * KernelBody` delegate here.
 */
function detectIidKernelBinding(
  name: string, bindings: any, fixedValues?: any,
): IidKernelDescriptor | null {
  if (!bindings || !bindings.has || !bindings.has(name)) return null;
  const b = bindings.get(name);
  if (!b || !b.ir) return null;
  const ir = b.ir;
  if (ir.kind !== 'call' || ir.op !== 'functionof') return null;
  const params: string[] = Array.isArray(ir.params) ? ir.params : [];
  if (params.length === 0) return null;
  const paramKwargs: string[] = Array.isArray(ir.paramKwargs)
    ? ir.paramKwargs : params;
  const body = ir.body;
  if (!body || body.kind !== 'call' || body.op !== 'lawof') return null;
  const innerMeasure = body.args && body.args[0];
  if (!innerMeasure || innerMeasure.kind !== 'call'
      || innerMeasure.op !== 'iid') return null;
  const iidArgs = innerMeasure.args || [];
  if (iidArgs.length !== 2) return null;
  // Dereference one level of anon ref. Post-lift, `iid(Normal(...), N)`
  // becomes `iid(ref(__anonM), N)` where the anon's IR holds the
  // literal Normal call.
  let distCall = iidArgs[0];
  if (distCall && distCall.kind === 'ref' && distCall.ns === 'self'
      && bindings.has(distCall.name)) {
    const anon = bindings.get(distCall.name);
    if (anon && anon.ir) distCall = anon.ir;
  }
  if (!distCall || distCall.kind !== 'call' || !distCall.op) return null;
  // The inner builtin must be sampleable. We lazy-require the
  // SAMPLEABLE_DISTRIBUTIONS set from ir-shared.ts (avoids an
  // import-cycle hazard with derivations.ts).
  const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;
  if (!SAMPLEABLE || !SAMPLEABLE.has(distCall.op)) return null;
  // Resolve n. Literal integer OR (when fixedValues is supplied) a
  // self-ref to a fixed-phase integer binding. AT CLASSIFY TIME (no
  // fixedValues), we accept the shape even when n is a ref — the
  // runtime caller resolves the actual value later. This matches
  // derivations.ts's pre-P7 behaviour (classify-time gate was purely
  // structural; n-value validation was a runtime concern).
  const nArg = iidArgs[1];
  let nLit: any = null;
  if (nArg && nArg.kind === 'lit') nLit = nArg.value;
  if (nLit === null && fixedValues && nArg && nArg.kind === 'ref'
      && nArg.ns === 'self'
      && (typeof fixedValues.has === 'function'
          ? fixedValues.has(nArg.name)
          : Object.prototype.hasOwnProperty.call(fixedValues, nArg.name))) {
    nLit = typeof fixedValues.get === 'function'
      ? fixedValues.get(nArg.name)
      : fixedValues[nArg.name];
  }
  // Validate n strictly ONLY when fixedValues was supplied (runtime
  // caller needs a concrete n). At classify-time the ref form is
  // admissible (returns descriptor with n=NaN as a sentinel — caller
  // ignores n at classify-time since it only needs the yes/no).
  if (fixedValues) {
    if (typeof nLit !== 'number' || !Number.isInteger(nLit) || nLit <= 0) {
      return null;
    }
  } else if (typeof nLit !== 'number') {
    // Classify-time: ref-to-non-fixed-binding is admissible; we
    // signal n-not-yet-known via NaN so the classify-time yes/no
    // gate via `isIidCompositeKernelBinding` returns true.
    nLit = NaN;
  }
  // Look up the inner builtin's parameter names via sampler.REGISTRY.
  // sampler may not be loadable in some contexts; when it IS loadable
  // but doesn't recognise distCall.op, reject (matches the original
  // _detectIidKernelBody behaviour of `if (!distParams) return null`).
  // When sampler isn't loadable (deep import cycle at module load),
  // accept with distParams empty — classify-time callers only need
  // the yes/no decision.
  let distParams: string[] = [];
  let samplerLoaded = false;
  try {
    const sampler = require('./sampler.ts');
    if (sampler && sampler._internal && sampler._internal.REGISTRY) {
      samplerLoaded = true;
      const entry = sampler._internal.REGISTRY[distCall.op];
      if (entry && Array.isArray(entry.params)) distParams = entry.params;
      else return null;       // sampler loaded but doesn't know this dist
    }
  } catch (_) {
    // sampler not loadable — fall through with distParams empty.
  }
  return {
    binding: b,
    params,
    paramKwargs,
    distOp: distCall.op,
    distParams,
    distKwargs: distCall.kwargs || {},
    n: nLit,
  };
}

/**
 * Lighter check: does `name` resolve to an iid-composite kernel
 * binding? Used by `derivations.classifyKernelBroadcast` which only
 * needs the yes/no answer at classify time. Returns true iff
 * `detectIidKernelBinding` would succeed. (Without fixedValues —
 * classify-time conservatism.)
 */
function isIidCompositeKernelBinding(name: string, bindings: any): boolean {
  return detectIidKernelBinding(name, bindings) !== null;
}

// =====================================================================
// Joint-bodied composite kernel (Phase 4.2)
// =====================================================================
//
// A `kernelof(joint(<components>), <kw>)` body — each cell of the
// surrounding `broadcast(K, …)` produces ONE draw from the joint
// product measure of the components. Components may be positional
// (`joint(M1, M2, ...)`) or keyword (`joint(name1 = M1, name2 = M2,
// ...)`); both lower to the same `joint` IR with `args` or `fields`
// populated. Phase 4.2 MVP restricts each component to a built-in
// sampleable scalar distribution (Normal, Beta, Gamma, ...). Vector-
// valued components (MvNormal) defer to Phase 5.1 where MvNormal joins
// the sampler REGISTRY so the worker's sampleN can handle it through
// the same kwarg-driven path; the joint executor will then route
// vector components without further changes.
//
// Per spec §06: positional and keyword joints both denote the same
// independent-product measure construction; they differ only in
// surface variate shape (concat-vector vs named record). The
// recogniser carries the layout flag for the executor to honour.

interface JointKernelComponent {
  /** Surface field name (keyword joint only). Undefined for positional. */
  surfaceName?: string;
  /** Component's built-in distribution opcode (e.g. 'Normal'). */
  distOp: string;
  /** Component's REGISTRY param names. */
  distParams: string[];
  /** Component's kwargs IR with kernel placeholders still embedded —
   *  the executor substitutes them per cell. */
  distKwargs: Record<string, any>;
}

interface JointKernelDescriptor {
  /** The user-kernel binding's IR (the functionof node). */
  binding: any;
  /** Kernel parameter names. */
  params: string[];
  /** Kernel surface kwarg names. */
  paramKwargs: string[];
  /** Component layout: 'positional' (concat-vector variate) vs
   *  'keyword' (named-record variate). */
  layout: 'positional' | 'keyword';
  /** Ordered components. Length ≥ 1. */
  components: JointKernelComponent[];
}

/**
 * Recognise a joint-composite user-kernel binding. Returns a
 * `JointKernelDescriptor` on match, null otherwise.
 *
 * Accepts the following IR shapes (post-lowering):
 *
 *     functionof(
 *       lawof(joint(args = [<componentRef>, ...])),      # positional
 *       kernel_kwargs...
 *     )
 *
 *     functionof(
 *       lawof(joint(fields = [{name, value: <componentRef>}, ...])),
 *       kernel_kwargs...
 *     )
 *
 * - Each `<componentRef>` must be a `self`-ref to an anon binding
 *   whose `ir` is a direct call to a sampler-REGISTRY-known
 *   distribution (kernel placeholders inside the component's kwargs
 *   resolve at execute time via the per-cell substitution pass).
 * - Phase 4.2 MVP does not recurse into nested composites; a
 *   component that is itself a `joint` / `iid` / `broadcast` returns
 *   null here. Future phases (4.3 jointchain, 4.4 nested broadcast)
 *   extend the recogniser with their own kinds.
 */
function detectJointKernelBinding(
  name: string, bindings: any,
): JointKernelDescriptor | null {
  if (!bindings || !bindings.has || !bindings.has(name)) return null;
  const b = bindings.get(name);
  if (!b || !b.ir) return null;
  const ir = b.ir;
  if (ir.kind !== 'call' || ir.op !== 'functionof') return null;
  const params: string[] = Array.isArray(ir.params) ? ir.params : [];
  if (params.length === 0) return null;
  const paramKwargs: string[] = Array.isArray(ir.paramKwargs)
    ? ir.paramKwargs : params;
  const body = ir.body;
  if (!body || body.kind !== 'call' || body.op !== 'lawof') return null;
  const innerMeasure = body.args && body.args[0];
  if (!innerMeasure || innerMeasure.kind !== 'call'
      || innerMeasure.op !== 'joint') return null;

  // Discriminate layout. Positional joint sets `args`; keyword joint
  // sets `fields`. The two are mutually exclusive in canonical IR.
  let componentRefs: any[];
  let surfaceNames: (string | undefined)[];
  let layout: 'positional' | 'keyword';
  if (Array.isArray(innerMeasure.fields) && innerMeasure.fields.length > 0) {
    layout = 'keyword';
    componentRefs = innerMeasure.fields.map((f: any) => f && f.value);
    surfaceNames = innerMeasure.fields.map((f: any) => f && f.name);
  } else if (Array.isArray(innerMeasure.args) && innerMeasure.args.length > 0) {
    layout = 'positional';
    componentRefs = innerMeasure.args;
    surfaceNames = componentRefs.map(() => undefined);
  } else {
    return null;
  }

  // Each component must dereference (one level of anon) to a direct
  // sampleable DistCall. Reject any nested composite — those are
  // future-phase recogniser targets.
  const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;

  // sampler REGISTRY lookup — lazy-required to avoid the import cycle
  // that bites at module load. When sampler is loadable, distParams
  // come from the entry; when not, the descriptor caller (classify-
  // time) only needs the yes/no decision, and distParams stay empty.
  let samplerLoaded = false;
  let samplerRegistry: any = null;
  try {
    const sampler = require('./sampler.ts');
    if (sampler && sampler._internal && sampler._internal.REGISTRY) {
      samplerLoaded = true;
      samplerRegistry = sampler._internal.REGISTRY;
    }
  } catch (_) { /* deep import cycle; fall through */ }

  const components: JointKernelComponent[] = [];
  for (let i = 0; i < componentRefs.length; i++) {
    const ref = componentRefs[i];
    if (!ref || ref.kind !== 'ref' || ref.ns !== 'self'
        || !bindings.has(ref.name)) return null;
    const anon = bindings.get(ref.name);
    if (!anon || !anon.ir) return null;
    const distCall = anon.ir;
    if (!distCall || distCall.kind !== 'call' || !distCall.op) return null;
    if (!SAMPLEABLE || !SAMPLEABLE.has(distCall.op)) return null;
    let distParams: string[] = [];
    if (samplerLoaded) {
      const entry = samplerRegistry[distCall.op];
      if (entry && Array.isArray(entry.params)) distParams = entry.params;
      else return null;       // sampler loaded but doesn't know this dist
    }
    components.push({
      surfaceName: surfaceNames[i],
      distOp: distCall.op,
      distParams,
      distKwargs: distCall.kwargs || {},
    });
  }

  return {
    binding: b,
    params,
    paramKwargs,
    layout,
    components,
  };
}

/**
 * Lighter check: does `name` resolve to a joint-composite kernel
 * binding? Mirrors `isIidCompositeKernelBinding`. Used by classify-
 * time (`derivations.classifyKernelBroadcast`) to gate the recogniser
 * before the runtime path runs.
 */
function isJointCompositeKernelBinding(name: string, bindings: any): boolean {
  return detectJointKernelBinding(name, bindings) !== null;
}

// =====================================================================
// Jointchain-bodied composite kernel (Phase 4.3)
// =====================================================================
//
// A `kernelof(jointchain(<base>, <K_1>, <K_2>, ...), <kw>)` body — a
// Markov-chain measure: step 0 draws from a base measure, each
// subsequent step k draws from kernel K_k applied to the previous
// step's variate. Per spec §06 jointchain factorises as
// p(v_0, v_1, …, v_n) = p_base(v_0) · K_1(v_0)(v_1) · K_2(v_1)(v_2) · …
//
// Phase 4.3 MVP scope:
//
// - **Closed-first only.** Step 0 is a base measure (a self-ref to an
//   anon binding whose `ir` is a sampleable DistCall with kernel
//   placeholders embedded). Kernel-first jointchain — where step 0 is
//   itself a kernel that takes the broadcast args as input — defers
//   to a follow-up; its admissibility hooks differ enough from the
//   closed-first surface that landing them together would muddle the
//   recogniser's contract.
//
// - **Scalar-step kernels only.** Each step k ≥ 1 must be a self-ref
//   to a kernel binding with body `lawof(<sampleable DistCall>)` —
//   single component, not joint or iid. Composite step bodies
//   (iid-step, joint-step) defer to a Phase 4.3 follow-up that wires
//   recursive composite recognition (similarly relevant to Phase 4.4
//   nested broadcast).
//
// - **Single-input kernel steps.** Each step kernel must have exactly
//   one parameter — bound to the immediately previous step's variate.
//   Multi-input steps (kernels that consume `cat(v_0, …, v_{k-1})`,
//   admissible per `classifyJointchain`) defer; the MVP fixture
//   (AR-1) doesn't exercise them.
//
// - **Positional layout only.** Keyword-form jointchain
//   (`jointchain(s0 = M, s1 = K)`, populating IR `fields`) is
//   admitted by `classifyJointchain` and produces a record-typed
//   variate; the MVP fixture uses positional, and record-typed
//   variates need additional plumbing in the materialiser's Value
//   contract. Defer to a follow-up.
//
// Per-step state threading: at execute time, step k's sampleN sees
// the prev-variate parameter bound to a per-atom refArray whose data
// is step k-1's already-sampled column. The executor walks the chain
// step-by-step within each broadcast cell; step k waits on step k-1.

interface JointChainStep {
  /** Closed-first base measure (step 0). The DistCall whose kwargs
   *  reference kernel placeholders (substituted per cell at execute
   *  time via `_substituteKernelParams`). */
  base?: {
    distOp: string;
    distParams: string[];
    distKwargs: Record<string, any>;
  };
  /** Kernel step (k ≥ 1). The single-component inner DistCall plus
   *  the kernel's parameter name (which receives the previous variate
   *  at execute time). */
  kernel?: {
    /** Kernel binding's parameter name. The step's DistCall references
     *  this name as a `%local` or `self` ref — the executor swaps it
     *  with a refArray pointing to step k-1's per-atom column. */
    inputParam: string;
    distOp: string;
    distParams: string[];
    distKwargs: Record<string, any>;
  };
}

interface JointChainKernelDescriptor {
  /** The user-kernel binding's IR (the functionof node). */
  binding: any;
  /** Outer kernel parameter names. */
  params: string[];
  /** Outer kernel surface kwarg names. */
  paramKwargs: string[];
  /** Chain length (including base). Length ≥ 2 (base + at least one
   *  kernel step — chains of length 1 are equivalent to plain joints
   *  and would route through that recogniser). */
  steps: JointChainStep[];
}

/**
 * Recognise a jointchain-composite user-kernel binding. Returns a
 * `JointChainKernelDescriptor` on match, null otherwise.
 *
 * Accepts the following IR shape (post-lowering):
 *
 *     functionof(
 *       lawof(jointchain(
 *         args = [
 *           <baseRef>,      # anon-ref to sampleable DistCall
 *           <kernelRef_1>,  # binding-ref to single-step kernel
 *           <kernelRef_2>,
 *           …]
 *       )),
 *       outer_kernel_kwargs...
 *     )
 *
 * - `<baseRef>` derefs to an anon binding whose `ir` is a sampleable
 *   DistCall. The DistCall's kwargs may reference outer kernel
 *   placeholders (substituted per cell at execute time).
 * - Each `<kernelRef_k>` derefs to a kernel binding with `ir.op ===
 *   'functionof'`, exactly one param, body `lawof(<DistCall>)`. The
 *   DistCall's kwargs reference that single param (the prev variate).
 *
 * Phase 4.3 MVP rejections (return null):
 *  - Kernel-first chains (step 0 is itself a kernel binding).
 *  - Composite step bodies (step kernel's body isn't a direct DistCall).
 *  - Multi-input step kernels (more than one param).
 *  - Keyword-form jointchain (`fields:` populated).
 *  - Chains of length < 2.
 */
function detectJointChainKernelBinding(
  name: string, bindings: any,
): JointChainKernelDescriptor | null {
  if (!bindings || !bindings.has || !bindings.has(name)) return null;
  const b = bindings.get(name);
  if (!b || !b.ir) return null;
  const ir = b.ir;
  if (ir.kind !== 'call' || ir.op !== 'functionof') return null;
  const params: string[] = Array.isArray(ir.params) ? ir.params : [];
  if (params.length === 0) return null;
  const paramKwargs: string[] = Array.isArray(ir.paramKwargs)
    ? ir.paramKwargs : params;
  const body = ir.body;
  if (!body || body.kind !== 'call' || body.op !== 'lawof') return null;
  const innerMeasure = body.args && body.args[0];
  if (!innerMeasure || innerMeasure.kind !== 'call'
      || innerMeasure.op !== 'jointchain') return null;
  // MVP: positional layout only. Keyword (record-typed variate) defers.
  if (Array.isArray(innerMeasure.fields) && innerMeasure.fields.length > 0) {
    return null;
  }
  if (!Array.isArray(innerMeasure.args) || innerMeasure.args.length < 2) {
    return null;
  }

  const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;
  let samplerLoaded = false;
  let samplerRegistry: any = null;
  try {
    const sampler = require('./sampler.ts');
    if (sampler && sampler._internal && sampler._internal.REGISTRY) {
      samplerLoaded = true;
      samplerRegistry = sampler._internal.REGISTRY;
    }
  } catch (_) { /* fall through */ }

  // Look up REGISTRY params for a distOp. Returns null when sampler is
  // loaded but doesn't recognise the op; empty when sampler isn't
  // loadable (classify-time conservatism — accept the structural
  // match, leave param resolution to runtime).
  const lookupDistParams = (distOp: string): string[] | null => {
    if (!samplerLoaded) return [];
    const entry = samplerRegistry[distOp];
    if (entry && Array.isArray(entry.params)) return entry.params;
    return null;
  };

  const steps: JointChainStep[] = [];
  for (let i = 0; i < innerMeasure.args.length; i++) {
    const ref = innerMeasure.args[i];
    if (!ref || ref.kind !== 'ref' || ref.ns !== 'self'
        || !bindings.has(ref.name)) return null;
    const dep = bindings.get(ref.name);
    if (!dep || !dep.ir) return null;
    if (i === 0) {
      // Base step: dep.ir must be a sampleable DistCall (closed-first).
      const distCall = dep.ir;
      if (distCall.kind !== 'call' || !distCall.op) return null;
      if (!SAMPLEABLE || !SAMPLEABLE.has(distCall.op)) return null;
      const distParams = lookupDistParams(distCall.op);
      if (distParams === null) return null;
      steps.push({
        base: {
          distOp: distCall.op,
          distParams,
          distKwargs: distCall.kwargs || {},
        },
      });
    } else {
      // Kernel step: dep must be a kernel binding with single param,
      // body lawof(<sampleable DistCall>).
      const stepIR = dep.ir;
      if (stepIR.kind !== 'call' || stepIR.op !== 'functionof') return null;
      const stepParams: string[] = Array.isArray(stepIR.params)
        ? stepIR.params : [];
      // MVP: single-input kernels only (the canonical prev-variate
      // form). Multi-input kernels defer.
      if (stepParams.length !== 1) return null;
      const stepBody = stepIR.body;
      if (!stepBody || stepBody.kind !== 'call'
          || stepBody.op !== 'lawof') return null;
      const stepDist = stepBody.args && stepBody.args[0];
      if (!stepDist || stepDist.kind !== 'call' || !stepDist.op) return null;
      if (!SAMPLEABLE || !SAMPLEABLE.has(stepDist.op)) return null;
      const distParams = lookupDistParams(stepDist.op);
      if (distParams === null) return null;
      steps.push({
        kernel: {
          inputParam: stepParams[0],
          distOp: stepDist.op,
          distParams,
          distKwargs: stepDist.kwargs || {},
        },
      });
    }
  }

  return {
    binding: b,
    params,
    paramKwargs,
    steps,
  };
}

/**
 * Lighter check: does `name` resolve to a jointchain-composite
 * kernel binding? Mirrors `isIid…` / `isJoint…`.
 */
function isJointChainCompositeKernelBinding(
  name: string, bindings: any,
): boolean {
  return detectJointChainKernelBinding(name, bindings) !== null;
}

module.exports = {
  detectIidKernelBinding,
  isIidCompositeKernelBinding,
  detectJointKernelBinding,
  isJointCompositeKernelBinding,
  detectJointChainKernelBinding,
  isJointChainCompositeKernelBinding,
};
