// Engine-wide type declarations.
//
// This file declares the **cross-file** type surfaces — Value shape
// contract (engine-concepts §2.1, including the Klein-4 tag and
// `struct` bitmask), FlatPIR IR node discriminated union, derivation
// kinds, and the EmpiricalMeasure shape — that the rest of the engine
// can consume to tighten function signatures.
//
// Most function bodies still pass parameters as `any` until each file
// is per-tightened (the engine noImplicitAny ratchet is the
// outstanding strict mode that would force those signatures); the
// types here are the **target shapes** for that tightening.

// ---------------------------------------------------------------------
// Value (engine-concepts §2.1)
// ---------------------------------------------------------------------

/**
 * Klein-4 transpose/adjoint tag.
 *   'N'  normal (default; absent ⇒ 'N')
 *   'T'  transposed
 *   'A'  adjoint = transpose + conjugate
 *   'C'  conjugated only (= T ∘ A)
 *
 * Generators: transpose flips swapped bit (N↔T, A↔C); adjoint flips
 * both bits (N↔A, T↔C); conjugate flips conjugate bit (N↔C, T↔A).
 */
export type ValueTag = 'N' | 'T' | 'A' | 'C';

/** Backend dtype tag — 'f64' is the only realised one today; 'complex'
 *  is the planar (parallel re/im) shape. */
export type ValueDtype = 'f64' | 'complex';

/**
 * Shape-tagged value used throughout the engine for every numeric quantity.
 * Flat Float64Array storage; explicit shape (leading axis = batch);
 * optional Klein-4 tag for transpose/adjoint state and optional
 * `struct` bitmask for structured matrices (engine-concepts §2.2).
 *
 * Storage conventions:
 *   shape=[]           atom-indep scalar              data.length === 1
 *   shape=[N]          atom-batched scalar            data.length === N
 *   shape=[k]          atom-indep vector              data.length === k
 *   shape=[N, k]       atom-batched vector            data.length === N*k, atom-major
 *   shape=[m, n]       atom-indep matrix              data.length === m*n, row-major
 *   shape=[N, m, n]    atom-batched matrix            data.length === N*m*n
 */
export interface Value {
  shape: number[];
  data: Float64Array;
  /** Backend dtype tag; absent ⇒ 'f64'. */
  dtype?: ValueDtype;
  /** Klein-4 transpose/adjoint tag; absent ⇒ 'N'. */
  t?: ValueTag;
  /** Imaginary buffer for planar complex storage (matches data layout). */
  im?: Float64Array;
  /** Structured-matrix bitmask (engine-concepts §2.2). Absent ⇒ dense. */
  struct?: number;
  /**
   * Number of LEADING axes that are outer/loop axes; the trailing
   * `shape.length - outerRank` axes are the inner cell axes (engine-
   * concepts §2.1). Per spec §03, a user-written nested literal like
   * `[[1,2],[3,4]]` is a vector-of-vectors (NOT a matrix): the engine
   * carries it as `{shape:[2,2], outerRank:1, data:F64[4]}` — flat
   * row-major storage (ArrayOfSimilarArrays-style) plus the semantic
   * tag. Absent ⇒ every axis is a loop axis (flat tensor / matrix);
   * matrix-input linalg ops refuse Values where this tag is set via
   * `valueLib.requireMatrix(v, opName)`.
   */
  outerRank?: number;
}

// ---------------------------------------------------------------------
// FlatPIR-JSON IR (engine/lower.ts + flatppl-design §11)
// ---------------------------------------------------------------------
//
// The IR shape lowering produces. The `kind` field is the discriminator.

export interface IRBase {
  loc?: any;
  meta?: any;
  /** Escape hatch: engine internals attach a handful of extension fields
   *  (originLoc, lowerError, dependsOn, …) onto IR nodes as they pass
   *  through the pipeline. Allowing `any` here keeps the discriminated-
   *  union narrowing intact while leaving room for those incremental
   *  annotations. Tighten field-by-field as each consumer is adopted. */
  [extra: string]: any;
}

export interface IRLit extends IRBase {
  kind: 'lit';
  value: number | string | boolean;
}

export interface IRConst extends IRBase {
  kind: 'const';
  name: string;
}

export interface IRRef extends IRBase {
  kind: 'ref';
  ns: 'self' | '%local' | string;
  name: string;
}

export interface IRHole extends IRBase {
  kind: 'hole';
}

/**
 * Axis label `.name` (spec §05 Axis names; FlatPIR `(%axis name)`).
 * Legal only inside an aggregate(...) call; the analyzer enforces.
 */
export interface IRAxis extends IRBase {
  kind: 'axis';
  name: string;
}

/**
 * Static outer-axis metadata attached to measure-op IR nodes
 * (engine-concepts §18.11 / P3a). Lists the outer iteration axes
 * surrounding a node — the spec-§04 / §06 "axis context" made
 * explicit at the IR layer.
 *
 * Each entry describes ONE outer axis, in OUTER-TO-INNER order:
 * the first entry is the outermost axis, the last is the
 * innermost. For example, `iid(broadcast(K, args), 5)` annotated
 * on the outer `iid` call carries
 *
 *   axisStack: [
 *     { source: 'iid',              size: 5 },
 *     { source: 'kernel_broadcast', size: <len of args> },
 *   ]
 *
 * The atom (sampling-time `N`) axis is NOT carried in IR
 * axisStack — it's an engine-internal concept that materialise-
 * time prepends. Backends without an atom-axis concept (RooFit,
 * Stan) consume IR axisStack as-is.
 *
 * Sources:
 *  - 'iid'                — added by `iid(M, n)` (size = n)
 *  - 'broadcast'          — added by value `broadcast(f, args)`
 *                           (size = axis length of args)
 *  - 'kernel_broadcast'   — added by `broadcast(K, args)` for a
 *                           kernel head (size = axis length of args)
 *  - 'aggregate'          — output axis of `aggregate(...)`
 *                           (name = axis name; size = dim length
 *                           if statically known)
 *
 * The `size` field is `number` when statically known, `string`
 * when symbolic (binding name, axis name, '%dynamic'). The
 * `name` field is set for aggregate axes (axis name) and for
 * kernel/value broadcast (typically the broadcast arg's binding
 * name, when available).
 *
 * P3a defines the schema and populates it for the simple cases
 * (iid / broadcast / aggregate at IR top level). Consumers
 * (materialiser, density walker) don't read axisStack yet — that
 * lands with fusion thread (b). Adding new sources or refining
 * the size/name resolution is additive: existing variants and
 * callers are unaffected.
 */
export interface AxisStackEntry {
  source: 'iid' | 'broadcast' | 'kernel_broadcast' | 'aggregate';
  size: number | string;
  name?: string;
}
export type AxisStack = AxisStackEntry[];

/**
 * Call IR — built-in (uses `op`) or user-defined (uses `target`).
 * Field forms (record, joint, jointchain, cartprod, table) carry
 * `fields: [{name, value}, …]`. Module-load forms carry `assigns`.
 */
export interface IRCall extends IRBase {
  kind: 'call';
  op?: string;
  target?: { ns: string; name: string };
  args?: IRNode[];
  kwargs?: Record<string, IRNode>;
  fields?: Array<{ name: string; value: IRNode }>;
  assigns?: Record<string, IRNode>;
  // Reification-specific:
  params?: string[];
  paramKwargs?: string[];
  paramSources?: any[];
  body?: IRNode;
  // Select-IR specific (engine-concepts §11):
  branches?: any[];
  logweights?: IRNode[] | null;
  selectorName?: string | null;
  selectorBase?: number | null;
  // Static axis-context metadata (P3a; engine-concepts §18.11 /
  // §20.10.5 item 4). Populated by `propagateAxisStack` in
  // dissolver.ts for measure-op IR nodes whose variate carries
  // outer iteration axes (iid / kernel_broadcast / aggregate).
  // Not all calls have this — only measure-op calls in a
  // recognised axis-introducing shape.
  axisStack?: AxisStack;
}

export type IRNode = IRLit | IRConst | IRRef | IRHole | IRAxis | IRCall;

// ---------------------------------------------------------------------
// Derivation kinds (engine/derivations.ts → buildDerivations output)
// ---------------------------------------------------------------------

export type DerivationKind =
  | 'alias'
  | 'array'
  | 'tuple'
  | 'record'
  | 'sample'
  | 'evaluate'
  | 'weighted'
  | 'normalize'
  | 'superpose'
  | 'iid'
  | 'randsample'
  | 'jointchain'
  | 'truncate'
  | 'pushfwd'
  | 'bayesupdate'
  | 'logdensityof'
  | 'totalmass'
  | 'broadcast_logdensity'
  | 'select'
  | 'kernelbroadcast'
  | 'mvnormal'
  | 'dirichlet'
  | 'multinomial'
  | 'wishart'
  | 'inversewishart'
  | 'lkjcholesky'
  | 'lkj'
  | 'binnedpoissonprocess';

/**
 * Base derivation shape. Each per-kind interface below extends this
 * and refines `kind` to the literal tag plus declares its own
 * fields. The `Derivation` union below is the canonical consumer
 * type; `DerivationBase` is kept for cross-mode walkers that
 * intentionally don't narrow (e.g. derivationRefsValid).
 */
export interface DerivationBase {
  kind: DerivationKind;
  /** Binding name this derivation is associated with. Required for
   *  named derivations in the buildDerivations output (which is keyed
   *  by name); internal synthesised derivations the materialiser
   *  constructs in flight may omit it. */
  name?: string;
  [extra: string]: any;
}

// Per-kind derivation interfaces. Field shapes mirror the classifier
// + materialiser-builder sites in derivations.ts / materialiser.ts;
// see those files for the semantic meaning of each field. The
// universal `name?` is set when the derivation belongs to a named
// binding (i.e. it's keyed by name in the buildDerivations output);
// internal synthesised derivations the materialiser builds in flight
// omit it.

/** Alias to another binding — shares samples / logWeights / metadata. */
export interface DerivationAlias {
  kind: 'alias';
  name?: string;
  from: string;
}

/** Numeric array literal (e.g. `xs = [1, 2, 3]`). */
export interface DerivationArray {
  kind: 'array';
  name?: string;
  values: number[];
}

/** Positional joint over named-binding refs — tuple shape. */
export interface DerivationTuple {
  kind: 'tuple';
  name?: string;
  elems: string[];
}

/** Record-typed joint over named-binding refs. */
export interface DerivationRecord {
  kind: 'record';
  name?: string;
  /** Field name → referenced binding name. */
  fields: Record<string, string>;
}

/** Sample N draws from `distIR` per atom (the universal leaf).
 *
 *  `logTotalmass` overrides the default 0 for unnormalised reference
 *  measures whose sampling shape coincides with a probability measure
 *  (e.g. `Lebesgue(interval(a, b))` samples like `Uniform(interval(a,
 *  b))` but carries totalmass `b − a` rather than 1). matSample reads
 *  this; consumers that downstream `normalize` discard it, consumers
 *  that read totalmass directly (`totalmass(M)`, plotting an
 *  unnormalised measure) see the spec-canonical value.
 */
export interface DerivationSample {
  kind: 'sample';
  name?: string;
  distIR: IRNode;
  logTotalmass?: number;
}

/** Element-wise deterministic evaluation of `ir` over upstream sample arrays. */
export interface DerivationEvaluate {
  kind: 'evaluate';
  name?: string;
  ir: IRNode;
}

/**
 * Weighted variant — either a closed-form log-shift (`logShift`) or an
 * arbitrary weight expression (`weightIR`). `isLog` distinguishes the
 * surface `weighted(w, M)` (linear) from `logweighted(lw, M)` (already
 * log-domain) at materialise time.
 */
export interface DerivationWeighted {
  kind: 'weighted';
  name?: string;
  from: string;
  logShift?: number;
  weightIR?: IRNode;
  isLog?: boolean;
}

/** Normalise `from` measure by dividing through its totalmass. */
export interface DerivationNormalize {
  kind: 'normalize';
  name?: string;
  from: string;
}

/** Mixture-of-measures (`superpose`). */
export interface DerivationSuperpose {
  kind: 'superpose';
  name?: string;
  fromNames: string[];
}

/** IID measure: `iid(M, n)` — atom-major buffer of shape [N, n]. */
export interface DerivationIid {
  kind: 'iid';
  name?: string;
  from: string;
  /** Per-axis sizes (multi-axis iid is single-axis today, dims = [n]). */
  dims: number[];
}

/**
 * Demand-driven composite `rand` draw (engine-concepts §17.4 stage 2).
 * The draw half (`samples, _ = rand(state, iid(M, count))`, i.e.
 * `tuple_get(<rand>, 0)`) for a COMPOSITE inner measure `M` — one the
 * per-draw measure walker (sampler.walk) can't sample (a forward `lawof` of a
 * broadcast / aggregate, a `pushfwd`, …). Materialised on demand by
 * drawing `count` independent realizations of `from` in a child ctx
 * seeded off `stateIR`. Leaf-distribution inner stays on the batched
 * `sampleLeafN` path (see `classifyRandSample`'s leaf gate).
 */
export interface DerivationRandSample {
  kind: 'randsample';
  name?: string;
  /** Measure-binding ref name to draw from (the iid's inner / the bare M). */
  from: string;
  /** Number of iid draws (the iid `size`; 1 for a bare `rand(state, M)`). */
  count: number;
  /** IR of the rand's first arg (the rng state) — resolved at materialise. */
  stateIR: any;
}

/** Truncate `from` measure to `setDescr`. */
export interface DerivationTruncate {
  kind: 'truncate';
  name?: string;
  from: string;
  setDescr: any;
}

/** Pushforward of `from` measure through function-binding `fnRef`. */
export interface DerivationPushfwd {
  kind: 'pushfwd';
  name?: string;
  from: string;
  fnRef: string;
}

/** Bayesian update — posterior = importance-reweighted prior. */
export interface DerivationBayesupdate {
  kind: 'bayesupdate';
  name?: string;
  from: string;
  /** Likelihood-kernel body source: a named binding (bodyName) or
   *  inline call IR (bodyIR). Exactly one is set. */
  bodyName: string | null;
  bodyIR: IRNode | null;
  obsIR: IRNode;
  /** The kernel's parametric (reified-boundary) input names. matBayesupdate
   *  FEEDS these from the prior's atoms — per the spec lowering
   *  bayesupdate(L,prior)=logweighted(fn(logdensityof(L,_)),prior) the prior's
   *  variate IS the kernel's parametric input — rather than re-materialise a
   *  like-named module binding via getMeasure (audit §3 / H1/H6). */
  paramKwargs?: string[];
  params?: string[];
}

/** `logdensityof(M, x)` — evaluate density at fixed observation `x`. */
export interface DerivationLogdensityof {
  kind: 'logdensityof';
  name?: string;
  measureName: string;
  obsIR: IRNode;
}

/** Standalone likelihood density (spec §06, audit H2):
 *  `logdensityof(L, θ)` with L = likelihoodof(K, obs) scores the kernel
 *  at the GIVEN θ against the FIXED obs — pdf(κ(θ), obs). Carries the
 *  same L→K payload as DerivationBayesupdate plus the evaluation point;
 *  matLikelihoodDensity feeds paramKwargs EXPLICITLY from θ (no prior). */
export interface DerivationLikelihoodDensity {
  kind: 'likelihood_density';
  name?: string;
  bodyName: string | null;
  bodyIR: IRNode | null;
  obsIR: IRNode;
  paramKwargs: string[];
  params: string[];
  pointIR: IRNode;
}

/** `totalmass(M)` — surface measure's tracked totalmass as scalar value. */
export interface DerivationTotalmass {
  kind: 'totalmass';
  name?: string;
  measureName: string;
}

/** `broadcast(logdensityof, M, points)` — vectorised density. */
export interface DerivationBroadcastLogdensity {
  kind: 'broadcast_logdensity';
  name?: string;
  measureName: string;
  pointsIR: IRNode;
}

/** First-class jointchain / kchain (engine-concepts §10 consume/rest). */
export interface DerivationJointchain {
  kind: 'jointchain';
  name?: string;
  /** kchain ⇒ true (keep last only); jointchain ⇒ false. */
  marginalize: boolean;
  /** Kwarg form ⇒ field names; positional ⇒ null. */
  labels: string[] | null;
  steps: any[];
}

/** Discrete-selector mixture (`select` IR, also reached by ifelse/get). */
export interface DerivationSelect {
  kind: 'select';
  name?: string;
  branches: any[];
  /** Per-branch log-weights (closed-form) or null when weights are
   *  computed at materialise time from a runtime selector. */
  logweightIRs: IRNode[] | null;
  selectorRef?: string | null;
  selectorBase?: number;
  marginalize?: boolean;
  mode?: string;
  /** CONSTANT per-branch weights for a no-external-selector mixture /
   *  superpose (engine-concepts §12): matSelect synthesizes the selector
   *  (Bernoulli/Categorical over the normalised weights) when selectorRef
   *  is absent. Folds in the former materialiseSelectIR. */
  synthWeights?: number[] | null;
}

/** Stochastic kernel-broadcast (`broadcast(Normal, mus, sigmas)`). */
export interface DerivationKernelBroadcast {
  kind: 'kernelbroadcast';
  name?: string;
  distOp: string;
  argIRs: IRNode[];
  kwargIRs: Record<string, IRNode> | null;
}

/** MvNormal via Cholesky factorisation of `distIR`'s covariance. */
export interface DerivationMvNormal {
  kind: 'mvnormal';
  name?: string;
  distIR: IRNode;
}

/** Dirichlet(alpha) — atom is a probability simplex vector. */
export interface DerivationDirichlet {
  kind: 'dirichlet';
  name?: string;
  distIR: IRNode;
}

/** Multinomial(n, p) — atom is a length-K integer count vector. */
export interface DerivationMultinomial {
  kind: 'multinomial';
  name?: string;
  distIR: IRNode;
}

/** Wishart(nu, scale) — atom is an n×n SPD matrix (Bartlett decomposition). */
export interface DerivationWishart {
  kind: 'wishart';
  name?: string;
  distIR: IRNode;
}

/** InverseWishart(nu, scale) — atom is an n×n SPD matrix (inverse of Wishart). */
export interface DerivationInverseWishart {
  kind: 'inversewishart';
  name?: string;
  distIR: IRNode;
}

/** LKJCholesky(n, eta) — atom is an n×n lower-triangular Cholesky factor of a correlation matrix. */
export interface DerivationLKJCholesky {
  kind: 'lkjcholesky';
  name?: string;
  distIR: IRNode;
}

/** LKJ(n, eta) — atom is an n×n correlation matrix (LKJCholesky * LKJCholesky^T). */
export interface DerivationLKJ {
  kind: 'lkj';
  name?: string;
  distIR: IRNode;
}

/** BinnedPoissonProcess(rates) — atom is a length-K integer count vector of independent Poisson counts. */
export interface DerivationBinnedPoissonProcess {
  kind: 'binnedpoissonprocess';
  name?: string;
  distIR: IRNode;
}

/** Discriminated union over every kind buildDerivations may emit. */
export type Derivation =
  | DerivationAlias
  | DerivationArray
  | DerivationTuple
  | DerivationRecord
  | DerivationSample
  | DerivationEvaluate
  | DerivationWeighted
  | DerivationNormalize
  | DerivationSuperpose
  | DerivationIid
  | DerivationRandSample
  | DerivationJointchain
  | DerivationTruncate
  | DerivationPushfwd
  | DerivationBayesupdate
  | DerivationLogdensityof
  | DerivationLikelihoodDensity
  | DerivationTotalmass
  | DerivationBroadcastLogdensity
  | DerivationSelect
  | DerivationKernelBroadcast
  | DerivationMvNormal
  | DerivationDirichlet
  | DerivationMultinomial
  | DerivationWishart
  | DerivationInverseWishart
  | DerivationLKJCholesky
  | DerivationLKJ
  | DerivationBinnedPoissonProcess;

// ---------------------------------------------------------------------
// BindingInfo (engine/analyzer.ts → ParsedModule.bindings entries)
// ---------------------------------------------------------------------
//
// What the analyzer attaches to each `name → BindingInfo` map entry.
// The shape is incrementally tightened: required fields land here as
// they're stabilised; the [extra: string]: any escape hatch holds the
// fields engine internals attach in flight (effectiveValue,
// originLoc, synthetic, dependsOn, bijection, …).

/**
 * Statement classification emitted by `analyzer.classifyStatement` —
 * the canonical shape tag every binding carries. Per-binding code
 * branches on this string discriminator.
 */
export type BindingType =
  | 'call'           // RHS is a call expression / op invocation
  | 'literal'        // RHS is a literal value (number, string, boolean, array, record)
  | 'draw'           // RHS is `draw(<measure>)` — stochastic variate
  | 'lawof'          // RHS is `lawof(<expr>)` — measure-of-variate
  | 'input'          // RHS is `elementof(<set>)` or `external(<type>)`
  | 'functionof'     // RHS is a function reification
  | 'kernelof'       // RHS is a kernel reification
  | 'fn'             // RHS is `fn(<expr-with-holes>)` shorthand
  | 'bijection'      // RHS is `bijection(f, f_inv, logvolume)`
  | 'likelihood'     // RHS is `likelihoodof(<kernel>, <obs>)`
  | 'bayesupdate'    // RHS is `bayesupdate(<likelihood>, <prior>)`
  | 'module'         // RHS is `load_module` / `standard_module`
  | 'data';          // RHS is `load_data(<source>, <valueset>)`

/**
 * Binding stored in the analyzer's `bindings` Map. Every binding has at
 * minimum `name` + `type` + `node`. The cached lowered IR (`ir`) is
 * populated by liftInlineSubexpressions before classification; the
 * inferred type + phase land later in the pipeline.
 *
 * Most engine modules read these via the index signature for now (the
 * shape grows over time; promoting more fields to first-class lives in
 * the incremental tightening). Tightening order matches the
 * classifier-side narrowing roadmap.
 */
export interface BindingInfo {
  name: string;
  type: BindingType;
  /** AST node — the AssignStatement this binding was lowered from. */
  node?: any;
  /** Lowered IR (post liftInlineSubexpressions). Cached for fast reads. */
  ir?: IRNode;
  /** Effective RHS AST (multi-LHS pass attaches a per-name projection). */
  effectiveValue?: any;
  /** Inferred type (typeinfer.ts output). */
  inferredType?: any;
  /** Phase: 'fixed' | 'parameterized' | 'stochastic'. */
  phase?: 'fixed' | 'parameterized' | 'stochastic';
  /** Bijection metadata for bijection-typed bindings. */
  bijection?: { fName: string; fInvName: string; logVolume: any;
                registryName?: string; paramIRs?: Record<string, any> };
  [extra: string]: any;
}

/**
 * The output shape of `buildDerivations(bindings)` — the canonical
 * orchestrator surface every consumer (viewer Ctx.derivationsState,
 * materialiser dispatch, density walker) reads through.
 *
 * `bindings` is the post-lift binding Map (some lifts inject extra
 * synthetic bindings). `derivations` is keyed by binding name; values
 * are loose for now (DerivationBase) — see DerivationKind for the
 * tag inventory. `fixedValues` is the pre-eval cache for fixed-phase
 * bindings; `discrete` is per-name resolved-leaf discreteness.
 */
export interface DerivationsState {
  bindings: Map<string, BindingInfo>;
  derivations: Record<string, Derivation>;
  fixedValues: Map<string, any>;
  discrete: Record<string, boolean>;
  /** Classifier diagnostics surfaced by buildDerivations (e.g.
   *  fixed-phase dead ends). Empty when nothing to report. */
  diagnostics?: Array<{ message: string; [extra: string]: any }>;
  /** alias → resolved module descriptor, copied from the lowered
   *  module by the viewer bridge. Populated by pir.lowerToModule. */
  moduleRegistry?: Record<string, any>;
}

// ---------------------------------------------------------------------
// EmpiricalMeasure (engine-concepts §2)
// ---------------------------------------------------------------------
//
// The universal value: every binding's runtime value at evaluation
// time is one of these (or a record/tuple/array composite). Materialiser
// handlers produce these; downstream consumers (viewer, density walker,
// joint diagnostics) read them via shape-aware accessors.

export interface EmpiricalMeasure {
  /** Scalar-leaf storage (length N for scalar measures, atom-major for arrays). */
  samples?: Float64Array;
  /** Per-atom log-weights; null ⇒ uniform 1/N. */
  logWeights: Float64Array | null;
  /** Log-total-mass (algebraically propagated where closed-form, empirical otherwise). */
  logTotalmass: number;
  /** Kish-style effective sample size. */
  n_eff: number;

  // ---- shape-rich variants ----
  /** Shape-tagged Value view (populated on every scalar-leaf measure). */
  value?: Value;
  /** Inner-array shape for atom-batched-vector measures (matIid, MvNormal). */
  dims?: number[];
  /** Composite-record measure: field-name → sub-measure. */
  fields?: Record<string, EmpiricalMeasure>;
  /** Composite-tuple measure: positional sub-measures. */
  elems?: EmpiricalMeasure[];
  /** Shape discriminator for array / tuple variants. */
  shape?: 'array' | 'tuple';
  /** Complex-valued measures: parallel imaginary buffer + dtype tag. */
  imag?: Float64Array;
  dtype?: 'complex' | 'f64';
}

// ---------------------------------------------------------------------
// Histogram (engine/histogram.ts)
// ---------------------------------------------------------------------
//
// Output of integerHistogram / freedmanDiaconisHistogram — the two
// estimators the viewer consumes for sample-mode plots. `reference`
// discriminates: counting (discrete bars, no binEdges/binWidth) vs
// lebesgue (continuous bins).

export interface HistogramResult {
  xs: Float64Array;
  ys: Float64Array;
  support: number[];
  reference: 'counting' | 'lebesgue';
  /** Bin edges (lebesgue only — length === xs.length + 1). */
  binEdges?: Float64Array;
  /** Common bin width (lebesgue only). */
  binWidth?: number;
}

// Module — must export something for TS to treat as a module file
// (otherwise the `declare global` block would be in a script context).
// All declarations above are already exported; this is harmless.
export {};
