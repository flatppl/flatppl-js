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
  weightsFrom?: { ref: string; K: number; base: number } | null;
  selectorName?: string | null;
  selectorBase?: number | null;
}

export type IRNode = IRLit | IRConst | IRRef | IRHole | IRCall;

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
  | 'jointchain'
  | 'truncate'
  | 'pushfwd'
  | 'bayesupdate'
  | 'logdensityof'
  | 'totalmass'
  | 'broadcast_logdensity'
  | 'select'
  | 'kernelbroadcast'
  | 'mvnormal';

/**
 * Base derivation shape — every kind has at minimum `kind` + `name`.
 * Per-kind fields vary (distIR / args / steps / branches / etc.); a
 * full discriminated union is the per-kind tightening work that
 * lands alongside materialiser per-kind handler refactors.
 */
export interface DerivationBase {
  kind: DerivationKind;
  name: string;
  [extra: string]: any;
}

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
  bijection?: { fName: string; fInvName: string; logVolume: any };
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
  derivations: Record<string, DerivationBase>;
  fixedValues: Map<string, any>;
  discrete: Record<string, boolean>;
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
  /** Shape-tagged Value view (Phase 4b — populated on every scalar-leaf measure). */
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
