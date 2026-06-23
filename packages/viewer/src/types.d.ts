// Type-only imports from the engine package — pulled in via the
// cross-package subpath. `import type` is erased at runtime, so the
// viewer bundle stays decoupled from the engine bundle at build time;
// the types just sharpen our Ctx field declarations below.
import type {
  DerivationsState as EngineDerivationsState,
  EmpiricalMeasure as EngineEmpiricalMeasure,
  HistogramResult as EngineHistogramResult,
} from '@flatppl/engine/engine-types';

// Ambient declarations for the viewer's host environment.
//
// The viewer is a browser IIFE bundle loaded into a webview (VS Code) or
// a standalone HTML page (web gallery, embed-test.html). Several runtime
// dependencies arrive as globals injected by <script> tags BEFORE the
// viewer bundle runs:
//
//   - FlatPPLEngine   — packages/engine bundle (engine.min.js)
//   - echarts         — chart library
//   - cytoscape       — graph layout / rendering
//   - dagre / cytoscape-dagre / cytoscape-bubblesets / cytoscape-layers
//
// They are not imported because the host already loaded them; tsc just
// needs to know they exist.
//
// The FlatPPLViewer global is the viewer's own export surface — set up
// by src/index.js and consumed by the host page.

declare global {
  // --- runtime globals injected by the host page ---

  // The engine bundle is large and dynamically structured; we leave it as
  // `any` so each call site doesn't pull in a full d.ts of the engine
  // surface (which lives in packages/engine and would couple the two
  // packages tightly via types). Tighten per-call-site with JSDoc casts
  // when a particular shape matters.
  const FlatPPLEngine: any;

  const echarts: any;
  const cytoscape: any;
  const dagre: any;

  // VS Code webview API factory. Present only inside a webview; bare-global
  // because host code references it without `window.` (matches the
  // vscode-webview runtime contract).
  function acquireVsCodeApi(): VsCodeApi;

  interface Window {
    FlatPPLViewer?: FlatPPLViewerGlobal;
    __FLATPPL_CONFIG__?: FlatPPLConfig;
    acquireVsCodeApi?: () => VsCodeApi;
  }

  // The viewer's own global. index.js merges into a pre-existing
  // FlatPPLViewer if the host populated one — see src/index.js.
  interface FlatPPLViewerGlobal {
    mount?: (container: HTMLElement, opts?: MountOpts) => void;
    [extra: string]: any;
  }

  interface FlatPPLConfig {
    samplerWorkerUrl?: string;
    playground?: boolean;
    [extra: string]: any;
  }

  // VsCodeApi is exported below (out of the global block) so it can be
  // imported via `import('./types').VsCodeApi`. The interface alias here
  // keeps `acquireVsCodeApi`'s return type and `window.acquireVsCodeApi`
  // resolvable at the global scope without two separate declarations.
  type VsCodeApi = import('./types').VsCodeApi;
}

/** VS Code webview API surface. */
export interface VsCodeApi {
  postMessage(msg: any): void;
  getState(): any;
  setState(s: any): void;
}

// --- viewer-internal types ---
// Imported via JSDoc `@import { Ctx } from './types'` in source modules,
// or referenced as `import('./types').Ctx`.

// --- Plot Plan discriminated union ---
//
// buildPlotPlan() in plot-plan.ts returns one of six concrete plan
// shapes, each with a literal `mode` tag. Consumers — render-frame /
// render-plot / render-controls / render-kernel / render-profile /
// render-samples / persist / overrides — dispatch on mode. The union
// captures the structural difference (e.g. only 'profile' has
// sweepKey / outputs); the per-mode interfaces share PlanBase.
//
// Most interior fields are still `any` for v1 — they reference engine
// types (signatures, axes, presets, domains, IR) that don't yet have a
// shared TS-typed surface. Tightening them is the next per-module
// follow-up after the Plan typedef itself lands.

export interface PlanBase {
  name: string;
}

export interface ProfilePlan extends PlanBase {
  mode: 'profile';
  signature: any;
  axes: any[];
  sweepKey: string;
  matchedPresets: any[];
  presetName: string | null;
  outputs: any[];
  outputKey: string | null;
  autoOverride: any;
  matchedDomains: any[];
  domainName: string | null;
  domainAutoOverride: any;
  /** Set lazily by buildProfileControls when the user picks a y-cutoff. */
  yCutoff?: number;
}

export interface KernelSamplePlan extends PlanBase {
  mode: 'kernel-sample';
  signature: any;
  axes: any[];
  matchedPresets: any[];
  presetName: string | null;
  autoOverride: any;
  matchedDomains: any[];
  domainName: string | null;
  domainAutoOverride: any;
}

export interface SamplesPlan extends PlanBase {
  mode: 'samples';
  discrete: boolean;
  analyticalIR: any | null;
}

export interface ArrayPlan extends PlanBase {
  mode: 'array';
}

export interface MatrixPlan extends PlanBase {
  mode: 'matrix';
  // Optional static shape hint. Present when typeinfer pinned both
  // axis lengths to literal integers; absent for computed-shape
  // rank-2 bindings (`iid(M, length(data))`, `cartpow(reals, n)`,
  // etc.). The renderer prefers the runtime measure's
  // `intrinsicShape` when both are available; this field is just a
  // performance hint that lets it skip the lookup.
  shape?: [number, number];   // rows, cols
}

export interface FixedScalarPlan extends PlanBase {
  mode: 'fixed-scalar';
  discrete: boolean;
}

export interface FixedRecordPlan extends PlanBase {
  mode: 'fixed-record';
}

/** Discriminated union over the seven plan modes buildPlotPlan emits. */
export type Plan =
  | ProfilePlan
  | KernelSamplePlan
  | SamplesPlan
  | ArrayPlan
  | MatrixPlan
  | FixedScalarPlan
  | FixedRecordPlan;

/** Per-record-binding selection state set up by renderRecordMarginals. */
export interface RecordSelection {
  bindingName: string;
  mode: 'correlations' | 'marginals' | 'table';
  /** Per-axis selection (correlations mode) — list of axis keys. */
  selected: string[];
  /** Group-level selection (marginals mode) — list of group keys. */
  marginalGroups: string[];
  /** Generated-quantity names currently toggled on (bayesupdate posteriors only).
   *  Default []; toggled names are appended as derived fields via
   *  appendGeneratedQuantities before axis enumeration and rendering. */
  genQuantities?: string[];
}

/**
 * Per-mount context object. All viewer state lives here; every module
 * function that needs it takes `ctx` as its first or last parameter
 * (style varies per module — see each module's own JSDoc). Never
 * reach for it through a closure that wasn't passed in.
 *
 * The contract: ONE ctx per call to mount(). Field tightening from
 * `any` is incremental — the high-value fields (recordSelection,
 * Plan-typed currentPlotPlan, EmpiricalMeasure-typed measureCache,
 * etc.) have already been narrowed; the rest carry the
 * `[extra: string]: any` escape hatch until each consumer is
 * tightened.
 */
export interface Ctx {
  // ---- host adapter ----
  host: HostAdapter;

  // ---- config + constants (set in mount prologue) ----
  CONFIG: FlatPPLConfig;
  HINT: string;
  SAMPLE_COUNT: number;
  /** MC sample count M for marginalizing internal latents in an
   *  intractable density (likelihood / posterior); separate from
   *  SAMPLE_COUNT. VS Code: flatppl.visualization.marginalizationSampleCount. */
  MARGINALIZATION_COUNT: number;
  REJECTION_BUDGET: number;
  /** Inference backend + knobs for posterior (bayesupdate) measures. The
   *  engine's matBayesupdate reads this off the matCtx; 'is' (default) is the
   *  importance-sampling path, 'mh'/'emcee' run the MCMC driver. */
  inferenceOpts: { backend: string; chains: number; walkers: number | null; warmup: number; draws: number; seed: number | null; amisIters: number; amisSamples: number; smcParticles: number; smcSteps: number; smcCESS: number };
  /** Shared onChange closure for the inference-backend selector. Clears
   *  caches, persists the choice, and re-renders. Set once in main.ts init;
   *  consumed by the record-measure plot toolbar (bayesupdate-only). */
  onInferenceChange?: () => void;
  SAMPLER_WORKER_URL: string;
  HISTORY_CAP: number;
  CORRELATIONS_MAX_AXES: number;
  MODULE_TARGET: string;
  PALETTE: Record<string, string>;
  PHASE_COLORS: Record<'stochastic' | 'parameterized' | 'fixed', string>;
  DRAW_EDGE_COLOR: string;
  TYPE_STYLE: Record<string, { color: string; shape: string; label: string }>;
  CODICON_PATHS: Record<string, string>;

  // ---- DOM root + cytoscape ----
  /** Container element handed to mount(). */
  X?: HTMLElement;
  cy: any;
  bb: any;

  // ---- navigation history ----
  history: any[];
  currentState: any;

  // ---- sampler worker ----
  samplerWorker: Worker | null;
  samplerWorkerPromise: Promise<Worker> | null;
  samplerWorkerError: Error | null;
  samplerReqId: number;
  pendingRequests: Map<number, { resolve: (v: any) => void; reject: (e: any) => void; onProgress?: (m: any) => void }>;
  /** Set by render-plot before an MCMC/AMIS run; runMcmcPool calls it with an
   *  aggregated fraction in [0,1] and phase string so a determinate progress
   *  bar can be drawn. Cleared when the run settles. */
  onSamplingProgress?: ((frac: number, phase: string) => void) | null;
  mcmcPool?: any[];   // transient worker pool for an in-flight MCMC posterior run

  // ---- current-render state ----
  plotEchart: any;
  plotEnabled: boolean;
  currentSource: string | null;
  /** Engine's binding map from processSource — Map<name, BindingInfo>.
   *  The PRIMARY module alone (the DAG renders this). Null at boot before
   *  the first source loads. */
  currentBindings: Map<string, any> | null;
  /** Engine's LINKED (flattened) binding map — primary + every
   *  transitively-loaded module spliced under namespaced names (spec §04
   *  load_module). Drives derivation building / materialisation so
   *  cross-module refs resolve. Equals `currentBindings` for a single-file
   *  model. Null at boot. */
  currentLinkedBindings: Map<string, any> | null;
  currentLoweredModule: any;
  currentVariantId: string | null;
  currentPlotBindingName: string | null;
  currentPlotPlan: Plan | null;
  /** Per-record-binding selection state. */
  recordSelection: RecordSelection | null;
  rootSeed: number;

  // ---- caches ----
  /** Output of buildDerivations — bindings, derivations, fixedValues,
   *  discrete. See engine-types.d.ts DerivationsState. Null at boot
   *  before the first source is loaded; populated by mountViewer's
   *  applySourceUpdate path. */
  derivationsState: EngineDerivationsState | null;
  /** Atom-major samples keyed by binding name; sub-fields populated per
   *  measure shape (scalar / array / record / tuple / complex). The
   *  engine's shape is reused so the engine and viewer agree on the
   *  empirical-measure contract — see engine-types.d.ts EmpiricalMeasure. */
  measureCache: Map<string, EngineEmpiricalMeasure>;
  /** Per-binding histogram cache. The viewer keys by name + discrete
   *  flag; values come from histogram.integerHistogram /
   *  freedmanDiaconisHistogram. See engine-types HistogramResult. */
  histogramCache: Map<string, EngineHistogramResult>;
  /** Cached auto-fit ranges, keyed by `${planName}|${kwarg}|D=${domainName}`.
   *  `fromAuto: true` indicates the range was computed by resolveSweepRange
   *  rather than user-set. */
  profileRangeCache: Map<string, { lo: number; hi: number; fromAuto?: boolean }>;
  /** Per-binding MLE point for likelihood plots, populated best-effort in the
   *  background (populateModeCache) and offered as the labelled `auto (MLE)`
   *  preset. 'failed' (incl. timeout) → the option is omitted. Cleared on every
   *  source rebuild. */
  modeCenterCache?: Map<string, {
    status: 'pending' | 'ready' | 'failed';
    /** Per-kwarg MLE: a scalar input → number; an array input → number[]. */
    values?: Record<string, number | number[]>;
    /** Per-AXIS-KEY Laplace curvature std (x-space) at the MLE — keyed like
     *  distributeAxes (`mu`, `theta[3]`) — the half-width a likelihood plot
     *  uses to auto-frame each sweep axis's peak (mode ± k·sd). */
    sd?: Record<string, number>;
    reason?: string;
  }>;
  /** Memoised plan-selection state per binding name. */
  planMemoryByName: Map<string, any>;
  presetOverrides: Map<string, any>;
  domainOverrides: Map<string, any>;
  pendingPresetName: string | null;
  pendingDomainName: string | null;

  // ---- escape hatch for fields that haven't been typed yet ----
  // Keep last; tighten and remove as modules are converted.
  [extra: string]: any;
}

/**
 * Host adapter. Each method is optional; viewer call sites guard with
 * `if (host.foo)` so a missing host method becomes a no-op. The default
 * adapter (host-adapter.js / defaultVscodeHost) wires the four methods
 * to vscode-webview postMessage / getState / setState; non-VS-Code hosts
 * (web gallery, embed-test) provide their own implementations.
 */
export interface HostAdapter {
  postMessage?(msg: any): void;
  getState?(): any;
  setState?(s: any): void;
  revealSourceLine?(line: number, name?: string): void;
  setTitle?(title: string): void;
  /** Navigate the host to a loaded module's file (spec §04 load_module).
   *  `path` is the resolved module path (the bundle / router key). Fired
   *  by double-clicking a `load_module` node in the DAG. */
  openModule?(path: string): void;
  [extra: string]: any;
}

/** Arguments accepted by FlatPPLViewer.mount(). */
export interface MountOpts {
  host?: HostAdapter;
  [extra: string]: any;
}

// Module — must export something for TS to treat as a module file
// (otherwise the `declare global` block is in a script context, not a
// module-augmentation context, and `import('./types').Ctx` doesn't work).
export {};
