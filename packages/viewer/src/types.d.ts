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

/**
 * Per-mount context object. Every closure-captured state used by the
 * pre-Phase-4 IIFE was moved onto this object in Phase 3; Phase 4 then
 * split the file into modules that all take a `ctx` parameter.
 *
 * The contract: ONE ctx per call to mount(). Every module function that
 * needs viewer state takes `ctx` as its first or last parameter (style
 * varies per module — see each module's own JSDoc); never reach for it
 * through a closure that wasn't passed in.
 *
 * v1 (this commit) leaves most fields permissive (`any`) so the JS
 * modules don't light up red at once. Tighten per-field as modules are
 * converted — the high-value fields are the ones whose Phase-4 misuse
 * caused the post-decomposition fixes (recordSelection, formatScalar,
 * etc. — see CONVENTIONS.md and the post-Phase-4 commit messages
 * 5113732 / 7174cf1 / 3435094 for the bug class TS catches).
 */
export interface Ctx {
  // ---- host adapter ----
  host: HostAdapter;

  // ---- config + constants (set in mount prologue) ----
  CONFIG: FlatPPLConfig;
  HINT: string;
  SAMPLE_COUNT: number;
  REJECTION_BUDGET: number;
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
  pendingRequests: Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>;

  // ---- current-render state ----
  plotEchart: any;
  plotEnabled: boolean;
  currentSource: string | null;
  currentBindings: any;
  currentLoweredModule: any;
  currentVariantId: string | null;
  currentPlotBindingName: string | null;
  currentPlotPlan: any;
  /** Per-record-binding selection map (Phase-4 fix migrated this onto ctx). */
  recordSelection: Map<string, any> | null;
  rootSeed: number;

  // ---- caches ----
  derivationsState: any;
  measureCache: Map<string, any>;
  histogramCache: Map<string, any>;
  profileRangeCache: Map<string, any>;
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
