// @flatppl/viewer main module
//
// All IIFE-scope content (the hoisted 117 functions + the mount
// definition + per-mount-constants prologue inside mount) lives
// here as ES module top-level.will progressively split
// this into the module map documented in the header below.
//
// The only export this module needs to surface is `mount`; the
// thin index.js entry handles the FlatPPLViewer global merge and
// DOMContentLoaded auto-mount.



  // Layout markup + stylesheet for the viewer. The next pass moves
  // them out of the host page into here so any container can
  // host the viewer without the host having to know the
  // internal DOM shape. The CSS goes once into <head>; the
  // markup goes into the supplied container.

import {
  $,
  esc,
  truncateExpr,
  samplesAreConstant,
  formatScalar,
  formatComplexScalar,
  complexReBadge,
  formatArrayParts,
  formatArrayWithEllipsis,
  formatIRValue,
  formatValue,
  formatLogTotalmass,
  formatCount,
  formatSampleCount,
  qualityTooltip,
  measureAtomCount,
  hexToRgba,
  makeMainThreadPrng,
  plotZoomOptions,
  listScalarAxes,
  resolveMeasureAlias,
  defaultValueForLeafType,
  defaultRangeForLeafType,
  rangeFromSetDescriptor,
  isPersistableSetField,
  presetValuesText,
  domainBoundsText,
  filterOverrideToAxes,
  bubbleMemberIds,
} from './util.js';


import {
  colorForBinding,
  resolveNodeColor,
} from './palette.js';
import {
  cancelAllSampling,
  ensureSamplerWorker,
  sendWorker,
  sendWorkerNow,
  wireWorker,
} from './worker.js';
import {
  collectRefArrays,
  fixedValueToMeasure,
  getMeasure,
  tryGetMeasure,
} from './engine-facade.js';


import {
  activeDomainRangesFor,
  activeFixedNamesFor,
  activePresetFor,
  applyRememberedSelections,
  baseRangesFor,
  baseValuesFor,
  computeAutoValues,
  domainOverrideEntryFor,
  ensureDomainOverrideFor,
  ensureOverrideFor,
  hasDomainOverrides,
  hasOverrides,
  overrideEntryFor,
  rememberPlanSelections,
  resolveSweepRange,
  setDomainOverrideFor,
  setOverrideFor,
} from './overrides.js';
import {
  buildPersistedDomainLine,
  buildPersistedPresetLine,
  canPersistActive,
  canPersistDomain,
  defaultSetSourceForKwarg,
  formatScalarForSource,
  persistActive,
  persistAutoAsNewBinding,
  persistAutoDomainAsNewBinding,
  persistDomain,
  persistNamedDomain,
  persistNamedPreset,
  setFieldToSource,
} from './persist.js';
import {
  errorsForBinding,
  makeActionButton,
  renderPlotFrame,
  renderTextValue,
  resetPlotContentStyle,
  setPlotEnabled,
  showPlotMessage,
} from './render-frame.js';


import {
  drawReificationLassos,
  enterModuleView,
  focusNode,
  initCy,
  renderDAG,
  showNodeInfo,
  teardownBubbles,
  updateBackBtn,
  updateHeader,
} from './dag.js';
import {
  formatConstantMeasure,
  measureIsConstant,
  renderAxisDropdown,
  renderConstantRecord,
  renderGroupDropdown,
  renderRecordMarginals,
  renderRecordToolbar,
  renderSampleStats,
} from './render-record.js';
import {
  renderCornerGrid,
  renderDensityStrips,
} from './render-density.js';
import {
  renderArrayStepPlot,
  renderEmpiricalMeasure,
  renderSamplesAndDensity,
} from './render-samples.js';
import {
  buildProfileBottomRow,
  buildProfileControls,
  commitSliceX,
  renderProfileLine,
  renderProfilePlotForCurrent,
} from './render-profile.js';


import {
  buildPlotPlan,
  materialiseConcreteMeasure,
} from './plot-plan.js';
import {
  buildDomainControl,
  buildPresetControl,
} from './render-controls.js';
import {
  renderFixedRecord,
  renderKernelSampleForCurrent,
  renderKernelSampleMeasure,
} from './render-kernel.js';
import {
  renderPlotForCurrent,
  updatePlotForBinding,
} from './render-plot.js';
import {
  rebuildDerivations,
} from './derivations.js';
import {
  applySourceUpdate,
  nameSeed,
  resizeAllEchartsInPlot,
  resizeAndFitCy,
} from './orchestration.js';


import { VIEWER_BODY_HTML, ensureCssInjected } from './templates.js';
import { defaultVscodeHost } from './host-adapter.js';




  // Cache acquireVsCodeApi()'s return value: VS Code permits calling
  // it at most once per webview. If the default host adapter is built
  // more than once (e.g. by re-mount), we hand out the same underlying
  // api object instead of throwing.

  // Default host adapter for VS Code webviews. Bridges the four
  // host-adapter methods to the corresponding postMessage / setState
  // / getState calls. When NOT inside a VS Code webview
  // (acquireVsCodeApi missing), returns an empty object — the
  // viewer's call sites guard each method with `if (host.foo)`, so
  // missing methods become no-ops cleanly.

// -------------------------------------------------------------
// Hoisted pure utilities (leaf L2).
// Each of these is a pure function: no references to per-mount
// ctx state and no calls to in-mount-only helpers. JS function-
// declaration hoisting keeps in-mount callers working unchanged
// (they reach these by the normal scope chain).
// -------------------------------------------------------------





















// -------------------------------------------------------------
// Hoisted ctx-taking utilities.
// Each takes `ctx` (the per-mount state container) as its first
// parameter; every call site in this file has been updated to
// pass ctx. In-mount callers reach ctx via lexical capture (the
// `var ctx = {}` at the top of mount()); IIFE-scope callers
// pass their own ctx parameter through.will turn this
// block into ES modules (palette.js / engine-facade.js / etc.).
// -------------------------------------------------------------










// -------------------------------------------------------------
// Hoisted L3 engine facade + L4 worker subgroup.
// Each takes ctx as first param; in-file call sites have been
// updated to pass ctx. Two callback sites (getMeasure and
// sendWorker passed to FlatPPLEngine.materialiser as 1-arg
// callbacks) are wrapped at the call point to bind ctx — the
// engine's signature contract stays unchanged.
// -------------------------------------------------------------










// ---- Hoisted L4 override store + plot-frame helpers ----



















// ---- Hoisted leaf batch — header/info, leaf defaults,
//      persist-helpers, plan-memory, bubble teardown ----




















// ===================================================================
// Hoisted L4–L7. Every function takes ctx as first
// parameter; in-mount callers reach ctx via lexical capture, all
// other callers pass it through.turns this block into ES
// modules grouped by responsibility (derivations / renderers / dag
// / orchestration / etc.).
// ===================================================================




















































export function mount(container: HTMLElement, opts?: import('./types').MountOpts) {
    opts = opts || {};

    // Per-mount state container. Every
    // captured mutable/shared identifier sits on `ctx` so the nested
    // functions don't rely on lexical capture and the implementation
    // is free to live in its own ES module. `ctx` is declared at the
    // very top of mount so it's
    // initialized BEFORE any `ctx.X = …` assignment in the prologue;
    // var-hoisting alone wouldn't suffice because the assignment
    // `ctx = {}` is what makes ctx an object, and the prologue's
    // first ctx-write (e.g. `ctx.host = …`) must see an object.
    // Cast to Ctx so the property writes below see the full shape
    // (Ctx's [extra: string]: any escape hatch keeps unconverted
    // modules quiet until per-field typing tightens — see
    // src/types.d.ts).
    const ctx = {} as import('./types').Ctx;
    // container: the element the viewer renders inside. Defaults to
    // document.body for backward-compat with the existing VS Code
    // wrapper. The viewer injects its layout markup as innerHTML and
    // ensures its stylesheet is present on the page once.
    container = container || (typeof document !== 'undefined' ? document.body : null);
    if (!container) {
      throw new Error('FlatPPLViewer.mount: no container available (document missing?)');
    }
    ensureCssInjected();
    container.innerHTML = VIEWER_BODY_HTML;
    // Host adapter: IDE-only concerns the viewer delegates outward
    // (cross-pane source navigation, panel-title updates, persistent
    // UI state). Each method is optional; missing methods become
    // no-ops, so a standalone embed can pass {} or omit opts.host
    // entirely and the viewer renders fine — just without the
    // navigate-to-source / restore-state niceties.
    //
    //   revealSourceLine(line)  — host moves its source view's cursor
    //   setTitle(name)          — host sets the surrounding panel title
    //   saveState(state)        — host persists viewer state across reloads
    //   loadState()             — host returns previously-saved state
    //
    // Default: when no host is supplied AND acquireVsCodeApi exists
    // (we're inside a VS Code webview), build a default adapter that
    // bridges to VS Code's postMessage / setState / getState. This
    // keeps the existing extension wrapper working without any
    // host-side changes.
    ctx.host = opts.host || defaultVscodeHost();

    // Host-supplied configuration. The vscode-extension host writes
    // window.__FLATPPL_CONFIG__ via a small inline bootstrap <script>
    // before this file loads. For a standalone embed (no VS Code), an
    // online host can do the same — set the config object before
    // including viewer.js. Currently expected fields:
    //   samplerWorkerUrl: string  — URL of the sampler-worker bundle,
    //                                loaded as a Web Worker.
    ctx.CONFIG = (typeof window !== 'undefined' && window.__FLATPPL_CONFIG__) || {};
    ctx.HINT = 'Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source';
    // Sampler-worker URL. Used lazily — no worker is spawned until the
    // user picks a binding for which the Plot tab is enabled (a 'draw'
    // of a known distribution with literal params).
    ctx.SAMPLER_WORKER_URL = ctx.CONFIG.samplerWorkerUrl || '';


  // ---- Palette ----
  //
  // Single source of truth for every node / edge / bubble colour the
  // visualizer uses. PHASE_COLORS, TYPE_STYLE, DRAW_EDGE_COLOR, and
  // the #info .phase-* CSS rules all reference these names — change a
  // hex here and every consumer follows.
  //
  // Naming reflects what the colour *means*, not where it shows up:
  //   phaseStochastic / parameterized / fixed
  //                          — value-producing nodes (draw, call) and
  //                            their #info phase tags
  //   measure                — lawof bindings + measure-kind reifications
  //   kernel                 — kernelof bindings + kernel-kind reifications
  //   fn                     — functionof / fn bindings
  //   literal/…/unknown      — purely structural type colours
  //   drawEdge               — the "draw" arrow (deterministic →
  //                            stochastic boundary)
  //
  // Hue strategy: lawof/kernelof/functionof form an additive triple
  // (blue + green ≈ teal) so family relationships read visually,
  // viridis-style and colourblind-safe. The phase trio reuses the
  // historical draw/input/call hex values so the visual story stays
  // familiar after the shift to phase-driven colouring.
  ctx.PALETTE = {
    phaseStochastic:    '#B39DDB',  // purple
    phaseParameterized: '#4DD0E1',  // teal
    phaseFixed:         '#90A4AE',  // blue-grey
    measure:            '#42A5F5',  // bright blue
    kernel:             '#26A69A',  // teal-green
    fn:                 '#66BB6A',  // green
    // Note: no `literal` entry — literal bindings are just fixed-
    // phase values, semantically the same kind as `call` bindings,
    // and reuse `phaseFixed` for color. Shape (rectangle vs
    // round-rectangle) carries the surface-form distinction.
    // Using a dedicated red/pink for literals overstated their
    // status and conflicted with red's conventional warning role
    // in dev UIs.
    likelihood:         '#EF9A9A',  // light red
    bayesupdate:        '#FFAB91',  // light orange
    module:             '#80CBC4',  // teal-green (lighter)
    table:              '#A1887F',  // brown
    unknown:            '#BDBDBD',  // grey
    drawEdge:           '#7E57C2',  // darker purple than phaseStochastic
  };

  // Mirror the phase colours into CSS custom properties so the
  // #info .phase-* tag rules pick them up without a duplicate hex
  // literal in the stylesheet.
  (function bindPaletteToCss() {
    const s = document.documentElement.style;
    s.setProperty('--phase-stochastic',    ctx.PALETTE.phaseStochastic);
    s.setProperty('--phase-parameterized', ctx.PALETTE.phaseParameterized);
    s.setProperty('--phase-fixed',         ctx.PALETTE.phaseFixed);
  })();

  // Phase → fill colour for value-producing nodes (draw / call /
  // computed values inside a kernel scope). Used by both the DAG
  // renderer and the legend.
  ctx.PHASE_COLORS = {
    stochastic:    ctx.PALETTE.phaseStochastic,
    parameterized: ctx.PALETTE.phaseParameterized,
    fixed:         ctx.PALETTE.phaseFixed,
  };

  // Stand-alone for the "draw" edge — visually distinct from any
  // node fill so a stochastic boundary reads as an edge, not a fill.
  ctx.DRAW_EDGE_COLOR = ctx.PALETTE.drawEdge;

  // Type → { color, shape, legend label }. The phase trio (input /
  // draw / call) intentionally reuses PALETTE.phase* so a
  // value-producing node falls back to the matching phase colour
  // when phase metadata is missing.
  ctx.TYPE_STYLE = {
    input:       { color: ctx.PALETTE.phaseParameterized, shape: 'diamond',         label: 'input (elementof)' },
    draw:        { color: ctx.PALETTE.phaseStochastic,    shape: 'ellipse',         label: 'draw' },
    call:        { color: ctx.PALETTE.phaseFixed,         shape: 'round-rectangle', label: 'call' },
    lawof:       { color: ctx.PALETTE.measure,            shape: 'round-rectangle', label: 'lawof (measure)' },
    kernelof:    { color: ctx.PALETTE.kernel,             shape: 'round-hexagon',   label: 'kernelof (kernel)' },
    functionof:  { color: ctx.PALETTE.fn,                 shape: 'hexagon',         label: 'functionof' },
    fn:          { color: ctx.PALETTE.fn,                 shape: 'hexagon',         label: 'fn' },
    literal:     { color: ctx.PALETTE.phaseFixed,         shape: 'rectangle',       label: 'literal' },
    likelihood:  { color: ctx.PALETTE.likelihood,         shape: 'octagon',         label: 'likelihood' },
    bayesupdate: { color: ctx.PALETTE.bayesupdate,        shape: 'octagon',         label: 'bayesupdate' },
    module:      { color: ctx.PALETTE.module,             shape: 'round-rectangle', label: 'module' },
    table:       { color: ctx.PALETTE.table,              shape: 'round-rectangle', label: 'table' },
    unknown:     { color: ctx.PALETTE.unknown,            shape: 'rectangle',       label: 'unknown' },
  };


  // G1 DAG/state (on ctx).
  ctx.cy = null;
  ctx.bb = null;
  ctx.history = [];
  ctx.currentState = null;
  // Bound on the DAG-navigation history (back-button stack). Cheap
  // insurance against pathological growth (a runaway extension or
  // rapid-fire navigation). Each entry is a sub-DAG's data plus a
  // name string, so a few hundred is plenty without thinking about
  // memory. Owned by the host setting flatppl.visualization.
  // dagNavigationHistoryCap (default 1000); the host pushes its
  // value via configUpdate alongside sampleCount.
  ctx.HISTORY_CAP = 1000;

  // VS Code codicons — `discard`, `save`, `save-as` paths copied
  // verbatim from microsoft/vscode-codicons (src/icons/*.svg) so the
  // viewer's reset/persist buttons match VS Code's own toolbar
  // language without pulling in a font dependency. SVGs use
  // fill="currentColor" so they inherit the button's text color.
  //
  // © Microsoft Corporation. Licensed under CC BY 4.0
  // (https://creativecommons.org/licenses/by/4.0/). See
  // packages/viewer/NOTICE.md for full attribution.
  ctx.CODICON_PATHS = {
    discard: 'M3.00098 2.5C3.00098 2.22386 3.22483 2 3.50098 2C3.77712 2 4.00098 2.22386 4.00098 2.5V6.34262L7.17202 3.17157C8.73412 1.60948 11.2668 1.60948 12.8289 3.17157C14.391 4.73367 14.391 7.26633 12.8289 8.82843L7.80375 13.8536C7.60849 14.0488 7.2919 14.0488 7.09664 13.8536C6.90138 13.6583 6.90138 13.3417 7.09664 13.1464L12.1218 8.12132C13.2933 6.94975 13.2933 5.05025 12.1218 3.87868C10.9502 2.70711 9.0507 2.70711 7.87913 3.87868L4.75781 7H8.50098C8.77712 7 9.00098 7.22386 9.00098 7.5C9.00098 7.77614 8.77712 8 8.50098 8H3.60098C3.26961 8 3.00098 7.73137 3.00098 7.4V2.5Z',
    save: 'M14.414 3.207L12.793 1.586C12.421 1.213 11.905 1 11.379 1H3C1.897 1 1 1.897 1 3V13C1 14.103 1.897 15 3 15H13C14.103 15 15 14.103 15 13V4.621C15 4.095 14.787 3.579 14.414 3.207ZM9 2V3.5C9 3.776 8.776 4 8.5 4H6.5C6.224 4 6 3.776 6 3.5V2H9ZM5 14V9.5C5 9.224 5.224 9 5.5 9H10.5C10.776 9 11 9.224 11 9.5V14H5ZM14 13C14 13.551 13.551 14 13 14H12V9.5C12 8.673 11.327 8 10.5 8H5.5C4.673 8 4 8.673 4 9.5V14H3C2.449 14 2 13.551 2 13V3C2 2.449 2.449 2 3 2H5V3.5C5 4.327 5.673 5 6.5 5H8.5C9.327 5 10 4.327 10 3.5V2H11.379C11.642 2 11.9 2.107 12.086 2.293L13.707 3.914C13.893 4.1 14 4.358 14 4.621V13Z',
    'save-as': 'M5 9.5C5 9.224 5.224 9 5.5 9H10.5C10.738 9 10.929 9.171 10.979 9.394L11.729 8.644C11.458 8.256 11.009 8 10.5 8H5.5C4.673 8 4 8.673 4 9.5V14H3C2.449 14 2 13.551 2 13V3C2 2.449 2.449 2 3 2H5V3.5C5 4.327 5.673 5 6.5 5H8.5C9.327 5 10 4.327 10 3.5V2H11.379C11.642 2 11.9 2.107 12.086 2.293L13.707 3.914C13.893 4.1 14 4.358 14 4.621V7.04C14.143 7.015 14.289 6.997 14.437 6.997C14.629 6.997 14.817 7.023 15 7.064V4.62C15 4.094 14.787 3.578 14.414 3.206L12.793 1.585C12.421 1.212 11.905 0.999001 11.379 0.999001H3C1.897 1 1 1.897 1 3V13C1 14.103 1.897 15 3 15H7.045L7.293 14H5V9.5ZM6 2H9V3.5C9 3.776 8.776 4 8.5 4H6.5C6.224 4 6 3.776 6 3.5V2ZM16 9.559C16 9.764 15.96 9.967 15.882 10.157C15.803 10.346 15.688 10.519 15.543 10.664L11.254 14.951C10.898 15.307 10.452 15.56 9.964 15.682L8.753 15.982C8.651 16.008 8.544 16.006 8.443 15.978C8.342 15.95 8.249 15.896 8.175 15.822C8.101 15.748 8.047 15.655 8.019 15.554C7.991 15.453 7.99 15.346 8.015 15.244L8.315 14.033C8.437 13.544 8.689 13.098 9.045 12.742L13.333 8.455C13.626 8.162 14.023 7.998 14.437 7.998C14.851 7.998 15.248 8.163 15.541 8.455C15.687 8.599 15.802 8.772 15.881 8.961C15.96 9.151 16 9.354 16 9.559Z',
  };




  // Sentinel name for the module-overview state. Distinct from any
  // user binding name (binding identifiers are barewords; the
  // sentinel uses ':' which the analyzer can't produce). Used by
  // updateHeader, updatePlotForBinding, and the back-button to
  // distinguish module view from a single-binding view.
  ctx.MODULE_TARGET = ':module';




  // ---------------------------------------------------------------
  // Plot panel — density / sample histograms via the sampler-worker.
  //
  // The Plot tab shows the analytical density of the currently focused
  // binding when that binding is a 'draw' of a known distribution with
  // literal parameters (so the worker doesn't need to dependency-walk
  // upstream randoms — that's the orchestrator's job, deferred to a
  // later iteration). When the binding isn't plottable, the Plot tab
  // is shown disabled.
  //
  // The sampler-worker is spawned lazily on first plot request so the
  // ~1 MB worker bundle (stdlib + sampler) doesn't load for users who
  // never open the Plot tab. We keep the worker alive across focus
  // changes; only its 'setSeed' / 'init' is replayed on demand. A
  // request-id counter pairs replies to outstanding promises so we
  // can multiplex multiple in-flight requests cleanly.
  // ---------------------------------------------------------------

  // G3 worker state (on ctx).
  ctx.samplerWorker = null;
  ctx.samplerWorkerPromise = null;   // Promise<Worker> while spawn is in-flight
  ctx.samplerWorkerError = null;     // last spawn error, surfaced in the UI
  ctx.samplerReqId = 0;
  ctx.pendingRequests = new Map(); // id → { resolve, reject }
  // G2 plot control (on ctx). Decls are
  // scattered across mount() — the other three live further down
  // near their first use; collected onto ctx incrementally.
  ctx.plotEchart = null;

  // ---------------------------------------------------------------
  // Main-thread empirical-measure cache.
  //
  // The cache holds an EmpiricalMeasure per binding:
  //   { samples:    Float64Array,            // the atom values
  //     logWeights: Float64Array | null }    // null = uniform 1/N
  //
  // Why a measure (not just samples)? When we add weighted,
  // bayesupdate, and superpose, we'll need per-atom weights to
  // represent the result correctly. Storing the structure now, even
  // with logWeights always null, lets those operations land later
  // without churning every consumer. For unweighted measures the
  // null-uniform convention costs nothing — logSumExp(null) = 0,
  // so total mass = 1 (probability measure), and histograms take
  // the simple count/N path.
  //
  // Why main-thread cache (not worker-side)?
  //   - Survives worker recycles (the user's Stop button terminates
  //     the worker; the cache stays valid).
  //   - Variates and their underlying measures share the SAME
  //     EmpiricalMeasure object (same samples, same logWeights) —
  //     theta1's measure IS theta1_dist's measure, by reference.
  //   - Click-around the DAG hits the cache → instant re-render.
  //   - Source edits invalidate everything by clearing the map.
  //
  // Per-binding seeding: we derive a deterministic seed from a
  // string hash of the binding name XOR'd with a root seed. Two
  // independent random variables (theta1_dist, theta2_dist) thus
  // get statistically independent streams without coupling to the
  // order of materialisation. A future "Resample" button can bump
  // rootSeed and clear the cache to redraw everything.
  // ---------------------------------------------------------------
  ctx.derivationsState = null;       // { derivations, discrete } from orchestrator
  ctx.measureCache = new Map();      // Map<name, EmpiricalMeasure>
  // Per-binding histogram cache. Histogram computation is O(N) and
  // for N=1M takes a noticeable few ms; caching keeps click-flipping
  // between previously-viewed bindings instant. Invalidated together
  // with measureCache (source change, configUpdate). Key includes
  // the discrete flag so the same name plotted discrete vs. continuous
  // gets distinct cache entries (defensive — discreteness is fixed
  // per binding today but the door's open for future modes).
  ctx.histogramCache = new Map();    // Map<"name|d"|"name|c", histogram>
  // Profile-plot per-axis range cache. Keyed by
  // "binding|sweepKey|presetName" so each (function, axis,
  // preset) combination remembers the user's x-axis edits across
  // navigation. Invalidated alongside measureCache /
  // histogramCache on source / sample-count changes.
  //   Map<key, { lo, hi, fromAuto: boolean }>
  // fromAuto distinguishes ranges initially populated by
  // resolveSweepRange (auto) vs. user-edited (override) — used
  // for tooltip / debug; the renderer treats both the same.
  ctx.profileRangeCache = new Map();
  // Module-wide overrides on named preset (record-point) bindings.
  // Persists across binding navigation, so tuning pars1 on a
  // likelihood plot applies the same overrides when the user visits
  // a forward kernel that shares those kwarg names. Reconciled on
  // every source change via value comparison against the freshly
  // parsed base values (see rebuildDerivations); a kwarg whose
  // source value now matches the override drops from the override
  // automatically.
  //   Map<presetName, { values: { kwargName → number } }>
  ctx.presetOverrides = new Map();
  // Module-wide overrides on named preset-domain bindings (cartprod
  // forms). Same lifetime/reconciliation pattern as presetOverrides:
  // persists across binding navigation, prunes per kwarg against
  // current source values, drops the entry when the source binding
  // is gone.
  //   Map<domainName, { ranges }>
  // ranges: { kwargName → { lo, hi } }   user-set range overrides
  ctx.domainOverrides = new Map();
  ctx.rootSeed = 1;
  // Sample budget for chain-based plots. Higher → smoother histograms,
  // marginal cost grows linearly. Tuned for sub-100ms response.
  // Sample budget per binding when the visualizer renders a histogram.
  // Owned by VS Code's configuration (flatppl.visualization.sampleCount,
  // default 100000, max 10_000_000); the host pushes it via a
  // configUpdate message and updates it on settings changes. Value
  // here is just an in-flight default until the first configUpdate
  // arrives — the panel always boots with a config push from the host.
  ctx.SAMPLE_COUNT = 100000;

  // Per-atom rejection budget for matTruncate's rejection-redraw path
  // (spec §06 truncate). When the parent measure isn't CDF-invertible,
  // the worker redraws from the underlying distribution up to this
  // many times per atom before giving up and emitting NaN. Higher
  // values trade compute for less ESS loss on tightly-truncated
  // measures; lower values keep large-N plots responsive. The host
  // pushes a new value through configUpdate when the user changes
  // the corresponding setting (VS Code: flatppl.truncate.rejectionBudget).
  ctx.REJECTION_BUDGET = 1000;




  /**
   * Recursively materialise the empirical measure for a binding,
   * reusing cache entries for any deps already computed.
   * Returns Promise<EmpiricalMeasure>.
   *
   * Aliases share the SAME EmpiricalMeasure object (same samples
   * array, same logWeights ref) so click-flipping between a variate
   * and its measure is free. With null-uniform logWeights the cache
   * is purely additive over today's behaviour — no extra allocation.
   */

  /** Soft-fail variant of getMeasure: resolves to null instead of
      rejecting when the binding can't produce a measure (no
      derivation, no fixed value — typically pure inputs like
      `elementof(reals)`). Used by plot paths that want to chase
      sample-derived defaults for every source-binding axis but
      shouldn't blow up on the ones that genuinely have no samples
      to chase. */

  // Plot-plan fallbacks below still need the helper as a local
  // function reference; expose the engine copies under their old
  // names so the rest of viewer.js keeps working without further
  // edits.
  // Current plot plan from buildPlotPlan(). Two shapes:
  //   { mode: 'analytical', ir }
  //   { mode: 'chain', chain, discrete }
  // Used both as the "is plot tab enabled?" flag and as the render
  // input. currentPlotBindingName tracks which binding produced it
  // (for the chart title and stale-reply guards).
  ctx.currentPlotPlan = null;
  ctx.currentPlotBindingName = null;



  // Fire-and-forget message send (used during init when we don't care
  // about the reply). Distinct from sendWorker so we don't allocate a
  // pending-request entry for messages whose reply is just an 'ok'.


  /**
   * Build a plot plan for a binding. The orchestrator decides
   * sample-ability and returns a topo-ordered chain. Here we
   * additionally decide whether the analytical PDF/pmf curve should
   * accompany the histogram.
   *
   * Two semantic rules govern the density overlay:
   *   1. A density belongs to a *measure*, not to a variate. A
   *      stochastic binding like 'theta1 = draw(theta1_dist)' is a
   *      single drawn value — its samples have an empirical
   *      distribution, but the density itself is a property of the
   *      law theta1_dist. So variates (binding.type === 'draw')
   *      get samples-only; their underlying measure is where the
   *      density curve lives.
   *   2. A measure binding shows the analytical density only when
   *      the leaf, after alias resolution, has all-literal kwargs.
   *      A measure with stochastic parents has no closed-form
   *      marginal density; we'd need numerical marginalisation,
   *      which is more honest as a histogram.
   *
   * Returns:
   *   { chain, discrete, analyticalIR? }   — plottable
   *   null                                 — not plottable
   */


  // Plot panel visibility — separate from "is the current binding
  // plottable?". When plotEnabled is true, the plot pane occupies
  // the bottom 40% of the panel; when false, it's collapsed and the
  // graph pane takes the full content area. The pane content always
  // reflects the current focused binding, so flipping plotEnabled
  // back on never shows stale data.
  ctx.plotEnabled = false;








  /**
   * Render the Plot panel from a samplesPlot worker reply.
   *
   * The reply has three parts:
   *   reply.samples   — raw Float64Array (kept for future use; not
   *                     directly drawn here, but available if we want
   *                     to add e.g. a sample trace later)
   *   reply.histogram — equal-width bars (FD for continuous, integer
   *                     for discrete), area-normalised so they read
   *                     directly against a PDF/PMF curve
   *   reply.density   — smooth analytical curve (when leaf has all-
   *                     literal kwargs) OR KDE estimate; null for
   *                     discrete-with-no-analytical (the histogram
   *                     itself is already the empirical pmf)
   *
   * Both layers (bars + curve) use the focused binding's ctx.TYPE_STYLE
   * color from the DAG view, so a stochastic 'draw' node plots
   * purple, a measure-alias 'call' node plots grey-blue, etc. Bars
   * sit at low alpha; the line/dots are opaque on top.
   */

  /**
   * Format a fixed scalar for display. JavaScript's default String()
   * gives "5" for 5.0 and full precision for things like 0.1+0.2;
   * we strip trailing zeros via toPrecision(12) → parseFloat → String
   * so 5.0 reads "5", 3.14159 stays "3.14159", and noisy
   * float-arithmetic results like 0.30000000000000004 become "0.3".
   */
  // Compact UI rendering of a numeric value. Truncates to 4
  // significant digits — enough to distinguish typical
  // posterior-style values (e.g. -0.1930 vs 0.2998) without the
  // false-precision look of floats printed at full Float64 width.
  // Used by inline labels (preset dropdowns, x-range inputs),
  // value-as-text displays, and as the echarts axisLabel formatter
  // so chart ticks match the same convention. Integers pass through
  // unchanged (Number.isInteger short-circuit) so axis ticks at
  // whole numbers stay readable as "1", "2", … rather than "1.000".

  // "a + b i" / "a - b i" for a complex scalar constant. Both parts
  // go through formatScalar so precision/integer handling matches
  // the real path. The sign is folded into the connector so we never
  // print "a + -b i"; -0 imaginary reads as "+ 0 i".

  // Toolbar badge for a complex binding rendered as its real part.
  // Static (no interaction) in v1 — the |z| / Im / Argand mode
  // toggle is a tracked follow-up and will replace this with a
  // button group in the same toolbar slot.

  // Compose pre-formatted element strings into "[a, b, c]" or
  // "[a, b, c, …, z] (length N)" for long arrays. The threshold
  // balances readability against verbosity: 8 fits on typical
  // screen widths even with ~5-digit values.

  // Back-compat shim: takes a numeric array, formats each element
  // via formatScalar, then composes with formatArrayParts.

  // Composable value-to-string for plain JS values — numbers,
  // booleans, strings, arrays, plain objects. Mirrors the
  // FlatPPL surface form (record(k = v, …) for objects, [v, …]
  // for arrays, ellipsised when long). The kind of light-weight
  // pretty-printer that Julia's Base.show pairs with each value
  // type. Used for preset value display in the toolbar dropdown
  // and as the leaf-formatter for constant-measure rendering.


  // True iff every scalar leaf of a record/tuple/array measure has
  // identical samples across all N atoms. The deterministic
  // detection drives the constant-as-text rendering for
  // record-shaped bindings whose value is the same at every atom
  // (literal records, deterministic arithmetic over literals, etc.)
  // — same idea as the scalar samplesAreConstant short-circuit, but
  // walks the SoA tree.
  //
  // Special case for literal-array fields: a `kind: 'array'`
  // derivation materialises as { samples: Float64Array(K), ... }
  // where K is the array length (NOT SAMPLE_COUNT). Per-atom these
  // are deterministic — all atoms see the same array — so we treat
  // them as "constant" even though the array's values differ from
  // each other. We detect this via samples.length !== SAMPLE_COUNT;
  // a per-atom scalar measure has length === SAMPLE_COUNT.

  // Render a constant measure as the FlatPPL surface form. Used by
  // the plot-pane dispatch when measureIsConstant returns true:
  // record-shaped bindings show "record(a = …, b = …)" text rather
  // than a corner plot of N copies of the same point. Array leaves
  // ellipsize past length 8 so a 10-observation literal stays
  // readable. Walks the SoA tree top-down — same shape conventions
  // as listScalarAxes.

  /**
   * Resolve a binding's plot color to match the DAG renderer's
   * choice exactly. The DAG picks color from ctx.TYPE_STYLE[node.type]
   * but then overrides it when node.kind says "measure" (lawof
   * blue) or "kernel" (kernelof teal). Without those overrides the
   * plot for a measure-typed binding (theta1_dist, type='call')
   * would draw in grey instead of the blue used in the DAG bubble,
   * breaking the visual link between the two views.
   *
   * Fall back to ctx.TYPE_STYLE[binding.type] when the binding isn't in
   * the current DAG — paths that update the plot independent of
   * the DAG (rare, but possible during config-update reflows).
   */



  /**
   * Render a fixed-length array as an index→value step plot. Used
   * for literal-array bindings (observed_data = [1.2, 3.4, …]),
   * which aren't samples of a distribution. The series is drawn as
   * piecewise-constant horizontal segments — same shape as the
   * legacy "data preview" view that used to swap into the graph
   * pane, now living in the plot pane next to the DAG.
   */
  // Persistent per-binding plot state for record-shaped measures:
  // selected axes (which scalar leaves to plot in correlations
  // mode) and the chosen view mode ('correlations' | 'marginals').
  // Reset when the focused binding changes; survives re-renders
  // triggered by checkbox / toggle clicks.
  //
  // Correlations mode (NxN matrix of marginals + joint scatters)
  // becomes unreadable past 4x4, so we cap selection at 4. Marginals
  // mode (one density-shaded column per axis) scales linearly to
  // the full axis count and ignores the selection.
  ctx.recordSelection = null;
  ctx.CORRELATIONS_MAX_AXES = 4;


  /**
   * Render a record-shaped EmpiricalMeasure as a corner plot,
   * with a checkbox row above for axis selection (max 4 axes).
   *
   * Corner plot:
   *   - diagonal:        1D marginal histogram of each selected axis
   *   - below-diagonal:  2D joint scatter for each (axis_j, axis_i)
   *                      pair with i > j
   *   - above-diagonal:  empty (corner-plot convention)
   *
   * SoA pays off here: marginals are sub.samples; joints are just
   * two columns zipped index-wise — no copy, no projection.
   */
  // Render a constant record/tuple as plain text — same scalar-display
  // styling the constant-scalar branch uses, just with the surface
  // form as the value. We cap font-size when the rendered string is
  // long so the corner-plot 36px doesn't overflow on a multi-field
  // record; the simple len-based cutoff is fine here (the value is
  // either short and reads at 36px or long enough to want 16px).



  /**
   * Compact "N: ...  ESS: ..." readout for the toolbar's right
   * edge. Format:
   *   "<N> samples (<label>: ESS <ratio>%, PSIS k̂ <value>)"
   * where <label> ∈ {good, ok, bad, unusable} colours the
   * parenthesised diagnostic span. Quality is computed by
   * FlatPPLEngine.empirical.importanceSamplingQuality, which
   * combines PSIS k̂ (Vehtari et al.; Pareto-tail shape of the
   * upper importance weights) with Kish ESS, max-weight share,
   * and a sample-size-aware k̂ threshold. See empirical.js for
   * the threshold table; the worst trigger across diagnostics
   * sets the label.
   *
   * Unweighted measures (logWeights == null) always read 'good'
   * with ratio 100% and k̂ shown as "—" (not meaningful for
   * uniform weights).
   */
  /** Format a log-total-mass for the stats readout. The engine carries
      mass on the log scale precisely because deep compositions can
      easily overflow Float64; the display layer formats it back.
      Returns null when the mass is essentially 1 (normalized) — the
      caller skips the badge entirely so the readout only surfaces
      info when there's something to say. */





  // Compact sample-count rendering: powers of 10 collapse to
  // superscript form ("10⁵" instead of "100,000") to save toolbar
  // width — typical default sample sizes (10⁴, 10⁵, 10⁶) all win.
  // Anything else falls back to the comma-grouped count. Only
  // exact powers ≥ 10² qualify; "10" itself stays "10" and small
  // counts read better verbatim.





  // ---- Fixed-value plot --------------------------------------------
  //
  // Phase-driven dispatch for compile-time-determinate bindings.
  // Records / tuples render the FlatPPL surface form as text;
  // scalars render the value as a single number when the per-atom
  // samples are constant, otherwise fall through to the histogram
  // path (engine-broadcast cases like lp_obs).

  // No separate renderFixedScalar — the existing
  // renderSamplesAndDensity already short-circuits to scalar-text
  // when samplesAreConstant. We keep mode='fixed-scalar' as the
  // plan label (so the source intent is visible in plan dumps /
  // logs) but route it through the same sample pipeline.

  // ---- Kernel sample plot ------------------------------------------
  //
  // Kernels (kernelof / functionof returning a measure) are plotted
  // by picking concrete values for their inputs (a preset, or
  // type-aware / source-empirical defaults), substituting those
  // into the kernel body, and sampling N atoms from the resulting
  // self-contained measure. The samples render via the existing
  // histogram / corner-plot pipeline in renderSamplesAndDensity.
  //
  // Cache: kernel-sample measures are stored in measureCache under
  // a synthetic key "<kernelName>|kernel-sample|<presetName>" so
  // switching presets doesn't re-sample, and switching back to a
  // previously-rendered kernel is instant.

  /** The override entry for the active selection — auto's lives
      on the plan (per-binding), named presets live in the
      module-wide ctx.presetOverrides map. Returns null if none. */

  /** Whether the active preset selection currently has any value
      overrides. Drives the "(modified)" tag and the reset/persist
      button visibility on the Inputs control. Axis-range overrides
      moved to the Domain control's hasDomainOverrides(ctx, plan). */

  /** Write back an override entry for the active selection.
      Routes auto entries to the plan (per-binding), named
      entries to the module-wide store. Pass null to clear. */

  /** Get-or-create a fresh override entry for the active
      selection (caller will mutate it and call setOverrideFor
      to commit). */

  /** Effective {values} for a plan, merging base preset values
      with any override on top. Base values for named presets come
      from matchedPresets[i].values; for auto, base is an empty
      object (the dropdown "auto: …" label uses computeAutoValues
      separately, but env-substitution falls through to type
      defaults + source-sample materialisation when no explicit
      value is present). */

  /** Source-declared base values for the active preset (no
      overrides applied). For named presets this is
      matchedPresets[i].values; for auto, an empty object. */

  // ===================================================================
  // Domain (cartprod) override plumbing — mirrors the preset path
  // above, but stores per-kwarg [lo, hi] ranges instead of per-kwarg
  // values. Domains drive the x-axis range; presets drive the
  // non-swept input values.
  // ===================================================================

  /** Override entry for the active domain, or null when none. Auto
      domain's entry lives on plan.domainAutoOverride; named ones in
      ctx.domainOverrides keyed by name. */

  /** Get-or-create a domain override entry for the active selection.
      Caller mutates entry.ranges and commits via setDomainOverrideFor. */

  /** Commit (or clear, with null) a domain override entry. */

  /** True when the active domain has at least one ranged kwarg in
      its override entry. Used to gate visibility of the reset /
      save buttons. */

  /** Source-declared base ranges for the active domain (no overrides
      applied). For named domains this is matchedDomains[i].ranges;
      for auto, an empty object (the auto domain has no source-side
      ranges — the per-axis auto-fit code computes them on demand). */

  /** Effective ranges for the active domain — base merged with any
      override entry's ranges on top. Returns { kwarg: {lo, hi} }. */

  /** Held-constant kwargs for the active preset. Drawn from the
      engine's findMatchingPresets `fixedNames` (kwargs whose source
      value was wrapped in `fixed(...)` — spec §03's "hold constant
      during optimization" hint). For the auto preset no fixed-hint
      exists, so the set is always empty. Returns a Set so callers
      can do .has(name) directly. */

  // Shared icon-button helper used by the Inputs and Domain
  // toolbars' reset / save / save-as buttons. iconKey picks a
  // codicon (see CODICON_PATHS); title is the hover tooltip and the
  // accessible name. Buttons are icon-only — the toolbar already
  // shows the action verb implicitly via context, and dropping the
  // text labels frees the horizontal space the dropdown needs to
  // breathe.

  // Build a "Inputs: [auto / pars1 / …]" control fragment for the
  // profile / kernel-sample plot toolbar.
  //
  // We use a custom button-plus-popup instead of <select> so the
  // collapsed control can show just the short label
  // ("auto (modified)") while the open dropdown shows the longer
  // "name: theta1 = X, theta2 = Y" form. <select> doesn't support
  // different text in collapsed vs. open states across browsers
  // (the `label` attribute is spec but Chromium ignores it).

  // ----- Domain control: parallel of buildPresetControl, but for
  // cartprod(...) preset domains. Drives x-axis range per kwarg
  // rather than non-swept input values.

  /** Compose a human-readable summary of a domain's effective
      bounds, one entry per kwarg in `kwargOrder`. Reads bounded
      kwargs from `ranges` (lo/hi pairs from interval(...) fields
      or user overrides) and unbounded kwargs from `setNames`
      (bare `reals` / `posreals` / … fields). */


  /** Persist is supported when there's an override AND the ctx.host
      adapter can write (ctx.host.editSource defined and ctx.host.canPersist
      returns true). For named presets the source RHS also has to
      be literal-friendly; for auto we additionally need
      ctx.host.promptForName for the new-binding name. Hidden
      otherwise so the user never sees a disabled-looking button. */

  /** Format a JS number for source emission. We use String(v)
      rather than formatScalar because formatScalar rounds to 4
      significant figures for display; source needs full
      precision. */

  /** Build the replacement source text for a named preset-point
      record binding, merging the current source RHS kwargs with the
      active override values. Preserves source kwarg order and
      re-wraps overridden values in `fixed(...)` when the original
      source did so — the spec's "held constant" hint must survive
      the round-trip, otherwise persisting a tweak to an
      optimization starting point would silently strip the
      hold-constant annotation. */

  /** Invoke ctx.host.persistPreset for the active selection. Routes
      to "replace existing binding" or "append new binding"
      depending on whether the active selection is a named
      preset or auto. Host applies the edit; the next source-
      update cycle reconciles the override away because the
      source values now match. */


  /** Auto persist: ask the ctx.host to prompt for a binding name,
      then ask it to append the new preset binding at end-of-
      source. The two-step contract keeps line/text construction
      and queuing the next-active-preset hint in the viewer
      (single source of truth); each ctx.host implements only the
      primitives (UI prompt + edit application). */

  // ===================================================================
  // Domain persist — parallel of canPersistActive / persistActive
  // ===================================================================

  // KNOWN_NAMED_SETS moved into util.js alongside its sole consumer
  // isPersistableSetField — the extraction missed
  // the cross-module reference; nothing exercised it until the
  // persist-domain path would have been triggered.

  /** Whether an AST node is a recognized cartprod field value:
      either `interval(NumberLiteral, NumberLiteral)` or a bare
      named-set reference. Used by canPersistDomain to gate the
      save button. */

  /** Serialize a recognized cartprod field value back to source
      text. Mirrors isPersistableSetField — caller has already
      gated. */

  /** Pick a "natural" set-source-text for one of a plan's input
      kwargs — used when the user persists a partial domain and we
      want to fill the unset kwargs with something matchable rather
      than dropping them (which would make the resulting cartprod
      fail findMatchingDomains' shape check). Strategy: ask the
      engine for the axis's base set descriptor; map known kinds to
      their source names; fall back to 'reals' for empirical /
      unresolved descriptors.

      For multi-axis kwargs (vector / record-typed inputs) the
      simple per-axis mapping is wrong (we'd need cartpow /
      cartprod), so we surface 'reals' there too — the user can
      edit the source by hand if they want a tighter set. */

  /** Persist is supported when there's an override AND the ctx.host
      adapter can write. For named domains every source field has
      to be a recognized set form (interval-with-literal-bounds OR
      a named set like `reals` / `posreals` / …). For auto we
      additionally need ctx.host.promptForName for the new-binding name. */


  /** Build the replacement source text for a named cartprod domain,
      merging source-declared field values with overridden ranges.
      For each kwarg:
        - if the override has a range → emit interval(lo, hi)
        - else → preserve the source field as-is (interval(...)
          with original bounds, or the bare named-set reference)
      Preserves source kwarg order. */


  /** Append a fresh cartprod(...) binding capturing the current
      domain override. Asks the ctx.host for a name via promptForName.
      Fills *every* input kwarg in the signature: overridden ones
      get `interval(lo, hi)`, the rest get their natural base set
      (`reals` / `posreals` / …) via defaultSetSourceForKwarg.
      Filling in the unset kwargs keeps the new cartprod matchable
      in findMatchingDomains' shape check — otherwise a partial
      domain like `cartprod(theta1 = interval(-4, 4))` wouldn't
      appear in the Domain dropdown after persist. */

  // Strip the outer "record(...)" wrapper from formatValue's
  // output so the dropdown reads cleanly:
  //   record(theta1 = 1.4, theta2 = 1.0)  →  theta1 = 1.4, theta2 = 1.0

  // Synthesise the auto-mode fixed-input values for a profile /
  // kernel-sample plan, matching the renderer's fallback
  // behaviour: source-binding axes use the cached samples[0] (or
  // type default if not yet cached); placeholder/other axes use
  // the type default. Returned as { kwargName: value }.

  // Render a kernel-sampled empirical measure. Record / tuple /
  // array measures route through renderRecordMarginals with the
  // preset dropdown injected into its toolbar (no extra row).
  // Scalar sampled measures use a simple histogram via
  // renderSamplesAndDensity; constant scalars / records get the
  // text-render path. This avoids wrapping the existing renderers
  // in a flex-column container that compressed the corner-plot
  // cells under a layout race.
  // Kernel sampling produces an empirical measure exactly like any
  // other measure binding — semantically a kernel IS a nullary
  // kernel once its inputs are bound. So the entire shape dispatch
  // (constant text / record-marginals / scalar histogram) is shared
  // with the standard measure path via renderEmpiricalMeasure; the
  // only kernel-specific bit is the preset selector in the toolbar.
  // Discreteness defaults to false: kernel bodies don't go through
  // orchestrator typing, so we don't have a discrete flag to plumb.
  // (No analytical density either — kernel bodies are empirical.)
  //
  // toolbarControls is passed as a *builder thunk*, not a static
  // Element / DocumentFragment. The corner-plot rerender path
  // (mode toggle, axis selection) blows away and rebuilds the
  // toolbar; appendChild on a DocumentFragment moves its children
  // out and leaves it empty, so a static fragment would render
  // once and disappear on every subsequent rebuild. The thunk
  // produces fresh DOM each call.

  // Recursively materialise a self-contained measure IR (no
  // measure-position self-refs; %local refs already substituted to
  // literals) into an EmpiricalMeasure. Used by the kernel-sample
  // path to draw N atoms from a kernel body at fixed parameter
  // values.
  //
  // Cases:
  //   leaf distribution (Normal, Exp, …)  → worker.sampleN with
  //                                         refArrays for captured
  //                                         self-refs (per-atom
  //                                         semantics matching the
  //                                         closed-measure getMeasure
  //                                         path)
  //   joint(field=M, …) / record(…)       → recordMeasure(materialise(field), …)
  //   iid(M, dim, …)                      → arrayMeasure(materialise(M, count×∏dims))
  //   lawof(M)                            → recurse into M (lawof is a no-op on measures)
  //
  // weighted / normalize / superpose / kernels-applied-to-iid
  // surface as a clear error rather than a silent broken plot.

  // ---- Profile plot ------------------------------------------------
  //
  // Type-aware default value for an axis leafType. Used to populate
  // fixedEnv for non-swept inputs at first plot. Posreals defaults
  // to 1.0 (avoids degenerate cases like sigma=0); intervals
  // default to the midpoint; integers default to 0; etc. F4b will
  // let the user override these via the fixed-values panel.

  // Default sweep range for an axis from leaf-type alone. Used as
  // the final fallback after the axis-set descriptor and empirical
  // backref both fail to give a range.

  // Map a structural set descriptor (from
  // orchestrator.resolveAxisBaseSet) to a concrete sweep range.
  // Empirical sets defer to the caller (the viewer materialises
  // the source binding and computes a 4-σ quantile range).
  //   reals          → [-5, 5]            (Gaussian-like default)
  //   posreals       → [eps, 5]           (avoid 0 boundary for log etc.)
  //   nonnegreals    → [0, 5]
  //   integers       → [-10, 10]
  //   posintegers    → [1, 20]
  //   nonnegintegers → [0, 20]
  //   booleans       → [0, 1]
  //   interval(a, b) → [a, b]

  // Resolve the auto-range for a swept axis. Three-tier fallback:
  //   1. Set descriptor (interval / reals / posreals / …) from
  //      resolveAxisBaseSet — covers identifier-bound elementof
  //      bindings.
  //   2. Empirical 4-σ quantile from the source binding's samples
  //      — covers identifier boundaries pointing at stochastic /
  //      derived bindings.
  //   3. Leaf-type default — placeholders, anything unresolved.
  // Returns a Promise<[lo, hi]> since step 2 may need to await
  // getMeasure(...).

  // Render the profile plot for a callable binding. Builds env with
  // default values for non-swept inputs, picks a default range for
  // the swept axis, fires worker.profileN, then draws a line plot.
  //
  // Limitations (F4a):
  //   - Top-level scalar inputs only — record / array inputs
  //     classify a path on each axis, but populating a fixedEnv with
  //     a record literal is F4b work.
  //   - Plain kernelof bindings (not wrapped in likelihoodof) need
  //     an obs value the user has to provide; defer to F4b.




  /** Commit a clicked x value as the sweep-axis value of the
      active preset. Writes through the unified override store
      (autoOverride for auto, ctx.presetOverrides for named). */




  // Set by persistAutoAsNewBinding / persistAutoDomainAsNewBinding
  // — the freshly-coined preset / domain name the rebuilt plan
  // should land on as its initial selection. Consumed once by the
  // next updatePlotForBinding call (then cleared). Two separate
  // slots so a domain save-as can't accidentally knock the user
  // off a selected preset and vice-versa.
  ctx.pendingPresetName = null;
  ctx.pendingDomainName = null;

  // Per-binding memory of the user's plan-level selections (sweep
  // axis, output leaf, preset / domain name, auto-overrides for
  // both inputs and domain). Keyed by binding name. Repopulated
  // after every plan build, consulted at the start of the next
  // build so navigating away and back restores the prior view.
  // No clear-on-source-change: stale entries are filtered by the
  // matchedPresets / matchedDomains / axes existence checks in
  // applyRememberedSelections, so a renamed-then-restored binding
  // re-applies its memory rather than discarding it.
  ctx.planMemoryByName = new Map();

  // Drop override values for kwargs the rebuilt plan no longer has
  // (e.g. the source was edited so an input went away). Without
  // this, navigating back later would re-apply a value to a kwarg
  // that no longer exists, leaking stale state into the override
  // map.



  // Call after every focusNode() to update the Plot tab's enabled
  // state and (if visible) re-render its content.

  // Plot toggle click handler. Restores from VS Code webview state on
  // first paint (see initial setPlotEnabled call below) so the user's
  // preference survives reloads.
  $('plot-toggle').addEventListener('click', function() {
    setPlotEnabled(ctx, !ctx.plotEnabled);
  });

  // Drag handle between the DAG and plot panes. Lets the user
  // redistribute vertical space; both panes have a min-height clamp
  // so neither can be dragged into invisibility. The DAG and plot
  // ResizeObservers (set up further below) pick up the resulting
  // size change and refit cytoscape / echarts automatically — no
  // explicit resize / fit calls needed here.
  $('plot-divider').addEventListener('mousedown', function (ev) {
    if (!ctx.plotEnabled) return;
    ev.preventDefault();
    const graph = $('graph-panel');
    const plot  = $('plot-panel');
    const startY = ev.clientY;
    const startGraphPx = graph.getBoundingClientRect().height;
    const startPlotPx  = plot.getBoundingClientRect().height;
    const combinedPx = startGraphPx + startPlotPx;
    const MIN_PX = 80;
    function onMove(mv: any) {
      const dy = mv.clientY - startY;
      let newGraph = startGraphPx + dy;
      let newPlot  = startPlotPx  - dy;
      if (newGraph < MIN_PX) { newGraph = MIN_PX; newPlot = combinedPx - MIN_PX; }
      if (newPlot  < MIN_PX) { newPlot  = MIN_PX; newGraph = combinedPx - MIN_PX; }
      // Use flex-basis in px so the two panes' relative split is
      // exactly what the user dragged to. flex-grow stays 1 on
      // both so subsequent host-pane resizes redistribute the
      // delta proportionally rather than parking it on one side.
      graph.style.flex = '1 1 ' + newGraph + 'px';
      plot.style.flex  = '1 1 ' + newPlot  + 'px';
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  // --- DAG rendering ---

  // Tear down all bubble paths and clear leftover scratch. Two bubblesets-js
  // bugs we work around here:
  //   1) path.remove() sets scratch.bubbleSets to {} on each element. The
  //      next addPath's update sees the truthy empty object and crashes on
  //      linesEquals(undefined, lines). Fix: removeScratch fully.
  //   2) path.remove() detaches listeners but does NOT cancel callbacks
  //      already queued in the throttle. Those queued callbacks fire after
  //      tear-down and call this.update() on the dead path. Fix: stomp
  //      path.update to a no-op before removing.

  // Member-id set for one reification's bubble: its own kernel PLUS the
  // full kernel of any nested reification whose name appears in this
  // kernel. Nested-reification synthetic nodes need positive potential —
  // not just "avoid exemption" — for the outer contour to wrap around
  // them rather than pinching past.




  // ---------------------------------------------------------------
  // Local model state
  //
  // The webview parses the .flatppl source itself (via the bundled
  // FlatPPLEngine) instead of receiving pre-rendered DAG data from the
  // extension host. This keeps the visualizer self-contained and lets
  // the same code run in a future standalone web preview.
  //
  // Two caches:
  //   currentSource  — last parsed source text (string)
  //   currentBindings — engine.processSource(currentSource).bindings
  // We re-parse only when source changes; clicking through nodes (zoom-
  // into) reuses currentBindings and just recomputes the sub-DAG.
  // ---------------------------------------------------------------
  ctx.currentSource = null;
  // Active surface-syntax variant id for the in-memory source —
  // drives both processSource grammar selection and persist write-
  // back syntax. Updated whenever a sourceUpdate carries a
  // variant; defaults to 'flatppl'.
  ctx.currentVariantId = 'flatppl';
  ctx.currentBindings = null;
  // The lowered module forwarded by processSource — used by
  // typeinfer.inferExprInScope for on-demand call-site
  // specialization (multi-output Output: selector, etc.).
  ctx.currentLoweredModule = null;



  // Back button: pop the previous view; bindings are unchanged, only
  // re-render with the saved sub-DAG data. (We push state objects that
  // hold both the data and the target name, so we don't have to recompute
  // when going back.)
  $('back-btn').addEventListener('click', function() {
    if (ctx.history.length === 0) return;
    ctx.currentState = ctx.history.pop();
    renderDAG(ctx, ctx.currentState.data);
    updateBackBtn(ctx);
    // Module view has no per-binding plot target and no per-binding
    // title — call updatePlotForBinding(null) so the plot pane shows
    // its module-mode placeholder, and tell the host to set a
    // generic title rather than the sentinel string.
    if (ctx.currentState.targetName === ctx.MODULE_TARGET) {
      updatePlotForBinding(ctx, null);
      if (ctx.host.setTitle) ctx.host.setTitle('module');
    } else {
      updatePlotForBinding(ctx, ctx.currentState.targetName);
      if (ctx.host.setTitle) ctx.host.setTitle(ctx.currentState.targetName);
    }
  });

  // Source-update handler shared by:
  //   - the postMessage listener below (VS Code extension host pushes
  //     fresh source on cursor moves and edits)
  //   - the public view.update(source, target?) method (programmatic
  //     re-render from any host)
  //   - the initial-source bootstrap (opts.source / opts.target on
  //     mount)

  window.addEventListener('message', function(event) {
    const msg = event.data;
    if (!msg) return;

    if (msg.type === 'configUpdate') {
      // The host pushed updated visualization settings.
      const cfg = msg.config || {};

      // sampleCount: drop every cached EmpiricalMeasure on change
      // (each was sized to the old SAMPLE_COUNT and can't be reused)
      // and re-render the current plot at the new count. The
      // histogram cache must go too — it's keyed by binding name
      // but the underlying samples will be different.
      if (typeof cfg.sampleCount === 'number'
          && cfg.sampleCount > 0
          && cfg.sampleCount !== ctx.SAMPLE_COUNT) {
        ctx.SAMPLE_COUNT = cfg.sampleCount | 0;
        ctx.measureCache = new Map();
        ctx.histogramCache = new Map();
        if (ctx.plotEnabled) renderPlotForCurrent(ctx);
      }

      // dagNavigationHistoryCap: re-bind the limit and trim oldest
      // entries that exceed the new cap. Doesn't affect currentState
      // or the back button beyond the trim.
      if (typeof cfg.dagNavigationHistoryCap === 'number'
          && cfg.dagNavigationHistoryCap >= 0) {
        ctx.HISTORY_CAP = cfg.dagNavigationHistoryCap | 0;
        while (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
        updateBackBtn(ctx);
      }

      // truncateRejectionBudget: re-bind. Drop the cache because any
      // cached truncate(...)-derived measure was sized at the prior
      // budget — its n_eff and NaN slots would no longer reflect the
      // current setting. Re-render to recompute against the new value.
      if (typeof cfg.truncateRejectionBudget === 'number'
          && cfg.truncateRejectionBudget >= 1
          && cfg.truncateRejectionBudget !== ctx.REJECTION_BUDGET) {
        ctx.REJECTION_BUDGET = cfg.truncateRejectionBudget | 0;
        ctx.measureCache = new Map();
        ctx.histogramCache = new Map();
        if (ctx.plotEnabled) renderPlotForCurrent(ctx);
      }
      return;
    }

    if (msg.type !== 'sourceUpdate' && msg.type !== 'showModule') return;
    applySourceUpdate(ctx, msg);
  });

  initCy(ctx);

  // Tell the host the message listener is attached. VS Code's
  // webview.postMessage doesn't reliably buffer pre-load — messages
  // sent before this point can be lost, which produced the
  // "empty panel on first Visualize" issue. The host buffers
  // sourceUpdate / showModule / configUpdate until it sees this
  // 'webviewReady' and then flushes in order.
  if (ctx.host && ctx.host.signalReady) {
    try { ctx.host.signalReady(); } catch (_) {}
  }

  // Resize every echart instance inside #plot-content whenever the
  // plot pane changes size. Multi-chart layouts (corner plot,
  // density strips) hold many echart instances we don't track in
  // a single global; a ResizeObserver on plot-content lets us
  // resize them uniformly without needing each renderer to wire
  // up its own listener. Falls back to window.resize where
  // ResizeObserver isn't available (older webview hosts).
  if (typeof ResizeObserver === 'function') {
    // Wrap to bind ctx — the
    // ResizeObserver / window resize callbacks invoke their handlers
    // with no args, so the bare function reference would receive
    // `undefined` as ctx and throw.
    const plotResizeObserver = new ResizeObserver(function () { resizeAllEchartsInPlot(ctx); });
    const plotRoot = document.getElementById('plot-content');
    if (plotRoot) plotResizeObserver.observe(plotRoot);
  } else {
    window.addEventListener('resize', function () { resizeAllEchartsInPlot(ctx); });
  }

  // Resize the cytoscape DAG when its container changes size. Without
  // this, hosts that mount the viewer inside a flex/grid layout that
  // hasn't fully settled at mount time end up with a DAG sized to
  // whatever the container was at first paint — typically too small,
  // with the layout fit-zoomed and panned for the wrong dimensions, so
  // the visible nodes appear off-center against the post-settle pane.
  // The VS Code webview avoids this because the panel resizes through
  // window.resize, which cytoscape already handles internally; the
  // standalone web host (CSS Grid + flex) needs the explicit observer.
  if (typeof ResizeObserver === 'function') {
    const cyResizeObserver = new ResizeObserver(function () { resizeAndFitCy(ctx); });
    const cyRoot = document.getElementById('cy');
    if (cyRoot) cyResizeObserver.observe(cyRoot);
  } else {
    window.addEventListener('resize', function () { resizeAndFitCy(ctx); });
  }

  // Restore Plot toggle state from the host's persistent state so the
  // user's preference survives panel close/reopen and reloads. Default
  // is OFF for first-time use — the plot panel is opt-in to keep the
  // initial DAG-only experience clean.
  let prevState: any = null;
  if (ctx.host.loadState) { try { prevState = ctx.host.loadState(); } catch (_) {} }
  setPlotEnabled(ctx, prevState && prevState.plotEnabled === true);

  // Initial source bootstrap. When opts.source is supplied, render
  // immediately. Otherwise the viewer waits for a postMessage
  // sourceUpdate (the existing VS Code flow) — the message listener
  // above feeds applySourceUpdate when the host sends one.
  if (typeof opts.source === 'string') {
    applySourceUpdate(ctx, {
      source: opts.source,
      targetName: opts.target,
      type: opts.target ? 'sourceUpdate' : 'showModule',
      pushHistory: false,
      variant: opts.variant,
    });
  }

  // Public control surface. update(source, target, opts?) re-parses
  // and re-renders. opts.pushHistory: when true (default false),
  // treat the update as a user-initiated navigation and grow the
  // viewer's internal back-button stack (matching how DAG dbltap
  // pushes). Hosts that route through the browser's URL history
  // (e.g. the gallery's hash-based router) set this on user-driven
  // navigations so the in-viewer back button stays usable for
  // target-only steps within one model.
  // dispose() is a placeholder for now.
  return {
    update: function(source: any, target: any, opts: any) {
      applySourceUpdate(ctx, {
        source: source,
        targetName: target,
        type: target ? 'sourceUpdate' : 'showModule',
        pushHistory: !!(opts && opts.pushHistory),
        variant: opts && opts.variant,
      });
    },
    dispose: function() {},
  };
  }
