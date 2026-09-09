// @ts-check
// @flatppl/viewer — DOM templates + CSS injection —
//
// VIEWER_CSS is injected once into <head>; VIEWER_BODY_HTML is the
// markup mount() drops into its container. ensureCssInjected is
// idempotent — the cssInjected flag is module-level state so a
// second mount() on the same page doesn't re-inject.
export var VIEWER_CSS = `
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--vscode-editor-background);
  color: var(--vscode-editor-foreground);
  font-family: var(--vscode-font-family, sans-serif);
  font-size: var(--vscode-font-size, 13px);
  overflow: hidden;
}
#header {
  padding: 5px 14px;
  border-bottom: 1px solid var(--vscode-panel-border, #444);
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: var(--vscode-editor-font-size, 14px);
  white-space: nowrap;
  overflow: visible;
  opacity: 0.8;
  min-height: 26px;
  display: flex;
  align-items: center;
  gap: 0.4em;
  /* Own stacking context above the graph canvas (#main paints after #header
     in DOM order); without this a cytoscape layer can sit over the header and
     swallow clicks on the sampler controls. */
  position: relative;
  z-index: 10;
}
/* The expression is the element that shrinks/ellipsizes — NOT the controls.
   Without this a long model expression pushed the right-aligned sampler
   selector off-screen (header overflow), making it unclickable. */
#header-expr {
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
#inference-controls { flex: 0 0 auto; }
#header .target-name { font-weight: 600; }
#header .target-eq { opacity: 0.5; margin: 0 4px; }
/* Independent Graph / Plots / Math toggles, hosted here or in the web
   header. The visible panels share the content area by the weights in
   panels.ts (graph 3 : plots 2 : math 2 — two panels keep the historical
   60/40 split; a lone panel fills the area). Graph defaults on, plots
   and math off (hosts pick their own first-use default). A panel renders
   whenever enabled — even for a non-plottable binding the plot pane shows
   a "Not plottable" message, so users navigating the graph see a stable
   layout instead of panels appearing/disappearing.

   Heights subtract header(~32px) + info(60px). */
#view-controls { display: flex; gap: 6px; margin-left: auto; }
#graph-toggle, #plot-toggle, #math-toggle {
  background: var(--vscode-button-secondaryBackground, #3a3d41);
  color: var(--vscode-button-secondaryForeground, #ccc);
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 3px;
  padding: 2px 10px;
  font-size: 12px;
  cursor: pointer;
  font-family: var(--vscode-font-family, sans-serif);
  flex-shrink: 0;
}
#graph-toggle:hover, #plot-toggle:hover, #math-toggle:hover { background: var(--vscode-button-secondaryHoverBackground, #505355); }
#graph-toggle.on, #plot-toggle.on, #math-toggle.on {
  background: var(--vscode-button-background, #0e639c);
  color: var(--vscode-button-foreground, #fff);
  border-color: var(--vscode-button-border, transparent);
}
#graph-toggle.on:hover, #plot-toggle.on:hover, #math-toggle.on:hover { background: var(--vscode-button-hoverBackground, #1177bb); }
#main {
  display: flex; flex-direction: column;
  width: 100vw; height: calc(100vh - 86px);
  overflow: hidden;
}
/* Each panel's flex share is set INLINE by applyPanelLayout from
   panels.ts (the one layout rule); the rules here carry only what is
   not a share: the minimum height, the top border between panels, and
   the hidden / full states. */
.viewer-panel {
  flex: 1 1 0; min-height: 80px; position: relative; overflow: hidden;
  border-top: 1px solid var(--vscode-panel-border, #444);
}
/* Floor for the skeleton before the first applyPanelLayout; the inline
   share written by the layout rule overrides it. */
.viewer-panel.full { flex: 1 1 100%; }
/* Hidden panels. Spelled with the ids so the rule outranks the per-panel
   id rules below (#plot-panel sets display: flex; an id beats any number
   of classes) — a bare .viewer-panel.hidden left the plot panel and its
   toolbar on screen at min-height while "off". */
#graph-panel.hidden, #plot-panel.hidden, #math-panel.hidden { display: none; }
/* The first visible panel sits directly under the header: no border. */
.viewer-panel.first { border-top: none; }
#plot-panel {
  display: flex; align-items: center; justify-content: center;
}
#panels-hidden { margin: auto; padding: 1em; opacity: 0.65; text-align: center; }
#panels-hidden[hidden] { display: none; }
#plot-content { width: 100%; height: 100%; }
/* Math pane: a scrollable column of per-binding rows (render-math.ts). */
#math-content {
  width: 100%; height: 100%;
  overflow: auto;
}
/* The pane's placeholder messages ("not available", "rendering…"),
   styled like the plot pane's hints. */
#math-content .math-empty {
  padding: 1.6em; text-align: center;
  font-size: 1.08em; line-height: 1.5;
  font-style: italic; opacity: 0.5;
}
/* Rows: one per binding, the equation left-aligned (MathML Core lays
   it out; Temml's stylesheet, loaded by both hosts, supplies the math
   font chain), an optional one-line doc-comment above it, the Rust
   side's short annotation to its right, per-row diagnostics below.
   The focused row gets the phase-neutral selection tint. Identifiers
   are links back to their binding (render-math.ts). */
#math-content .math-notice,
#math-content .math-module-diags {
  margin: 0.6em 1em 0.2em; padding: 0.4em 0.8em;
  font-size: 0.92em; opacity: 0.85;
  border-left: 3px solid #FFB300;
  list-style: none;
}
#math-content .math-module-diags li { margin: 0.15em 0; }
/* The module introduction (the flatppl_compat doc-comment): prose
   above the rows, its first heading the model's title. */
#math-content .math-module-doc {
  padding: 0.6em 1em 0.5em;
  border-bottom: 1px solid var(--vscode-panel-border, #444);
  font-size: 0.95em; line-height: 1.45;
}
#math-content .math-module-doc h1, #math-content .math-module-doc h2,
#math-content .math-module-doc h3 { font-size: 1.1em; margin: 0 0 0.3em; }
#math-content .math-module-doc p { margin: 0.3em 0; }
#math-content .math-module-doc p:last-child { margin-bottom: 0; }
#math-content .math-empty .math-retry { color: var(--vscode-textLink-foreground, #3794ff); font-style: normal; }
#math-content .math-row {
  padding: 0.45em 1em;
  border-left: 3px solid transparent;
  cursor: pointer;
}
#math-content .math-row:hover { background: rgba(127, 127, 127, 0.08); }
#math-content .math-row.focused {
  background: var(--vscode-list-inactiveSelectionBackground, rgba(14, 99, 156, 0.18));
  border-left-color: var(--vscode-button-background, #0e639c);
}
#math-content .math-row-doc {
  font-size: 0.9em; opacity: 0.7; margin-bottom: 0.15em;
}
#math-content .math-row-doc p { margin: 0; }
#math-content .math-row-eq {
  display: flex; align-items: baseline; gap: 1em;
  font-size: 1.15em;
}
#math-content .math-row-eq math[display="block"] {
  display: inline-block; margin: 0; text-align: left;
}
#math-content .math-row-annotation {
  margin-left: auto; font-size: 0.78em; opacity: 0.6;
  font-family: var(--vscode-font-family, sans-serif);
  white-space: nowrap;
}
#math-content [data-flatppl-ref] { cursor: pointer; border-radius: 2px; }
#math-content [data-flatppl-ref]:hover {
  background: var(--vscode-editor-selectionBackground, rgba(14, 99, 156, 0.35));
}
#math-content .math-row-diags {
  margin: 0.2em 0 0 1.2em; font-size: 0.88em; color: #E57373;
}
/* Drag handles between adjacent visible panels (one after the graph
   panel, one after the plot panel — see panels.ts for the pairing rule).
   Hidden when their panel or every later panel is hidden; the border-top
   on the following panel doubles as the handle's visible band while in
   resting state, so the divider only adds the interactive hover
   affordance. */
.viewer-divider {
  flex: 0 0 5px;
  cursor: row-resize;
  user-select: none;
  position: relative;
  background: transparent;
}
.viewer-divider::before {
  content: '';
  position: absolute;
  left: 0; right: 0; top: 2px; bottom: 2px;
  background: transparent;
  transition: background 0.15s ease;
}
.viewer-divider:hover::before {
  background: var(--vscode-button-background, #0e639c);
}
.viewer-divider.hidden { display: none; }
/* Plot pane layout, controls, and chart ctx.host are styled inline by
   renderPlotFrame — no CSS rules needed here for the per-renderer
   layout. The constant-value / message blocks below still rely on
   global rules. */
#plot-empty {
  opacity: 0.7; padding: 1.6em; text-align: center;
  font-size: 1.08em; line-height: 1.5;
  max-width: 45em; margin: 0 auto;
}
/* Italics for the placeholder hints ("Click a binding…", "Not
   plottable…") but NOT for type-error messages — those need to
   read clearly. */
#plot-empty.hint { font-style: italic; opacity: 0.5; }
#plot-empty ul { text-align: left; display: inline-block; }
/* Stop button shown alongside "Sampling…" while a request is in
   flight. Clicking it terminates the worker (which aborts any
   running tight loop) and rejects in-flight promises; the cache
   on the main thread is preserved, so any binding that finished
   before the cancel stays available. */
#plot-content .plot-stop-btn {
  margin-top: 14px;
  background: var(--vscode-button-secondaryBackground, #3a3d41);
  color: var(--vscode-button-secondaryForeground, #ccc);
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 3px;
  padding: 4px 14px;
  font-size: 12px;
  cursor: pointer;
  font-style: normal; opacity: 0.9;
  font-family: var(--vscode-font-family, sans-serif);
}
#plot-content .plot-stop-btn:hover {
  background: var(--vscode-button-secondaryHoverBackground, #505355);
  opacity: 1;
}
/* Determinate progress bar for off-thread samplers (MH / emcee / AMIS),
   updated by updatePlotProgress as the worker streams progress. */
#plot-content .plot-progress {
  margin: 14px auto 4px;
  width: 220px; height: 6px;
  background: var(--vscode-progressBar-background, rgba(255,255,255,0.12));
  border-radius: 3px; overflow: hidden;
}
#plot-content .plot-progress-fill {
  width: 0%; height: 100%;
  background: var(--vscode-progressBar-foreground, #0e70c0);
  transition: width 0.15s linear;
}
#plot-content .plot-progress-label {
  font-size: 11px; opacity: 0.6; font-style: normal;
  font-family: var(--vscode-font-family, sans-serif);
}
/* Constant-value display: shown when every sample is the same
   value (literal binding, deterministic arithmetic of literals,
   or a degenerate distribution). A histogram of identical values
   is uninformative, so we render the value as readable text. */
#plot-content .scalar-display {
  width: 100%; height: 100%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 6px;
  font-family: var(--vscode-editor-font-family, monospace);
}
#plot-content .scalar-display .name {
  font-size: 13px; opacity: 0.6;
}
#plot-content .scalar-display .value {
  font-size: 36px; font-weight: 300;
  max-width: 100%; box-sizing: border-box; padding: 0 16px;
  overflow-wrap: anywhere; text-align: center;
}
/* Composite values (records, arrays, Dirac wrappers around
   non-trivial bodies, …) drop to a comfortable monospace size
   so long surface forms don't overflow the pane. The class is
   applied by renderTextValue when the text contains structural
   punctuation. */
#plot-content .scalar-display .value.composite {
  font-size: 16px; font-weight: normal; line-height: 1.4;
}
/* Importance-sampling quality readout in the toolbar.
   The base layout is set inline by renderSampleStats; this
   block only carries the colour by quality band. The same
   palette as the phase tags (with green added) so the visual
   vocabulary is consistent across phase / type / quality. */
.is-quality.is-good     { color: #66BB6A; }   /* green     */
.is-quality.is-ok       { color: #FFD54F; }   /* yellow    */
.is-quality.is-bad      { color: #FFB300; }   /* orange    */
.is-quality.is-unusable { color: #E57373; }   /* red       */
/* Graph internals fill graph-panel — switched from full-viewport
   sizing to 100% of the parent so the split-flex layout governs. */
#cy { width: 100%; height: 100%; }
/* All font-sizes in this stylesheet are relative units (em),
   so the panel scales with VS Code's zoom factor (which adjusts
   --vscode-font-size at the root). The body sets the base font
   size from --vscode-font-size; everything else here is a multiple
   of that. */
#info {
  min-height: 5.5em;
  padding: 0.6em 1em;
  border-top: 1px solid var(--vscode-panel-border, #444);
  font-size: 1em;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.3em;
}
#info .row { display: flex; gap: 0.75em; align-items: baseline; flex-wrap: wrap; }
#info .name { font-weight: 600; font-size: 1.15em; }
/* The inferred FlatPIR type/shape — sits to the right of the
   name and phase, monospaced so types like "array of real
   (length 10)" align consistently. */
#info .infer {
  font-size: 0.92em; opacity: 0.8;
  font-family: var(--vscode-editor-font-family, monospace);
  padding: 0.05em 0.45em; border-radius: 3px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
}
#info .phase {
  font-size: 0.92em;
  padding: 0.05em 0.45em; border-radius: 3px;
  color: #fff;
}
/* Phase tag colors. CSS custom properties are set at startup from
   the JS ctx.PALETTE so the in-bar tag and the node fill share one
   source of truth. */
#info .phase-fixed         { background: var(--phase-fixed);         color: #222; }
#info .phase-parameterized { background: var(--phase-parameterized); color: #222; }
#info .phase-stochastic    { background: var(--phase-stochastic);    color: #222; }
#info .expr {
  opacity: 0.6; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
  font-size: 1em;
}
#info .hint { opacity: 0.5; font-style: italic; font-size: 1em; }
#tooltip {
  position: absolute;
  display: none;
  pointer-events: none;
  background: var(--vscode-editorHoverWidget-background, #2d2d30);
  color: var(--vscode-editorHoverWidget-foreground, #ccc);
  border: 1px solid var(--vscode-editorHoverWidget-border, #454545);
  border-radius: 3px;
  padding: 4px 8px;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: var(--vscode-editor-font-size, 14px);
  white-space: pre;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  z-index: 100;
}
#tooltip .tooltip-expr { white-space: pre; }
#tooltip .tooltip-doc {
  margin-top: 4px;
  padding-top: 4px;
  border-top: 1px solid var(--vscode-editorHoverWidget-border, #454545);
  opacity: 0.95;
  font-family: var(--vscode-font-family, sans-serif);
  /* white-space:normal lets the marked-emitted HTML use its own
     block-level layout (paragraphs, lists, headings) instead of
     being treated as a single preformatted line. The doc body can
     still wrap; the outer #tooltip max-width:400px keeps it
     bounded. */
  white-space: normal;
  max-width: 380px;
}
/* Tighten the default user-agent margins inside the tooltip — full
   browser <p>/<h1>/<ul> margins are too generous in a hover popup
   and push the content off-screen quickly. The selectors stay
   scoped to #tooltip .tooltip-doc so nothing leaks into the rest
   of the viewer or the surrounding page. */
#tooltip .tooltip-doc p,
#tooltip .tooltip-doc ul,
#tooltip .tooltip-doc ol,
#tooltip .tooltip-doc pre {
  margin: 0.25em 0;
}
#tooltip .tooltip-doc h1,
#tooltip .tooltip-doc h2,
#tooltip .tooltip-doc h3,
#tooltip .tooltip-doc h4 {
  margin: 0.4em 0 0.2em;
  font-size: 1.05em;
  font-weight: 600;
}
#tooltip .tooltip-doc code {
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.95em;
  background: rgba(255, 255, 255, 0.07);
  padding: 0 3px;
  border-radius: 2px;
}
#tooltip .tooltip-doc pre {
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.95em;
  background: rgba(255, 255, 255, 0.05);
  padding: 4px 6px;
  border-radius: 3px;
  overflow-x: auto;
}
#tooltip .tooltip-doc pre code { background: none; padding: 0; }
/* MathML rendered by Temml. Inline math inherits surrounding text
   colour and size; display math is centred with a small vertical
   margin so it doesn't crowd adjacent prose. */
#tooltip .tooltip-doc math[display="block"] {
  display: block;
  margin: 0.4em 0;
  text-align: center;
}
.tooltip-typst-src {
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.95em;
  white-space: pre-wrap;
  margin: 0.25em 0;
}
#back-btn {
  display: none;
  background: var(--vscode-button-secondaryBackground, #3a3d41);
  color: var(--vscode-button-secondaryForeground, #ccc);
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 12px;
  cursor: pointer;
  font-family: var(--vscode-font-family, sans-serif);
  flex-shrink: 0;
  margin-right: 10px;
}
#back-btn:hover {
  background: var(--vscode-button-secondaryHoverBackground, #505355);
}
#collapse-all-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 5;
  background: var(--vscode-button-secondaryBackground, #3a3d41);
  color: var(--vscode-button-secondaryForeground, #ccc);
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 12px;
  cursor: pointer;
  font-family: var(--vscode-font-family, sans-serif);
}
#collapse-all-btn:hover {
  background: var(--vscode-button-secondaryHoverBackground, #505355);
}
`;

export var VIEWER_BODY_HTML = `
<div id="header">
<button id="back-btn">&larr; Back</button>
<span id="header-expr"></span>
<span id="inference-controls"></span>
<span id="view-controls">
<button id="graph-toggle" type="button" aria-pressed="true" title="Toggle the graph panel">Graph</button>
<button id="plot-toggle" type="button" aria-pressed="false" title="Toggle the plot panel">Plots</button>
<button id="math-toggle" type="button" aria-pressed="false" title="Toggle the math panel">Math</button>
</span>
</div>
<div id="main">
<div id="graph-panel" class="viewer-panel full">
  <div id="cy"></div>
  <button id="collapse-all-btn" type="button" title="Collapse every reification bubble in view">Collapse all</button>
</div>
<div id="plot-divider" class="viewer-divider hidden" title="Drag to resize"></div>
<div id="plot-panel" class="viewer-panel hidden">
  <div id="plot-content"></div>
</div>
<div id="math-divider" class="viewer-divider hidden" title="Drag to resize"></div>
<div id="math-panel" class="viewer-panel hidden">
  <div id="math-content"></div>
</div>
<p id="panels-hidden" hidden>Graph, plots and math are hidden.</p>
</div>
<div id="tooltip"></div>
<div id="info">
<span class="hint">Click a node or equation to see details <span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to sourcemiddot; double-click to drill down <span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to sourcemiddot; Ctrl+click to jump to source &middot; click &#8862;/&#8863; or Shift+click an anchor to collapse/expand that group</span>
</div>
`;

var cssInjected = false;
export function ensureCssInjected() {
  if (cssInjected) return;
  const styleEl = document.createElement('style');
  styleEl.setAttribute('data-flatppl-viewer-css', '');
  styleEl.textContent = VIEWER_CSS;
  document.head.appendChild(styleEl);
  cssInjected = true;
}
