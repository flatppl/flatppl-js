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
  overflow: hidden;
  text-overflow: ellipsis;
  opacity: 0.8;
  min-height: 26px;
  display: flex;
  align-items: center;
}
#header .target-name { font-weight: 600; }
#header .target-eq { opacity: 0.5; margin: 0 4px; }
/* Vertical split layout: graph on top, plot on bottom. The header
   hosts a Plot on/off toggle (default off). When the toggle is off
   the plot panel collapses to zero height and the graph fills the
   content area. When on, the area is split 60/40 between the two
   panels. The plot panel is always rendered when enabled — even
   for non-plottable bindings it shows a "Not plottable" message,
   so users navigating the graph see a stable layout instead of
   the panel appearing/disappearing.

   Heights subtract header(~32px) + info(60px). */
#plot-toggle {
  margin-left: auto;
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
#plot-toggle:hover { background: var(--vscode-button-secondaryHoverBackground, #505355); }
#plot-toggle.on {
  background: var(--vscode-button-background, #0e639c);
  color: var(--vscode-button-foreground, #fff);
  border-color: var(--vscode-button-border, transparent);
}
#plot-toggle.on:hover { background: var(--vscode-button-hoverBackground, #1177bb); }
#main {
  display: flex; flex-direction: column;
  width: 100vw; height: calc(100vh - 86px);
  overflow: hidden;
}
#graph-panel {
  flex: 1 1 60%; min-height: 80px;
  position: relative; overflow: hidden;
}
#graph-panel.full { flex: 1 1 100%; }
#plot-panel {
  flex: 1 1 40%; min-height: 80px;
  border-top: 1px solid var(--vscode-panel-border, #444);
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
  position: relative;
}
#plot-panel.hidden {
  flex: 0 0 0; min-height: 0; border-top: none;
}
#plot-content { width: 100%; height: 100%; }
/* Drag handle between #graph-panel and #plot-panel. Lets the user
   redistribute vertical space between the DAG view and the plot
   pane. Hidden when the plot panel itself is hidden; the existing
   border-top on #plot-panel doubles as the handle's visible band
   while in resting state, so the divider only adds the
   interactive hover affordance. */
#plot-divider {
  flex: 0 0 5px;
  cursor: row-resize;
  user-select: none;
  position: relative;
  background: transparent;
}
#plot-divider::before {
  content: '';
  position: absolute;
  left: 0; right: 0; top: 2px; bottom: 2px;
  background: transparent;
  transition: background 0.15s ease;
}
#plot-divider:hover::before {
  background: var(--vscode-button-background, #0e639c);
}
#plot-divider.hidden { display: none; }
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
`;

export var VIEWER_BODY_HTML = `
<div id="header">
<button id="back-btn">&larr; Back</button>
<span id="header-expr"></span>
<button id="plot-toggle" title="Toggle the plot panel">Plot: off</button>
</div>
<div id="main">
<div id="graph-panel" class="full">
  <div id="cy"></div>
</div>
<div id="plot-divider" class="hidden" title="Drag to resize"></div>
<div id="plot-panel" class="hidden">
  <div id="plot-content"></div>
</div>
</div>
<div id="tooltip"></div>
<div id="info">
<span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source</span>
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
