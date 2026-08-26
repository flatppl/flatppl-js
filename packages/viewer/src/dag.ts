// @flatppl/viewer — DAG (cytoscape) layer —
//
// initCy builds the cytoscape instance + style stanzas + tap/dbltap/
// hover/zoom handlers; renderDAG repopulates from a parsed module's
// data; focusNode / enterModuleView are the navigation entry points
// (drill into sub-DAGs / back to the module overview);
// drawReificationLassos draws the bubblesets around reification
// groups; teardownBubbles tears them down on free-event. The small
// info-bar helpers (showNodeInfo, updateHeader, updateBackBtn) live
// here too — they're only called from the DAG event handlers.

import { updatePlotForBinding } from './render-plot.js';
import { $, bubbleMemberIds, esc, hexToRgba, truncateExpr } from './util.js';
import { renderDoc } from './markdown.js';
import { resolveNodeColor } from './palette.js';
import { errorsForBinding, saveViewState } from './render-frame.js';
import type { Ctx } from './types';
// The `load_module` registry entry for a DAG node, if it is a user
// module-load binding with a resolved path (spec §04) — the handle for
// cross-module navigation. Returns null for any other node.
function _loadModuleEntry(ctx: Ctx, nodeId: any): { path: string } | null {
  const reg = ctx.currentLoweredModule && ctx.currentLoweredModule.moduleRegistry;
  const e = reg && reg[nodeId];
  return (e && e.kind === 'load_module' && e.path) ? e : null;
}

export function showNodeInfo(ctx: Ctx, d: any) {
  const phase = d.phase || 'unknown';
  const phaseTag = '<span class="phase phase-' + esc(phase) + '">' + esc(phase) + ' phase</span>';
  let unsupportedRow = '';
  if (d.unsupported) {
    let msg = 'disintegration unresolved: ' + esc(d.unsupportedReason || '');
    if (d.unsupportedDetail) msg += ' — ' + esc(d.unsupportedDetail);
    unsupportedRow = '<div class="expr" style="color:#FF8A65;">' + msg + '</div>';
  }
  // Type-error row(s). Drawn in the same red as the node border so
  // the visual link reads at a glance. Each diagnostic gets its own
  // line — a single binding can pick up several mismatches if its
  // RHS has multiple bad arg positions.
  let errorRow = '';
  const errors = errorsForBinding(ctx, d.id);
  if (errors && errors.length > 0) {
    for (let i = 0; i < errors.length; i++) {
      errorRow += '<div class="expr" style="color:#E57373;">' + esc(errors[i].message) + '</div>';
    }
  }
  // Construction kind (binding.type — draw, lawof, call, …) is
  // intentionally omitted: the expression always starts with the
  // operator, and the DAG node's shape + color already encodes
  // the same axis. The inferred FlatPIR type/shape carries
  // strictly richer information (structural result type) and
  // takes that pill's slot.
  const inferTag = d.inferredType
    ? '<span class="infer">' + esc(d.inferredType) + '</span>'
    : '';
  // A `load_module` node is navigable — surface that it opens another
  // file on double-click (spec §04 cross-module navigation).
  let moduleRow = '';
  const modEntry = _loadModuleEntry(ctx, d.id);
  if (modEntry) {
    moduleRow = '<div class="expr" style="color:#80CBC4;">↳ loaded module — '
      + 'double-click to open ' + esc(modEntry.path) + '</div>';
  }
  $('info').innerHTML =
    '<div class="row"><span class="name">' + esc(d.label) + '</span>'
    + phaseTag
    + inferTag + '</div>'
    + '<div class="expr">' + esc(d.expr) + '</div>'
    + moduleRow
    + unsupportedRow
    + errorRow;
}

export function updateHeader(ctx: Ctx, data: any) {
  const el = $('header-expr');
  // Module view: no per-node target; just label the view.
  if (ctx.currentState && ctx.currentState.targetName === ctx.MODULE_TARGET) {
    el.innerHTML = '<span class="target-name">module</span>';
    return;
  }
  let target: any = null;
  for (let i = 0; i < data.nodes.length; i++) {
    if (data.nodes[i].isTarget) { target = data.nodes[i]; break; }
  }
  if (!target) { el.innerHTML = ''; return; }
  const name = target.label || target.id;
  const expr = truncateExpr(target.expr);
  el.innerHTML = '<span class="target-name">' + esc(name) + '</span>'
    + (expr ? '<span class="target-eq">=</span>' + esc(expr) : '');
}

export function updateBackBtn(ctx: Ctx) {
  $('back-btn').style.display = ctx.history.length > 0 ? 'block' : 'none';
}

export function teardownBubbles(ctx: Ctx) {
  if (!ctx.bb) return;
  ctx.bb.getPaths().forEach(function(p: any) {
    p.update = function() {};
    ctx.bb.removePath(p);
  });
  ctx.cy.elements().forEach(function(el: any) { el.removeScratch('bubbleSets'); });
}

export function initCy(ctx: Ctx) {
  ctx.cy = cytoscape({
    container: $('cy'),
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '13px',
          'color': '#333',
          'background-color': 'data(color)',
          'shape': 'data(shape)',
          'width': 'data(width)',
          'height': 36,
          'border-width': 2,
          'border-color': '#888',
        }
      },
      {
        // Reification anchor nodes — bindings that head a
        // reification group (lawof / functionof / kernelof / fn
        // with internal kernel members). They sit at the entrance
        // of their bubble; the translucent fill + same-color
        // border read as belonging to the bubble rather than
        // floating inside it.
        //
        // Selecting on the engine-computed isReifAnchor flag
        // (rather than nodeType alone) excludes synthesized
        // measure bindings that happen to have type=lawof but no
        // visible bubble (e.g. prior2 = lawof(disintegrate(…))
        // where disintegrate produces a closed-form rewrite, no
        // new scope to render). Those fall through to the default
        // solid fill — same visual treatment as joint_model and
        // other measure-producing operations without a bubble.
        selector: 'node[?isReifAnchor]',
        style: {
          'background-color': 'data(color)',
          'background-opacity': 0.18,
          'border-color': 'data(color)',
          'border-width': 1.5,
          'color': 'data(color)',
        }
      },
      {
        selector: 'node[?isBoundary]',
        style: {
          'border-color': '#FFD600',
          'border-width': 3,
          'border-style': 'dashed',
        }
      },
      {
        // Disintegration result whose Plan came back Unsupported —
        // the trace through it is the user's literal source, not a
        // structural decomposition. Dotted orange border distinguishes
        // it from boundary inputs (dashed yellow) and target (solid blue).
        selector: 'node[?unsupported]',
        style: {
          'border-color': '#FF8A65',
          'border-width': 3,
          'border-style': 'dotted',
        }
      },
      {
        // Bindings with analyzer-level error diagnostics (typeinfer
        // mismatch, undefined ref, etc.) get a solid red border.
        // Distinct from the dashed yellow boundary and dotted orange
        // unsupported markers so the three semantic signals don't
        // collide visually.
        selector: 'node[?hasError]',
        style: {
          'border-color': '#E57373',
          'border-width': 3,
          'border-style': 'solid',
        }
      },
      {
        selector: 'node[?isTarget]',
        style: {
          'border-color': '#1565C0',
          'border-width': 4,
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 2,
          'line-color': '#999',
          'target-arrow-color': '#999',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'arrow-scale': 1.0,
        }
      },
      {
        selector: 'edge[edgeType = "call"]',
        style: {
          'line-style': 'dashed',
          'line-dash-pattern': [6, 4],
          'line-color': '#bbb',
          'target-arrow-color': '#bbb',
          'width': 1.5,
        }
      },
      {
        // Draw edges: the boundary between deterministic and
        // stochastic. Solid line in a darker purple than the
        // node fill so it reads boldly as a line; thicker than
        // dataflow edges so the eye lands on where stochasticity
        // enters the model.
        selector: 'edge[edgeType = "draw"]',
        style: {
          'line-color': ctx.DRAW_EDGE_COLOR,
          'target-arrow-color': ctx.DRAW_EDGE_COLOR,
          'width': 2.5,
        }
      },
      {
        // Hidden edges — present so dagre uses them for layout, but
        // not rendered (the enclosing bubble conveys the relation).
        selector: 'edge[?hidden]',
        style: {
          'visibility': 'hidden',
        }
      },
      {
        // Tether: faint connection from a reified value to its
        // reification node. Same kernel-internal flow as the hidden
        // edges, but drawn so you can see what is being reified.
        // Labeled with the reification keyword (lawof / functionof /
        // kernelof / fn) so the operation is legible without having
        // to read the target node.
        selector: 'edge[edgeType = "tether"]',
        style: {
          'line-color': function(ele: any) { return ele.target().data('color') || '#aaa'; },
          'opacity': 0.6,
          'width': 1.5,
          'target-arrow-shape': 'none',
          'curve-style': 'straight',
          'label': 'data(tetherLabel)',
          'font-size': '10px',
          'font-style': 'italic',
          'color': function(ele: any) { return ele.target().data('color') || '#aaa'; },
          // Full text opacity overrides the edge's 0.6 — the line stays
          // faint, the label reads as bright as a node label.
          'text-opacity': 1,
          // Center the label on the line and let an opaque background
          // pad visually break the line at the label — the tether
          // appears to connect into the lawof/kernelof/… box on both
          // sides, like a labeled link in an electrical schematic.
          // Literal hex (not a CSS var) — cytoscape draws on HTML canvas
          // and cannot resolve "var(--name)" values, so a CSS variable
          // would silently fall back to a transparent background and
          // let the line show through.
          'text-rotation': 'autorotate',
          'text-background-color': '#1e1e1e',
          'text-background-opacity': 1,
          'text-background-padding': '2px',
          'text-background-shape': 'roundrectangle',
          'text-border-width': 1,
          'text-border-color': function(ele: any) { return ele.target().data('color') || '#aaa'; },
          'text-border-opacity': 0.6,
        }
      },
      {
        selector: 'node:selected',
        style: {
          'border-color': '#2196F3',
          'border-width': 3,
          'overlay-opacity': 0,
        }
      },
      {
        // Collapsed reification anchor: its bubble's members are hidden
        // behind it (see renderDAG's `dropped` map). Double border reads
        // as "this node stands in for a container", distinct from the
        // thin translucent isReifAnchor treatment an expanded anchor gets.
        // Listed after node[?isReifAnchor] so border-width/style here win
        // on a node matching both selectors.
        selector: 'node[?collapsed]',
        style: {
          'border-style': 'double',
          'border-width': 7,
        }
      },
    ],
    elements: [],
    layout: { name: 'preset' },
    wheelSensitivity: 2,
  });

  if (typeof ctx.cy.bubbleSets === 'function') {
    // bubblesets uses one scratch key per cytoscape node; when paths
    // share nodes (e.g. theta1 belongs to both prior and forward_kernel),
    // their cached geometry stomps on each other and one path goes empty
    // on update. Workaround: tear down and rebuild all paths on drag
    // release, rAF-batched. Updates skipped during drag for snappiness.
    ctx.bb = ctx.cy.bubbleSets({ interactive: false });
    let bbRedrawScheduled = false;
    ctx.cy.on('free', 'node', function() {
      if (!ctx.bb || bbRedrawScheduled || !ctx.currentState) return;
      bbRedrawScheduled = true;
      requestAnimationFrame(function() {
        bbRedrawScheduled = false;
        if (ctx.currentState) drawReificationLassos(ctx, ctx.currentState.data);
      });
    });
  }

  // Ctrl/Cmd+click: jump to source.
  // Plain click: select the node — info bar updates AND the plot
  // panel re-targets to this binding. The plot follows the
  // selection rather than the DAG's terminal target so users can
  // explore the graph node-by-node and read each binding's
  // distribution in place.
  ctx.cy.on('tap', 'node', function(evt: any) {
    const oe = evt.originalEvent;
    // Shift+click on a reification anchor (collapsed or not) toggles
    // its bubble's collapse state and re-renders. Plain tap keeps its
    // select-and-plot meaning below; dbltap keeps drill-in — neither
    // is touched by this gesture.
    if (oe && oe.shiftKey) {
      const d = evt.target.data();
      if (d.collapsed || d.isReifAnchor) {
        toggleReification(ctx, d.id);
      }
      return;
    }
    if (oe && (oe.ctrlKey || oe.metaKey)) {
      const line = evt.target.data('line');
      if (line >= 0) {
        if (ctx.host.revealSourceLine) ctx.host.revealSourceLine(line);
      }
      return;
    }
    const d = evt.target.data();
    showNodeInfo(ctx, d);
    // Cross-module member node (spec §04, "Stochastic boundary": only
    // fixed/parameterized members are reachable across the boundary, so
    // this always lands on the profile or fixed-value paths, never a
    // sampled histogram). `moduleMember` is also stamped on an ORDINARY
    // top-level alias of a member (`f_b = common.f_b`, for dbltap's
    // drill-in) — that node's id is its own plain name and it's already
    // a key in ctx.currentBindings, so it must NOT take this branch; only
    // the anonymous `module.field` access node (id === the dotted form)
    // has no such key. buildPlotPlan/updatePlotForBinding resolve THAT
    // one against the LINKED binding graph instead, under its
    // `module$field` name (module-link.ts), via a shallow ctx clone. The
    // clone's writes (currentPlotPlan etc.) don't touch the real ctx, so
    // the DAG's own selection/back-stack state is untouched — this is a
    // one-off side-plot, not a navigation.
    const mm = d.moduleMember;
    if (mm && mm.module && d.id === mm.module + '.' + mm.field && ctx.currentLinkedBindings) {
      const linkedName = mm.module + '$' + mm.field;
      updatePlotForBinding({ ...ctx, currentBindings: ctx.currentLinkedBindings }, linkedName);
      return;
    }
    // Always re-target the plot to whatever the user clicked. For
    // synthetic nodes (anonymous inline expressions, placeholders,
    // holes — recognised by ':' in the id) there's no binding to
    // sample, so updatePlotForBinding ends up rendering a
    // "Not plottable" placeholder. Either way the plot reflects
    // the current selection rather than a stale earlier focus.
    updatePlotForBinding(ctx, d.id);
  });

  ctx.cy.on('tap', function(evt: any) {
    if (evt.target === ctx.cy) {
      $('info').innerHTML = '<span class="hint">' + ctx.HINT + '</span>';
    }
  });

  // Double-click: drill into node's sub-DAG. Handled locally — the
  // webview owns the parsed bindings and recomputes the sub-DAG itself
  // (no host round-trip). Title sync to the editor still goes via a
  // postMessage to the host since the title is on the VS Code panel.
  ctx.cy.on('dbltap', 'node', function(evt: any) {
    const node = evt.target;
    const nodeId = node.data('id');
    // Don't drill into synthetic nodes (placeholder/hole inputs).
    if (nodeId.indexOf(':') !== -1) return;
    // A cross-module member node (`common.f_a`) OR a bare alias of one
    // (`f_b = common.f_b`): double-click drills INTO the loaded module's
    // DAG focused on that member (spec §04) — as if its source were opened
    // and DAG-view selected for it. The owning module's resolved path
    // comes from the moduleRegistry keyed by the member's module name.
    // Drilling INTO a loaded module pushes the loader's CURRENT state first so
    // the in-DAG back-button can return here: the host's openModule REPLACES
    // the model (web router / VS Code editor) and its cross-model re-render
    // does NOT push (the hosts pass pushHistory=false on that re-render), so
    // without this the loader drops out of history and there's no way back.
    const pushLoaderState = function () {
      if (ctx.currentState) {
        ctx.history.push(ctx.currentState);
        if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
      }
    };
    const mm = node.data('moduleMember');
    if (mm && mm.module && ctx.host && typeof ctx.host.openModule === 'function') {
      const owner = _loadModuleEntry(ctx, mm.module);
      if (owner) { pushLoaderState(); ctx.host.openModule(owner.path, mm.field); return; }
    }
    // The `load_module` binding itself: drill into the WHOLE loaded module
    // (no member focus). A module boundary has no sub-DAG in the primary
    // graph; the resolved path is the bundle / router key from lowering.
    const modEntry = _loadModuleEntry(ctx, nodeId);
    if (modEntry && ctx.host && typeof ctx.host.openModule === 'function') {
      pushLoaderState(); ctx.host.openModule(modEntry.path);
      return;
    }
    focusNode(ctx, nodeId, /* pushHistory */ true);
    if (ctx.host.setTitle) ctx.host.setTitle(nodeId);
  });

  const tip = $('tooltip');
  // Hover-intent delay before the tooltip appears. The doc-comment now renders
  // as full HTML/MathML, so the tooltip can be large enough to sit over the
  // node beneath; an INSTANT pop made neighbouring nodes hard to aim at while
  // moving the cursor across to click one. (#tooltip is pointer-events:none so
  // it never eats the click — this is purely about it not flashing in the way.)
  const TOOLTIP_DELAY_MS = 450;
  let tipTimer: any = null;
  function hideTip() {
    if (tipTimer) { clearTimeout(tipTimer); tipTimer = null; }
    tip.style.display = 'none';
  }
  ctx.cy.on('mouseover', 'node', function(evt: any) {
    const d = evt.target.data();
    const expr = d.expr || '';
    const doc = d.doc;
    const hasDoc = doc && Array.isArray(doc.lines) && doc.lines.length > 0;
    if (!expr && !hasDoc) return;
    if (tipTimer) clearTimeout(tipTimer);
    const pos = evt.renderedPosition;  // node position; stable until a viewport move (which hides the tip)
    tipTimer = setTimeout(function() {
      tipTimer = null;
      const cyEl = $('cy');
      if (!cyEl) return;  // viewer torn down mid-hover
      // Tooltip shows the existing `label = expr` line on top; the attached
      // doc-comment renders below in dimmer styling (spec §04
      // §sec:documentation). The doc body goes through marked + Temml (see
      // markdown.ts) so `**bold**`, lists, code, and math render as
      // HTML/MathML. The `label = expr` line stays plain-text (textContent) —
      // no markup expected there, and textContent dodges accidental HTML in
      // identifier-derived strings.
      tip.textContent = '';
      if (expr) {
        const exprLine = document.createElement('div');
        exprLine.className = 'tooltip-expr';
        exprLine.textContent = d.label ? (d.label + ' = ' + expr) : expr;
        tip.appendChild(exprLine);
      }
      if (hasDoc) {
        const docBlock = document.createElement('div');
        docBlock.className = 'tooltip-doc';
        const html = renderDoc(doc);
        if (html) docBlock.innerHTML = html;
        else      docBlock.textContent = doc.lines.join('\n');
        tip.appendChild(docBlock);
      }
      tip.style.display = 'block';
      const cRect = cyEl.getBoundingClientRect();
      let tx = pos.x + cRect.left + 12;
      let ty = pos.y + cRect.top - 30;
      if (tx + tip.offsetWidth > cRect.right - 8) tx = cRect.right - tip.offsetWidth - 8;
      if (ty < cRect.top + 4) ty = pos.y + cRect.top + 16;
      tip.style.left = tx + 'px';
      tip.style.top = ty + 'px';
    }, TOOLTIP_DELAY_MS);
  });
  ctx.cy.on('mouseout', 'node', hideTip);
  ctx.cy.on('viewport', hideTip);
}

export function drawReificationLassos(ctx: Ctx, data: any) {
  if (!ctx.bb || !data.reifications) return;
  teardownBubbles(ctx);

  for (let k = 0; k < data.reifications.length; k++) {
    const r = data.reifications[k];
    if (r.kernel.length < 2) continue;
    if (!ctx.TYPE_STYLE[r.type]) continue;
    // Collapsed: its members aren't in the graph to lasso around.
    if (ctx.collapsedReifications.has(r.name)) continue;
    // Same colour the bubble's reification node would get — keeps
    // bubble fill, bubble stroke, and node fill in lockstep.
    const bubbleColor = resolveNodeColor(ctx, r);

    const memberIds = bubbleMemberIds(r, data.reifications);
    var nodes = ctx.cy.collection();
    for (const memId in memberIds) {
      nodes = nodes.union(ctx.cy.getElementById(memId));
    }
    // Hidden edges (visibility:hidden) can return undefined endpoints,
    // which silently corrupts bubblesets' potential field — exclude.
    const edges = ctx.cy.edges().filter(function(e: any) {
      return nodes.contains(e.source())
        && nodes.contains(e.target())
        && !e.data('hidden');
    });
    const avoid = ctx.cy.nodes().difference(nodes);

    ctx.bb.addPath(nodes, edges, avoid, {
      // virtualEdges: connect spatially-disconnected member groups via
      // routed connectors. Required for kernels spread across the
      // canvas — marching squares only traces one component per call.
      virtualEdges: true,
      style: {
        fill: hexToRgba(bubbleColor, 0.12),
        stroke: bubbleColor,
        strokeWidth: '1.5px',
        strokeOpacity: '0.7',
      },
    });
  }
}

/**
 * Flip one reification anchor's collapse state and re-render the
 * currently focused sub-DAG. No-op with no current view (nothing to
 * re-render against).
 */
export function toggleReification(ctx: Ctx, anchorName: string): void {
  if (ctx.collapsedReifications.has(anchorName)) ctx.collapsedReifications.delete(anchorName);
  else ctx.collapsedReifications.add(anchorName);
  saveViewState(ctx);
  if (ctx.currentState) renderDAG(ctx, ctx.currentState.data);
}

/**
 * Toolbar collapse-all / expand-all: acts on every reification
 * anchor in the currently rendered sub-DAG (not the whole model —
 * consistent with renderDAG itself only ever seeing one sub-DAG's
 * `data.reifications` at a time).
 */
export function toggleAllReifications(ctx: Ctx, collapse: boolean): void {
  if (!ctx.currentState || !ctx.currentState.data.reifications) return;
  const reifications = ctx.currentState.data.reifications;
  for (let i = 0; i < reifications.length; i++) {
    const name = reifications[i].name;
    if (collapse) ctx.collapsedReifications.add(name);
    else ctx.collapsedReifications.delete(name);
  }
  saveViewState(ctx);
  renderDAG(ctx, ctx.currentState.data);
}

/**
 * Follow a chain of dropped-member -> anchor mappings to the
 * surviving anchor at its end. A collapsed reification's anchor can
 * itself be a member of an OUTER collapsed reification — e.g.
 * phase_weight ⊃ intensity_fn ⊃ angular_tensors in
 * dminus-to-3pi-amplitude.flatppl, where each anchor is directly
 * listed as a member of the one enclosing it. A single-hop
 * `dropped.get(id)` stops at `intensity_fn`, which is itself dropped
 * under `phase_weight` — pure-testable seam for the #176 regression
 * (a dangling edge crashed cytoscape's whole render).
 */
export function resolveDroppedChain(dropped: Map<string, string>, id: string): string {
  let cur = id;
  let hops = 0;
  while (dropped.has(cur) && hops < dropped.size + 1) {
    cur = dropped.get(cur) as string;
    hops++;
  }
  return cur;
}

/**
 * Rewrite a collapsed sub-DAG's edges through `dropped` (chain-
 * resolved via resolveDroppedChain) so every endpoint that used to
 * point into a hidden bubble member now points at the surviving
 * anchor instead. A self-edge produced by both endpoints resolving to
 * the same anchor is dropped (the bubble it explained is gone).
 * Duplicate (source, target, edgeType) triples produced by collapsing
 * distinct member-to-member edges into one anchor-to-anchor edge are
 * deduped. Finally, ANY edge whose resolved endpoint still isn't in
 * `survivingNodeIds` is dropped rather than kept — cytoscape throws
 * hard on a dangling edge and takes down the whole render, so a gap
 * in the drop/resolve accounting must fail quiet, not fail loud.
 */
export function rewriteEdgesForCollapse(
  edges: Array<{ source: string; target: string; edgeType: string; [extra: string]: any }>,
  dropped: Map<string, string>,
  survivingNodeIds: Set<string>,
): Array<{ source: string; target: string; edgeType: string; [extra: string]: any }> {
  const seenEdgeKeys = new Set<string>();
  const out: Array<{ source: string; target: string; edgeType: string; [extra: string]: any }> = [];
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const src = resolveDroppedChain(dropped, e.source);
    const tgt = resolveDroppedChain(dropped, e.target);
    if (src === tgt) continue;
    if (!survivingNodeIds.has(src) || !survivingNodeIds.has(tgt)) continue;
    const key = src + '|' + tgt + '|' + e.edgeType;
    if (seenEdgeKeys.has(key)) continue;
    seenEdgeKeys.add(key);
    out.push(Object.assign({}, e, { source: src, target: tgt }));
  }
  return out;
}

export function renderDAG(ctx: Ctx, data: any) {
  if (!ctx.cy) initCy(ctx);
  updateHeader(ctx, data);

  const elements: any[] = [];

  // Reification anchor names — bindings that head a reification
  // group (i.e. spawn a bubble with internal kernel members).
  // Used to gate the "hollow fill" cytoscape style: only nodes
  // that actually anchor a visible bubble get the translucent
  // treatment, so synthesized bindings like prior2 =
  // lawof(disintegrate(...)) (no internal scope, no bubble
  // drawn) render with the default solid measure style.
  const reifAnchorNames: Record<string, boolean> = {};
  if (data.reifications) {
    for (let ra = 0; ra < data.reifications.length; ra++) {
      reifAnchorNames[data.reifications[ra].name] = true;
    }
  }

  // Graph-view compactor: decide which reification bubbles render
  // collapsed. A bubble not yet decided this session defaults to
  // collapsed at >= 3 members (`_reifSeen` marks the decision so a
  // later re-render, e.g. after shift+click-expanding a *different*
  // anchor, doesn't reconsider this one). `dropped` maps every member
  // id hidden by a collapsed bubble to its anchor's name; moduleMember
  // nodes are excluded so cross-module drill-in targets stay tappable
  // even while their enclosing bubble is collapsed.
  const dropped: Map<string, string> = new Map();
  const nodeById: Record<string, any> = {};
  for (let ni2 = 0; ni2 < data.nodes.length; ni2++) nodeById[data.nodes[ni2].id] = data.nodes[ni2];
  if (data.reifications) {
    for (let rd = 0; rd < data.reifications.length; rd++) {
      const r = data.reifications[rd];
      if (!ctx._reifSeen.has(r.name)) {
        ctx._reifSeen.add(r.name);
        const memberCount = Object.keys(bubbleMemberIds(r, data.reifications)).length;
        if (memberCount >= 3) ctx.collapsedReifications.add(r.name);
      }
      if (!ctx.collapsedReifications.has(r.name)) continue;
      const memberIds = bubbleMemberIds(r, data.reifications);
      for (const memId in memberIds) {
        if (memId === r.name) continue;
        const n = nodeById[memId];
        if (n && n.moduleMember) continue;
        dropped.set(memId, r.name);
      }
    }
  }
  const dropCountByAnchor: Record<string, number> = {};
  dropped.forEach(function(_anchorName, memId) {
    const finalAnchor = resolveDroppedChain(dropped, memId);
    dropCountByAnchor[finalAnchor] = (dropCountByAnchor[finalAnchor] || 0) + 1;
  });

  for (let i = 0; i < data.nodes.length; i++) {
    const node = data.nodes[i];
    // Hidden behind a collapsed reification bubble it belongs to.
    if (dropped.has(node.id)) continue;
    const ts = ctx.TYPE_STYLE[node.type] || ctx.TYPE_STYLE.unknown;

    // Shape: type-driven (carries the structural info — what *kind*
    // of binding this is). The engine-computed reification kind
    // overrides for "functionof acting on a measure → render as a
    // kernel" so the user sees a kernel regardless of which
    // keyword they wrote.
    let shape = ts.shape;
    if (node.kind === 'kernel')      shape = 'round-hexagon';
    else if (node.kind === 'measure') shape = 'round-rectangle';

    const color = resolveNodeColor(ctx, node);
    // Anonymous nodes (inline-expression targets) have label === ''
    // deliberately and show their expression on hover only. Others
    // fall back to their id.
    let displayLabel = node.label === '' ? '' : (node.label || node.id);
    const collapsed = !!reifAnchorNames[node.id] && ctx.collapsedReifications.has(node.id);
    if (collapsed) displayLabel = displayLabel + '  ⊞' + (dropCountByAnchor[node.id] || 0);
    const width = displayLabel === ''
      ? 60
      : Math.max(displayLabel.length * 9 + 24, 60);
    elements.push({
      group: 'nodes',
      data: {
        id: node.id,
        label: displayLabel,
        color: color,
        shape: shape,
        nodeType: node.type,
        phase: node.phase || '',
        expr: node.expr || '',
        doc: node.doc || null,
        line: node.line != null ? node.line : -1,
        isBoundary: node.isBoundary || false,
        isTarget: node.isTarget || false,
        unsupported: !!node.unsupported,
        unsupportedReason: node.unsupportedReason || '',
        unsupportedDetail: node.unsupportedDetail || '',
        inferredType: node.inferredType || '',
        hasError: !!(node.errors && node.errors.length > 0),
        isReifAnchor: !!reifAnchorNames[node.id],
        collapsed: collapsed,
        // Cross-module member node (spec §04): `{ module, field }` drill-in
        // target for the dbltap handler; null for ordinary bindings.
        moduleMember: node.moduleMember || null,
        width: width,
      },
    });
  }

  // For edges entering a reification node from inside its bubble:
  //   - if source is one of the reification's targets (the value being
  //     reified): keep visible but render as a faint "tether"
  //   - else (boundary arg or other kernel member): fully hide; the
  //     bubble already conveys that flow. Edge is kept in cy so dagre
  //     uses it for layout.
  const reifMembers: Record<string, Record<string, boolean>> = {}; // reifName -> {memberId: true}
  const reifTargets: Record<string, Record<string, boolean>> = {}; // reifName -> {targetId: true}
  if (data.reifications) {
    for (let ri = 0; ri < data.reifications.length; ri++) {
      const rf = data.reifications[ri];
      reifMembers[rf.name] = {};
      for (let mi = 0; mi < rf.kernel.length; mi++) reifMembers[rf.name][rf.kernel[mi]] = true;
      reifTargets[rf.name] = {};
      const ts2 = rf.targets || [];
      for (let ti = 0; ti < ts2.length; ti++) reifTargets[rf.name][ts2[ti]] = true;
    }
  }

  // Map binding name -> binding type, used to label tether edges with
  // the reification keyword (lawof / functionof / kernelof / fn).
  const typeByName: Record<string, string> = {};
  for (let ni = 0; ni < data.nodes.length; ni++) {
    typeByName[data.nodes[ni].id] = data.nodes[ni].type;
  }

  for (let j = 0; j < data.edges.length; j++) {
    const edge = data.edges[j];
    let edgeType = edge.edgeType || 'data';
    let hidden = false;
    const membersForTarget = reifMembers[edge.target];
    if (membersForTarget && membersForTarget[edge.source] && edge.source !== edge.target) {
      if (reifTargets[edge.target] && reifTargets[edge.target][edge.source]) {
        edgeType = 'tether';
      } else {
        hidden = true;
      }
    }
    let tetherLabel = '';
    if (edgeType === 'tether') {
      const t = typeByName[edge.target];
      if (t === 'lawof' || t === 'functionof' || t === 'kernelof'
          || t === 'fn' || t === 'bijection' || t === 'fchain') {
        tetherLabel = t;
      }
    }
    elements.push({
      group: 'edges',
      data: {
        source: edge.source,
        target: edge.target,
        edgeType: edgeType,
        hidden: hidden,
        tetherLabel: tetherLabel,
      },
    });
  }

  // Rewrite edges through `dropped` (chain-resolved) and drop any
  // edge a future gap in the accounting still leaves dangling — see
  // rewriteEdgesForCollapse's own doc comment for the #176 story.
  if (dropped.size > 0) {
    const survivingNodeIds = new Set<string>();
    const nodeElements: any[] = [];
    const edgeData: any[] = [];
    for (let ei = 0; ei < elements.length; ei++) {
      const el = elements[ei];
      if (el.group === 'nodes') { survivingNodeIds.add(el.data.id); nodeElements.push(el); }
      else edgeData.push(el.data);
    }
    const newEdgeData = rewriteEdgesForCollapse(edgeData, dropped, survivingNodeIds);
    elements.length = 0;
    for (let ei = 0; ei < nodeElements.length; ei++) elements.push(nodeElements[ei]);
    for (let ei = 0; ei < newEdgeData.length; ei++) elements.push({ group: 'edges', data: newEdgeData[ei] });
  }

  // Tear down old bubble paths BEFORE detaching elements so we can
  // clear scratch on still-attached cytoscape elements.
  teardownBubbles(ctx);
  ctx.cy.elements().remove();
  ctx.cy.add(elements);

  ctx.cy.layout({
    name: 'dagre',
    rankDir: 'TB',
    nodeSep: 40,
    rankSep: 55,
    padding: 30,
    animate: false,
  }).run();

  ctx.cy.fit(undefined, 40);
  drawReificationLassos(ctx, data);

  // Show details for the target node automatically (the cursor is already
  // on it in the source). Falls back to the hint if no target is present.
  const target = data.nodes.find(function(n: any) { return n.isTarget; });
  if (target) {
    showNodeInfo(ctx, {
      label: target.label || target.id,
      nodeType: target.type,
      phase: target.phase || '',
      expr: target.expr || '',
    });
  } else {
    $('info').innerHTML = '<span class="hint">' + ctx.HINT + '</span>';
  }
}

/**
 * Re-render the DAG focused on targetName using the cached bindings.
 * If pushHistory is true, the current view is pushed onto the back-
 * button stack first. If targetName is null, falls back to the last
 * binding in document order (the same default the extension ctx.host used
 * before this refactor).
 *
 * `opts.autoTrigger` passes through to updatePlotForBinding / renderPlot
 * ForCurrent — set by applySourceUpdate on an actual model edit so a
 * stateful sampler backend (MH/nested/AMIS/SMC/…) isn't auto-resampled.
 * Explicit navigation (dbltap drill-down below) omits it.
 */
export function focusNode(ctx: Ctx, targetName: any, pushHistory: any, opts?: { autoTrigger?: boolean }) {
  if (!ctx.currentBindings) return;
  // No targetName supplied → prefer keeping the current focus.
  // This is the path used by source-only updates from the host
  // (the user is editing the RHS of the already-shown binding —
  // they don't want their place reset to "last binding"). Falls
  // through to the last binding when there's no prior focus or
  // the focused binding was deleted by the edit.
  if (!targetName) {
    if (ctx.currentState && ctx.currentBindings.has(ctx.currentState.targetName)) {
      targetName = ctx.currentState.targetName;
    } else {
      const allNames: string[] = [];
      ctx.currentBindings.forEach(function(_b: unknown, name: string) { allNames.push(name); });
      if (allNames.length === 0) return;
      targetName = allNames[allNames.length - 1];
    }
  }
  const dagData = FlatPPLEngine.computeSubDAG(ctx.currentBindings, targetName,
    { linkedBindings: ctx.currentLinkedBindings });
  if (!dagData || dagData.nodes.length === 0) return;

  // History grows only when (a) the caller asked us to push, and
  // (b) the target actually changed from what's currently shown.
  //   - cursor moves / ctrl-click / drill-down → push (target moved)
  //   - source-only updates (RHS edits) → no-op (target preserved)
  //   - same-target refocus → no-op
  // Capped at HISTORY_CAP entries to bound memory: each entry holds
  // a sub-DAG's nodes + edges (~few KB), so a few hundred entries
  // is plenty for navigation but well below any pressure point. On
  // overflow we drop the oldest entry (FIFO trim) — going way back
  // is rare enough that this is the right trade-off.
  if (pushHistory && ctx.currentState && ctx.currentState.targetName !== targetName) {
    ctx.history.push(ctx.currentState);
    if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
  }

  ctx.currentState = { data: dagData, targetName: targetName, path: ctx.currentPath };
  renderDAG(ctx, dagData);
  updateBackBtn(ctx);
  updatePlotForBinding(ctx, targetName, opts);
  // Notify the host so any URL / panel state stays in sync with
  // the viewer's actual focus. Internal navigations (DAG node
  // clicks, double-clicks, "show whole module" toolbar) used to
  // diverge from the host's recorded target, which then leaked
  // back into the viewer when the host pushed a fresh
  // sourceUpdate carrying its (stale) target — e.g. typing in
  // an editor triggered a debounced update that yanked focus
  // back to a previous binding. With this call, host and viewer
  // share one target.
  if (ctx.host && typeof ctx.host.setTarget === 'function') {
    try { ctx.host.setTarget(targetName); } catch (_) {}
  }
}

/**
 * Render the module-level (multi-root) DAG. Plot pane shows a
 * "click a binding to plot it" message because there's no single
 * focused binding here. Pushes onto ctx.history when requested and
 * the previous view wasn't already the module view.
 */
export function enterModuleView(ctx: Ctx, pushHistory: any) {
  if (!ctx.currentBindings) return;
  const dagData = FlatPPLEngine.computeFullDAG(ctx.currentBindings,
    { linkedBindings: ctx.currentLinkedBindings });
  if (!dagData || dagData.nodes.length === 0) return;

  if (pushHistory && ctx.currentState && ctx.currentState.targetName !== ctx.MODULE_TARGET) {
    ctx.history.push(ctx.currentState);
    if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
  }

  ctx.currentState = { data: dagData, targetName: ctx.MODULE_TARGET, path: ctx.currentPath };
  renderDAG(ctx, dagData);
  updateBackBtn(ctx);
  // Mirror module-view focus to the host (null = whole module).
  if (ctx.host && typeof ctx.host.setTarget === 'function') {
    try { ctx.host.setTarget(null); } catch (_) {}
  }
  // No specific binding to plot in module view. Pass null so the
  // Plot panel renders its placeholder; renderPlotForCurrent
  // recognizes module mode and tailors the message.
  updatePlotForBinding(ctx, null);
}
