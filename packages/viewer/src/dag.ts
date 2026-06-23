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
import { errorsForBinding } from './render-frame.js';
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
    if (oe && (oe.ctrlKey || oe.metaKey)) {
      const line = evt.target.data('line');
      if (line >= 0) {
        if (ctx.host.revealSourceLine) ctx.host.revealSourceLine(line);
      }
      return;
    }
    const d = evt.target.data();
    showNodeInfo(ctx, d);
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
    const nodeId = evt.target.data('id');
    // Don't drill into synthetic nodes (placeholder/hole inputs).
    if (nodeId.indexOf(':') !== -1) return;
    // A user `load_module` node: double-click navigates INTO the loaded
    // module's file (spec §04 Module composition) rather than drilling a
    // sub-DAG — a module boundary has none in the primary graph. The
    // resolved path is the bundle / router key recorded at lowering.
    const modEntry = _loadModuleEntry(ctx, nodeId);
    if (modEntry && ctx.host && typeof ctx.host.openModule === 'function') {
      ctx.host.openModule(modEntry.path);
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

  for (let i = 0; i < data.nodes.length; i++) {
    const node = data.nodes[i];
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
    const displayLabel = node.label === '' ? '' : (node.label || node.id);
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
 */
export function focusNode(ctx: Ctx, targetName: any, pushHistory: any) {
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
  const dagData = FlatPPLEngine.computeSubDAG(ctx.currentBindings, targetName);
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

  ctx.currentState = { data: dagData, targetName: targetName };
  renderDAG(ctx, dagData);
  updateBackBtn(ctx);
  updatePlotForBinding(ctx, targetName);
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
  const dagData = FlatPPLEngine.computeFullDAG(ctx.currentBindings);
  if (!dagData || dagData.nodes.length === 0) return;

  if (pushHistory && ctx.currentState && ctx.currentState.targetName !== ctx.MODULE_TARGET) {
    ctx.history.push(ctx.currentState);
    if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
  }

  ctx.currentState = { data: dagData, targetName: ctx.MODULE_TARGET };
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
