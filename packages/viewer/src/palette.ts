// @ts-check
// @flatppl/viewer — palette-driven node colour helpers (Phase 4c).
//
// resolveNodeColor / colorForBinding look up colours from the
// per-mount ctx.PALETTE / ctx.PHASE_COLORS / ctx.TYPE_STYLE objects
// (built in mount's prologue). No external function dependencies.

/** @typedef {import('./types').Ctx} Ctx */

/**
 * Single source of truth for "what colour does this node get?".
 * Used by the DAG renderer, the plot-view colorForBinding lookup,
 * and the reification-bubble fill so all three views stay coherent.
 *
 * Decision tree:
 *   kind === 'kernel'         → kernelof teal (overrides type)
 *   kind === 'measure'        → lawof blue   (overrides type)
 *   type ∈ {'draw', 'call'}   → ctx.PHASE_COLORS[phase]   (value node)
 *   else                      → ctx.TYPE_STYLE[type].color (structural)
 *
 * Inside a reification bubble, node.phase has already been
 * overridden to the scope-local phase by dag.js's
 * applyScopeLocalPhases — so the same theta1 reads stochastic in
 * the main view and parameterized inside a kernel bubble.
 *
 * @param {Ctx} ctx
 * @param {{ kind?: string; type?: string; phase?: 'stochastic'|'parameterized'|'fixed' }} node
 * @returns {string}
 */
export function resolveNodeColor(ctx: any, node: any) {
  if (node.kind === 'kernel')  return ctx.TYPE_STYLE.kernelof.color;
  if (node.kind === 'measure') return ctx.TYPE_STYLE.lawof.color;
  var ts = (node.type && ctx.TYPE_STYLE[node.type]) || ctx.TYPE_STYLE.unknown;
  if (node.type === 'draw' || node.type === 'call') {
    return (node.phase && ctx.PHASE_COLORS[node.phase]) || ts.color;
  }
  return ts.color;
}

/**
 * @param {Ctx} ctx
 * @param {string} bindingName
 * @returns {string}
 */
export function colorForBinding(ctx: any, bindingName: any) {
  if (ctx.currentState && ctx.currentState.data && ctx.currentState.data.nodes) {
    var nodes = ctx.currentState.data.nodes;
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].id === bindingName) return resolveNodeColor(ctx, nodes[i]);
    }
  }
  // Fallback when the plot is updating ahead of the DAG (rare, but
  // possible during config-update reflows). currentBindings has
  // .type but not .kind/.phase, so resolveNodeColor naturally
  // degrades to the type colour.
  var binding = ctx.currentBindings && ctx.currentBindings.get(bindingName);
  return resolveNodeColor(ctx, { type: (binding && binding.type) || 'draw' });
}
