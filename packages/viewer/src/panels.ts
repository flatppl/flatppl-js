// @ts-check
// @flatppl/viewer — the vertical panel split (pure layout rule) —
//
// The content area (#main) is a vertical flex column of up to three
// panels — graph, plots, math, top to bottom — each behind its own
// toggle. This module is the ONE rule that turns the three toggle
// states into a layout: which panels show, how the height is shared,
// which drag dividers are live and what pair each divider resizes.
// It is pure data so every combination is testable without a DOM;
// render-frame.ts applies the result to the elements.
//
// Weights, not fixed percentages: graph 3, plots 2, math 2. Two visible
// panels therefore keep the historical 60/40 (graph + plots) split, all
// three share 3:2:2, plots + math are 50/50, and a lone panel fills the
// area (and gets `full`, which drops its top border).
//
// Dividers sit AFTER a panel in the DOM (graph → divider → plots →
// divider → math). A divider is live when its panel is visible and some
// later panel is visible; the pair it resizes is the panel and the NEXT
// visible one, so hiding the plots leaves the graph/math split draggable
// through the first divider.

export type PanelId = 'graph' | 'plot' | 'math';

export const PANEL_ORDER: PanelId[] = ['graph', 'plot', 'math'];

export const PANEL_WEIGHTS: Record<PanelId, number> = { graph: 3, plot: 2, math: 2 };

export interface PanelLayout {
  panels: { id: PanelId; visible: boolean; flex: string; full: boolean }[];
  /** One entry per panel that can have a divider after it (the last
   *  panel never does). */
  dividers: { after: PanelId; visible: boolean }[];
  visibleCount: number;
}

export function computePanelLayout(enabled: Record<PanelId, boolean>): PanelLayout {
  const visibleIds = PANEL_ORDER.filter((id) => !!enabled[id]);
  const total = visibleIds.reduce((s, id) => s + PANEL_WEIGHTS[id], 0);
  const panels = PANEL_ORDER.map((id) => {
    const visible = !!enabled[id];
    return {
      id,
      visible,
      flex: visible ? '1 1 ' + (100 * PANEL_WEIGHTS[id] / total) + '%' : '',
      full: visible && visibleIds.length === 1,
    };
  });
  const dividers = PANEL_ORDER.slice(0, -1).map((id, i) => ({
    after: id,
    visible: !!enabled[id] && PANEL_ORDER.slice(i + 1).some((later) => !!enabled[later]),
  }));
  return { panels, dividers, visibleCount: visibleIds.length };
}

/** The panel a divider resizes against: the next visible panel after
 *  `after`, or null when there is none (the divider is then hidden). */
export function dividerPartner(layout: PanelLayout, after: PanelId): PanelId | null {
  const i = PANEL_ORDER.indexOf(after);
  for (let j = i + 1; j < PANEL_ORDER.length; j++) {
    if (layout.panels[j].visible) return PANEL_ORDER[j];
  }
  return null;
}

/** Redistribute `dy` pixels from the lower panel (b) to the upper one
 *  (a), never shrinking either below `minPx`. The pair's total height is
 *  preserved. */
export function clampSplit(startA: number, startB: number, dy: number, minPx: number): { a: number; b: number } {
  const total = startA + startB;
  let a = startA + dy;
  let b = startB - dy;
  if (a < minPx) { a = minPx; b = total - minPx; }
  if (b < minPx) { b = minPx; a = total - minPx; }
  return { a, b };
}
