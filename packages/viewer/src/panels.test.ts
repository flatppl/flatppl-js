// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { PANEL_ORDER, PANEL_DOM, computePanelLayout, clampSplit, dividerPartner } = await import('./panels.ts');

// The viewer's content area is a vertical split of up to three panels —
// graph, plots, math (top to bottom). Which ones show is three independent
// toggles; the layout below is the ONE rule every host and every toggle
// combination goes through, so it is pinned as pure data here.

function flexOf(layout, id) {
  return layout.panels.find((p) => p.id === id).flex;
}
function visibleOf(layout, id) {
  return layout.panels.find((p) => p.id === id).visible;
}
function dividerVisible(layout, after) {
  return layout.dividers.find((d) => d.after === after).visible;
}

test('panel order is graph, plot, math (top to bottom)', () => {
  assert.deepEqual(PANEL_ORDER, ['graph', 'plot', 'math']);
});

test('a lone visible panel fills the area and is marked full', () => {
  const l = computePanelLayout({ graph: true, plot: false, math: false });
  assert.equal(l.visibleCount, 1);
  assert.equal(flexOf(l, 'graph'), '1 1 100%');
  assert.equal(l.panels.find((p) => p.id === 'graph').full, true);
  assert.equal(visibleOf(l, 'plot'), false);
  assert.equal(visibleOf(l, 'math'), false);
  assert.equal(dividerVisible(l, 'graph'), false);
  assert.equal(dividerVisible(l, 'plot'), false);
});

test('graph + plots keep the historical 60/40 split', () => {
  const l = computePanelLayout({ graph: true, plot: true, math: false });
  assert.equal(flexOf(l, 'graph'), '1 1 60%');
  assert.equal(flexOf(l, 'plot'), '1 1 40%');
  assert.equal(l.panels.every((p) => !p.full), true);
  assert.equal(dividerVisible(l, 'graph'), true);
  assert.equal(dividerVisible(l, 'plot'), false);
});

test('all three panels share the area by weight 3 : 2 : 2 with both dividers', () => {
  const l = computePanelLayout({ graph: true, plot: true, math: true });
  assert.equal(l.visibleCount, 3);
  const pct = (id) => Number(flexOf(l, id).replace('1 1 ', '').replace('%', ''));
  assert.ok(Math.abs(pct('graph') - 300 / 7) < 1e-6);
  assert.ok(Math.abs(pct('plot') - 200 / 7) < 1e-6);
  assert.ok(Math.abs(pct('math') - 200 / 7) < 1e-6);
  assert.equal(dividerVisible(l, 'graph'), true);
  assert.equal(dividerVisible(l, 'plot'), true);
});

test('plots + math split 50/50 and only the divider between them shows', () => {
  const l = computePanelLayout({ graph: false, plot: true, math: true });
  assert.equal(flexOf(l, 'plot'), '1 1 50%');
  assert.equal(flexOf(l, 'math'), '1 1 50%');
  assert.equal(dividerVisible(l, 'graph'), false);
  assert.equal(dividerVisible(l, 'plot'), true);
});

test('graph + math (plots hidden): the divider after graph pairs graph with math', () => {
  const l = computePanelLayout({ graph: true, plot: false, math: true });
  assert.equal(flexOf(l, 'graph'), '1 1 60%');
  assert.equal(flexOf(l, 'math'), '1 1 40%');
  assert.equal(dividerVisible(l, 'graph'), true);
  assert.equal(dividerVisible(l, 'plot'), false);
  assert.equal(dividerPartner(l, 'graph'), 'math');
});

test('dividerPartner is the next visible panel, or null when there is none', () => {
  const l = computePanelLayout({ graph: true, plot: true, math: true });
  assert.equal(dividerPartner(l, 'graph'), 'plot');
  assert.equal(dividerPartner(l, 'plot'), 'math');
  assert.equal(dividerPartner(l, 'math'), null);
  const l2 = computePanelLayout({ graph: true, plot: false, math: false });
  assert.equal(dividerPartner(l2, 'graph'), null);
});

test('nothing visible: every panel and divider hidden, count 0', () => {
  const l = computePanelLayout({ graph: false, plot: false, math: false });
  assert.equal(l.visibleCount, 0);
  assert.equal(l.panels.every((p) => !p.visible && p.flex === ''), true);
  assert.equal(l.dividers.every((d) => !d.visible), true);
});

test('clampSplit moves the divider by dy and clamps both sides at the minimum', () => {
  assert.deepEqual(clampSplit(300, 200, 50, 80), { a: 350, b: 150 });
  assert.deepEqual(clampSplit(300, 200, -50, 80), { a: 250, b: 250 });
  // Dragging past the lower panel's minimum stops there.
  assert.deepEqual(clampSplit(300, 200, 500, 80), { a: 420, b: 80 });
  // Dragging past the upper panel's minimum stops there.
  assert.deepEqual(clampSplit(300, 200, -500, 80), { a: 80, b: 420 });
});

test('the first visible panel is marked first (it carries no top border); full implies first', () => {
  const l = computePanelLayout({ graph: true, plot: true, math: true });
  assert.deepEqual(l.panels.map((p) => p.first), [true, false, false]);
  const l2 = computePanelLayout({ graph: false, plot: true, math: true });
  assert.deepEqual(l2.panels.map((p) => p.first), [false, true, false]);
  assert.equal(l2.panels[1].full, false);
  const l3 = computePanelLayout({ graph: false, plot: false, math: true });
  assert.deepEqual(l3.panels.map((p) => [p.first, p.full]), [[false, false], [false, false], [true, true]]);
});

test('PANEL_DOM pins the element ids the layout drives (panel, toggle, divider after)', () => {
  assert.deepEqual(PANEL_DOM.graph, { panel: 'graph-panel', toggle: 'graph-toggle', label: 'Graph', dividerAfter: 'plot-divider' });
  assert.deepEqual(PANEL_DOM.plot,  { panel: 'plot-panel',  toggle: 'plot-toggle',  label: 'Plots', dividerAfter: 'math-divider' });
  assert.deepEqual(PANEL_DOM.math,  { panel: 'math-panel',  toggle: 'math-toggle',  label: 'Math',  dividerAfter: null });
});

// The stylesheet is a string, so the one CSS regression that bit — a
// hidden panel still displayed because an id rule (#plot-panel { display:
// flex }) outranked the class rule hiding it — is pinned here: every panel
// id must have an id-qualified `.hidden` rule that sets display: none.
test('VIEWER_CSS hides each panel with an id-qualified .hidden rule (ids outrank classes)', async () => {
  const { VIEWER_CSS } = await import('./templates.ts');
  for (const id of Object.values(PANEL_DOM).map((d) => d.panel)) {
    const rule = new RegExp('#' + id + '\\.hidden[^{]*\\{[^}]*display:\\s*none');
    assert.ok(rule.test(VIEWER_CSS), `no id-qualified hidden rule for #${id}`);
  }
});
