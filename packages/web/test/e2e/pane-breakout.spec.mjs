// Multi-window tests of the pane breakout: the graph, plots and math panels
// move into their own windows, stay in sync with the playground and come back.

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const MODEL = 'index.html#/files/examples/linear-regression.flatppl?target=sigma';
const PANEL = { graph: 'graph-panel', plot: 'plot-panel', math: 'math-panel' };
const TOGGLE = { graph: 'graph-toggle', plot: 'plot-toggle', math: 'math-toggle' };

async function openModel(page) {
  await page.goto(MODEL);
  await expect(page.locator('#cy canvas').first()).toBeAttached();
  // The plot panel is on by default; wait for its first chart.
  await expect(page.locator('#plot-content canvas').first()).toBeAttached();
}

async function popOut(page, pane) {
  await page.click('#detach-btn');
  const [popup] = await Promise.all([
    page.waitForEvent('popup'),
    page.click(`#detach-menu [data-panel="${pane}"]`),
  ]);
  await popup.waitForLoadState();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  return popup;
}

/** Click Dock in a panel window and wait for the window to close. Docking
 *  can close the window before Playwright's click returns; only that error
 *  is expected. */
async function dockFrom(popup) {
  const closed = popup.waitForEvent('close');
  try {
    await popup.click('#pane-dock');
  } catch (err) {
    if (!popup.isClosed()) throw err;
  }
  await closed;
}

async function expectDetached(page, pane) {
  await expect(page.locator('#' + PANEL[pane])).toHaveClass(/hidden/);
  await expect(page.locator('#' + TOGGLE[pane])).toHaveClass(/detached/);
  await expect(page.locator('#' + TOGGLE[pane])).toHaveAttribute('aria-pressed', 'true');
}

async function expectDocked(page, pane) {
  await expect(page.locator('#' + PANEL[pane])).not.toHaveClass(/hidden/);
  await expect(page.locator('#' + TOGGLE[pane])).not.toHaveClass(/detached/);
}

/** Click a graph node with the mouse, as a user does. */
async function clickNode(page, id) {
  const point = await page.evaluate((nodeId) => {
    const el = document.getElementById('cy');
    const cy = el._cyreg.cy;
    const p = cy.$id(nodeId).renderedPosition();
    const r = el.getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  }, id);
  await page.mouse.click(point.x, point.y);
}

const mainPlotted = (page) => page.evaluate(() => window.FlatPPLWeb.viewer.viewerHandle().plottedBinding());
const panePlotted = (popup) => popup.evaluate(() => window.FlatPPLPane.view.plottedBinding());
const paneHandover = (popup) => popup.evaluate(() => window.FlatPPLPane.view.measureHandoverStatus());

for (const pane of ['graph', 'plot', 'math']) {
  test(`${pane} pops out, renders in its window and docks back by its toggle`, async ({ page }) => {
    await openModel(page);
    const popup = await popOut(page, pane);
    expect(popup.url()).toMatch(new RegExp(`/pane\\.html\\?pane=${pane}#sid=[0-9a-f]{32}$`));
    await expectDetached(page, pane);
    if (pane === 'graph') await expect(popup.locator('#cy canvas').first()).toBeAttached();
    if (pane === 'plot') await expect(popup.locator('#plot-content canvas').first()).toBeAttached();
    if (pane === 'math') await expect(popup.locator('#math-content mtr[data-flatppl-binding]').first()).toBeAttached();
    await expect(popup).toHaveTitle(new RegExp(`linear-regression\\.flatppl`));
    const closed = popup.waitForEvent('close');
    await page.click('#' + TOGGLE[pane]);
    await closed;
    await expectDocked(page, pane);
  });
}

test('the Dock button returns the plots with their samples', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  expect(await paneHandover(popup)).toBe('adopted');
  expect(await panePlotted(popup)).toBe('sigma');
  await dockFrom(popup);
  await expectDocked(page, 'plot');
  await expect.poll(() => page.evaluate(() => window.FlatPPLWeb.viewer.viewerHandle().measureHandoverStatus()))
    .toBe('adopted');
});

test('a tap in the graph window plots and highlights in the main window', async ({ page }) => {
  await openModel(page);
  await page.click('#math-toggle');
  await expect(page.locator('#math-content mtr[data-flatppl-binding]').first()).toBeAttached();
  const popup = await popOut(page, 'graph');
  await expect(popup.locator('#cy canvas').first()).toBeAttached();
  await clickNode(popup, 'sigma_sq');
  await expect.poll(() => mainPlotted(page)).toBe('sigma_sq');
  await expect(page.locator('#math-content mtr.focused[data-flatppl-binding="sigma_sq"]')).toBeAttached();
});

test('a tap in the main graph replots in the plots window', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  await clickNode(page, 'sigma_sq');
  await expect.poll(() => panePlotted(popup)).toBe('sigma_sq');
});

/** Serve config.js with extra host settings, as a deployment would. */
async function configure(context, settings) {
  await context.route('**/config.js', async (route) => {
    const response = await route.fetch();
    const body = (await response.text()) + '\nObject.assign(window.__FLATPPL_CONFIG__, ' + JSON.stringify(settings) + ');\n';
    await route.fulfill({ response, body });
  });
}

test('above the size cap the plots window samples again', async ({ page }) => {
  await configure(page.context(), { handoverCapBytes: 1 });
  await openModel(page);
  const popup = await popOut(page, 'plot');
  expect(await paneHandover(popup)).toBe('rejected:cap');
  await expect(popup.locator('#plot-content canvas').first()).toBeAttached();
});

test('a reload of the panel window keeps the panel detached', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'graph');
  await popup.reload();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  await expect(popup.locator('#cy canvas').first()).toBeAttached();
  await expectDetached(page, 'graph');
});

test('a panel window that reloads slowly stays open and detached', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  // Past the owner's 1.5 s goodbye grace: the window must not be closed as gone.
  await popup.route('**/pane-app.js', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 3000));
    await route.continue();
  });
  await popup.reload();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.', { timeout: 20000 });
  expect(popup.isClosed()).toBe(false);
  await expectDetached(page, 'plot');
  expect(await paneHandover(popup)).toBe('adopted');
});

test('a reload of the main window finds the panel window again', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'math');
  await page.reload();
  await expect(page.locator('#cy canvas').first()).toBeAttached();
  await expectDetached(page, 'math');
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  await clickNode(page, 'sigma_sq');
  await expect(popup.locator('#math-content mtr.focused[data-flatppl-binding="sigma_sq"]')).toBeAttached();
});

test('closing the panel window docks the panel', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  await popup.close();
  await expectDocked(page, 'plot');
  await expect(page.locator('#plot-content canvas').first()).toBeAttached();
});

test('closing the main window tells the panel window', async ({ page, context }) => {
  await openModel(page);
  const popup = await popOut(page, 'graph');
  await page.close();
  await expect(popup.locator('#pane-banner-text')).toHaveText('The main playground window is closed.');
  await expect(popup.locator('#pane-banner-action')).toHaveText('Open playground here');
  await popup.click('#pane-banner-action');
  await expect(popup).toHaveURL(/index\.html#\/files\/examples\/linear-regression\.flatppl/);
  expect(context.pages()).toContain(popup);
});

test('a blocked popup keeps the panel and says why', async ({ page }) => {
  await page.context().addInitScript(() => { window.open = () => null; });
  await openModel(page);
  await page.click('#detach-btn');
  await page.click('#detach-menu [data-panel="plot"]');
  await expect(page.locator('#gallery-toast')).toContainText('The browser blocked the new window');
  await expectDocked(page, 'plot');
});

test('without BroadcastChannel the window fallback carries the session', async ({ page }) => {
  await page.context().addInitScript(() => { delete window.BroadcastChannel; });
  await openModel(page);
  expect(await page.evaluate(() => typeof window.BroadcastChannel)).toBe('undefined');
  const popup = await popOut(page, 'graph');
  await clickNode(popup, 'sigma_sq');
  await expect.poll(() => mainPlotted(page)).toBe('sigma_sq');
  await dockFrom(popup);
  await expectDocked(page, 'graph');
});

test('the Pop out control is hidden in the compact layout', async ({ page }) => {
  await openModel(page);
  await expect(page.locator('#detach-btn')).toBeVisible();
  await page.setViewportSize({ width: 600, height: 900 });
  await expect(page.locator('#detach-btn')).toBeHidden();
});

// The scope is what this feature adds. The viewer's own header and info bar
// fail color-contrast in the main window too (toggle "on" faces, `#info .expr`).
test('the panel window chrome and the Pop out menu pass axe checks', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  const describe = (r) => r.violations.map((v) => [v.id, v.nodes.map((n) => n.target.join(' '))]);
  const paneResults = await new AxeBuilder({ page: popup })
    .include('#pane-toolbar').include('#pane-banner').withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(describe(paneResults)).toEqual([]);
  await page.click('#detach-btn');
  await expect(page.locator('#detach-menu')).toBeVisible();
  const menuResults = await new AxeBuilder({ page })
    .include('#detach-btn').include('#detach-menu').withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(describe(menuResults)).toEqual([]);
});

const plotSettings = (snap) => [snap.plot.sampleCount, snap.plot.rootSeed];

/** Change the plot settings in a mount, as the plot controls do. */
async function setPlotSettings(page, getView, sampleCount, rootSeed) {
  await page.evaluate(([expr, n, seed]) => {
    const view = new Function('return ' + expr)();
    const plot = view.snapshotPanel('plot').plot;
    view.restorePanel('plot', { plot: { ...plot, sampleCount: n, rootSeed: seed } });
  }, [getView, sampleCount, rootSeed]);
}
const MAIN = 'window.FlatPPLWeb.viewer.viewerHandle()';
const PANE = 'window.FlatPPLPane.view';
const snapshotOf = (page, expr, panel) => page.evaluate(([e, p]) => new Function('return ' + e)().snapshotPanel(p), [expr, panel]);

test('plot settings and samples survive a reload of the plots window', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  await setPlotSettings(popup, PANE, 1234, 77);
  await expect(popup.locator('#plot-content canvas').first()).toBeAttached();
  await expect.poll(async () => (await snapshotOf(popup, PANE, 'plot')).plot.sampleCount).toBe(1234);
  await popup.reload();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  expect(plotSettings(await snapshotOf(popup, PANE, 'plot'))).toEqual([1234, 77]);
  expect(await paneHandover(popup)).toBe('adopted');
});

test('the main window keeps its plot settings through a panel reload and Dock', async ({ page }) => {
  await openModel(page);
  await setPlotSettings(page, MAIN, 2000, 5);
  await expect(page.locator('#plot-content canvas').first()).toBeAttached();
  const popup = await popOut(page, 'plot');
  expect(plotSettings(await snapshotOf(popup, PANE, 'plot'))).toEqual([2000, 5]);
  await popup.reload();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  await dockFrom(popup);
  await expectDocked(page, 'plot');
  expect(plotSettings(await snapshotOf(page, MAIN, 'plot'))).toEqual([2000, 5]);
});

test('graph history and viewport survive a reload of the graph window', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'graph');
  await expect(popup.locator('#cy canvas').first()).toBeAttached();
  const point = await popup.evaluate(() => {
    const el = document.getElementById('cy');
    const p = el._cyreg.cy.$id('sigma_sq').renderedPosition();
    const r = el.getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  });
  await popup.mouse.dblclick(point.x, point.y);
  await expect.poll(async () => (await snapshotOf(popup, PANE, 'graph')).graph.history.map((h) => h.targetName))
    .toEqual(['sigma']);
  await popup.evaluate(() => { const cy = document.getElementById('cy')._cyreg.cy; cy.zoom(1.6); cy.pan({ x: 30, y: 20 }); });
  const before = await snapshotOf(popup, PANE, 'graph');
  await popup.reload();
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  const after = await snapshotOf(popup, PANE, 'graph');
  expect(after.graph.history).toEqual(before.graph.history);
  expect(after.graph.viewport).toEqual(before.graph.viewport);
});

test('without BroadcastChannel a reload of the main window finds the panel window again', async ({ page }) => {
  await page.context().addInitScript(() => { delete window.BroadcastChannel; });
  await openModel(page);
  const popup = await popOut(page, 'math');
  await page.reload();
  await expect(page.locator('#cy canvas').first()).toBeAttached();
  await expectDetached(page, 'math');
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.');
  await clickNode(page, 'sigma_sq');
  await expect(popup.locator('#math-content mtr.focused[data-flatppl-binding="sigma_sq"]')).toBeAttached();
});

test('without BroadcastChannel a main window that reloads slowly still finds the panel window', async ({ page }) => {
  await page.context().addInitScript(() => { delete window.BroadcastChannel; });
  await openModel(page);
  const popup = await popOut(page, 'math');
  // Hold the owner script past the panel window's 3 s reconnect wait.
  await page.route('**/pane-owner.js', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await route.continue();
  });
  await page.reload();
  await expect(popup.locator('#pane-banner-text')).toHaveText('The main playground window is closed.', { timeout: 10000 });
  await expect(popup.locator('#pane-status')).toHaveText('Connected to the main window.', { timeout: 20000 });
  await expect(popup.locator('#pane-banner')).toBeHidden();
  await expectDetached(page, 'math');
  await clickNode(page, 'sigma_sq');
  await expect(popup.locator('#math-content mtr.focused[data-flatppl-binding="sigma_sq"]')).toBeAttached();
});

test('the owner repeating the graph window\'s focus keeps the user\'s zoom and pan', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'graph');
  const sid = new URL(popup.url()).hash.match(/sid=([0-9a-f]{32})/)[1];
  await page.evaluate((s) => {
    window.__seen = [];
    const ch = new BroadcastChannel('flatppl-pane:' + s);
    ch.onmessage = (e) => { if (e.data && e.data.from === 'owner') window.__seen.push(e.data); };
    window.__forge = ch;
  }, sid);
  // Any owner message (its liveness ping comes within 5 s) gives the epoch to forge with.
  await expect.poll(() => page.evaluate(() => window.__seen.length), { timeout: 15000 }).toBeGreaterThan(0);
  const last = await page.evaluate(() => window.__seen[window.__seen.length - 1]);
  const focus = (n, target) => page.evaluate(([e, k, t]) => window.__forge.postMessage({ ...e, seq: e.seq + 1000 + k,
    type: 'state.focus', body: { rev: 1000000 + k, target: t, pushHistory: true } }), [last, n, target]);
  await popup.evaluate(() => { const cy = document.getElementById('cy')._cyreg.cy; cy.zoom(1.6); cy.pan({ x: 30, y: 20 }); });
  // The focus the window already shows, as the owner echoes a double-click made here.
  await focus(1, 'sigma');
  await popup.waitForTimeout(500);
  expect((await snapshotOf(popup, PANE, 'graph')).graph.viewport).toEqual({ zoom: 1.6, pan: { x: 30, y: 20 } });
  // A different focus still applies: the path works.
  await focus(2, 'sigma_sq');
  await expect.poll(async () => (await snapshotOf(popup, PANE, 'graph')).graph.history.map((h) => h.targetName))
    .toContain('sigma');
});

test('a panel window ignores owner state older than what it applied', async ({ page }) => {
  await openModel(page);
  const popup = await popOut(page, 'plot');
  const sid = new URL(popup.url()).hash.match(/sid=([0-9a-f]{32})/)[1];
  // Listen on the session's channel in the main page to learn the owner's epoch, seq and rev.
  await page.evaluate((s) => {
    window.__seen = [];
    const ch = new BroadcastChannel('flatppl-pane:' + s);
    ch.onmessage = (e) => { if (e.data && e.data.from === 'owner') window.__seen.push(e.data); };
    window.__forge = ch;
  }, sid);
  await clickNode(page, 'sigma_sq');
  await expect.poll(() => panePlotted(popup)).toBe('sigma_sq');
  const last = await page.evaluate(() => window.__seen.filter((e) => e.type === 'state.select').pop());
  const forge = (seq, rev, binding) => page.evaluate(([e, sq, r, b]) => window.__forge.postMessage({
    ...e, seq: sq, type: 'state.select', body: { rev: r, binding: b } }), [last, seq, rev, binding]);
  await forge(last.seq + 100, last.body.rev - 1, 'sigma');
  await popup.waitForTimeout(500);
  expect(await panePlotted(popup)).toBe('sigma_sq');
  // A newer revision through the same path is applied: the path works.
  await forge(last.seq + 101, last.body.rev + 100, 'alpha');
  await expect.poll(() => panePlotted(popup)).toBe('alpha');
});

test('a window closed before its hello is found by the closed poll', async ({ page }) => {
  await page.context().route('**/pane-app.js', (route) => route.abort());
  await openModel(page);
  await page.click('#detach-btn');
  const [popup] = await Promise.all([page.waitForEvent('popup'), page.click('#detach-menu [data-panel="graph"]')]);
  await expectDetached(page, 'graph');
  await popup.close();
  // The 10 s start timeout would be too late: only the poll docks it this soon.
  await expect(page.locator('#graph-panel')).not.toHaveClass(/hidden/, { timeout: 4000 });
});

test('a window that never starts docks the plots back with their samples', async ({ page }) => {
  await page.context().route('**/pane-app.js', (route) => route.abort());
  await openModel(page);
  await page.click('#detach-btn');
  const [popup] = await Promise.all([page.waitForEvent('popup'), page.click('#detach-menu [data-panel="plot"]')]);
  await expectDetached(page, 'plot');
  await expect(page.locator('#plot-panel')).not.toHaveClass(/hidden/, { timeout: 15000 });
  await expect(page.locator('#gallery-toast')).toContainText('The new window did not start');
  await expect.poll(() => page.evaluate(() => window.FlatPPLWeb.viewer.viewerHandle().measureHandoverStatus()))
    .toBe('adopted');
  await popup.close();
});
