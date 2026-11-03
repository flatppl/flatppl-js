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
  const closed = popup.waitForEvent('close');
  await popup.click('#pane-dock');
  await closed;
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

test('above the size cap the plots window samples again', async ({ page }) => {
  await page.context().addInitScript(() => { window.__FLATPPL_TEST_HANDOVER_CAP__ = 1; });
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
  const closed = popup.waitForEvent('close');
  await popup.click('#pane-dock');
  await closed;
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
