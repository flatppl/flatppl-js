import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

/** Drive the real layout through its controls. Browser QA owns CSS and native input. */
function workspace(compact = false) {
  const elements = new Map();
  const saved = new Map();
  const document = {
    activeElement: null,
    handlers: {},
    addEventListener(name, fn) { this.handlers[name] = fn; },
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return element(); },
  };
  function element(id = '') {
    const el = {
      id, children: [], dataset: {}, style: {}, attributes: {}, handlers: {}, hidden: false, isConnected: true,
      clientWidth: 1200,
      addEventListener(name, fn) { this.handlers[name] = fn; },
      setAttribute(name, value) { this.attributes[name] = value; },
      getAttribute(name) { return this.attributes[name]; },
      removeAttribute(name) { delete this.attributes[name]; },
      insertBefore(child, next) { this.children.splice(this.children.indexOf(next), 0, child); },
      contains(child) { return child === this || this.children.some(el => el.contains(child)); },
      closest() { return this.hidden ? this : null; },
      focus() { document.activeElement = this; },
      click() { this.focus(); this.handlers.click?.(); },
    };
    if (id) elements.set(id, el);
    return el;
  }
  const app = element('app');
  for (const id of ['app-toolbar', 'files-pane', 'source-pane', 'viewer-pane', 'files-backdrop']) app.children.push(element(id));
  const toolbar = elements.get('app-toolbar');
  for (const id of ['toggle-files', 'show-source-btn', 'show-viewer-btn', 'source-toolbar']) toolbar.children.push(element(id));
  elements.get('source-toolbar').children.push(element('show-module-btn'));
  for (const id of ['close-files-btn', 'file-filter']) element(id);
  const media = { matches: compact, addEventListener(_, fn) { this.change = fn; } };
  const window = {
    localStorage: { getItem: key => saved.get(key), setItem: (key, value) => saved.set(key, value) },
    matchMedia: () => media,
    getComputedStyle: () => ({ paddingLeft: '0', paddingRight: '0' }),
    addEventListener() {},
  };
  const file = new URL('../src/layout.ts', import.meta.url);
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), { window, document, console });
  const layout = window.FlatPPLWebLayout.install({ toggleButton: elements.get('toggle-files') });
  return {
    app, layout, document,
    get: id => elements.get(id),
    click: id => elements.get(id).click(),
    compact(value) { media.matches = value; media.change(); },
  };
}

function panes(w, source, viewer) {
  assert.equal(w.get('source-pane').hidden, !source);
  assert.equal(w.get('source-toolbar').hidden, !source);
  assert.equal(w.get('viewer-pane').hidden, !viewer);
  assert.equal(w.get('show-source-btn').getAttribute('aria-pressed'), String(source));
  assert.equal(w.get('show-viewer-btn').getAttribute('aria-pressed'), String(viewer));
}

test('pane toggles add, remove, and switch the sole visible pane', () => {
  const w = workspace();
  panes(w, true, true);
  w.click('show-viewer-btn'); panes(w, true, false);
  w.click('show-source-btn'); panes(w, false, true);
  w.click('show-source-btn'); panes(w, true, true);
  w.click('show-source-btn'); panes(w, false, true);
  w.click('show-viewer-btn'); panes(w, true, false);
  w.click('show-viewer-btn'); panes(w, true, true);
});

test('Files stays independent and content toggles retain the resized split', () => {
  const w = workspace();
  const handle = w.app.children.find(el => el.dataset.resize === 'source-viewer');
  handle.handlers.keydown({ key: 'ArrowRight', preventDefault() {} });
  const resized = w.app.style.gridTemplateColumns;
  for (const [source, viewer] of [[true, true], [false, true], [true, false]]) {
    w.click('toggle-files');
    assert.equal(w.get('files-pane').hidden, true);
    panes(w, source, viewer);
    w.click('toggle-files');
    assert.equal(w.get('files-pane').hidden, false);
    panes(w, source, viewer);
    w.click(source ? 'show-source-btn' : 'show-viewer-btn');
  }
  panes(w, false, true);
  w.click('show-source-btn');
  assert.equal(w.app.style.gridTemplateColumns, resized);
});

test('compact controls preserve pane choices through the drawer, navigation, and resize', () => {
  const w = workspace(true);
  panes(w, true, false);
  w.click('show-source-btn'); panes(w, false, true);
  w.click('show-source-btn'); panes(w, true, true);
  w.click('toggle-files');
  assert.equal(w.get('files-pane').hidden, false);
  panes(w, true, true);
  w.click('close-files-btn');
  assert.equal(w.get('files-pane').hidden, true);
  assert.equal(w.document.activeElement, w.get('toggle-files'));
  w.compact(false); panes(w, true, true);
  w.click('show-source-btn'); panes(w, false, true);
  w.get('viewer-pane').focus();
  w.layout.showPane('source');
  panes(w, true, false);
  assert.equal(w.document.activeElement, w.get('show-source-btn'));
  w.compact(true); panes(w, true, false);
  w.get('show-module-btn').focus();
  w.layout.showPane('viewer');
  panes(w, false, true);
  assert.equal(w.document.activeElement, w.get('show-viewer-btn'));
});
