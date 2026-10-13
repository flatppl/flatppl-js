import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';
import * as fileTypes from '../src/file-types.mjs';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');

function runSource(name, globals) {
  const file = new URL('../src/' + name + '.ts', import.meta.url);
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), globals, { filename: file.pathname });
}

/** Boot the real gallery over an inert editor and pane, then drive its router. */
async function gallery({ editable = false, store, sidebar = false, manifest = { entries: [] }, localStorage = storage(), loadManifest } = {}) {
  let boot, onChange, editorOptions, beforeUnload, source = '', download, readOnly = true;
  let route = { model: null };
  const requests = [];
  const renders = [];
  const elements = new Map();
  for (const id of ['edit-toggle', 'save-btn', 'retry-save-btn', 'download-btn', 'source-status', 'save-error', 'save-error-message']) {
    elements.set(id, { hidden: false, textContent: '', handlers: {},
      setAttribute() {}, addEventListener(name, fn) { this.handlers[name] = fn; },
    });
  }
  const document = {
    readyState: 'loading', title: '',
    addEventListener(name, fn) { if (name === 'DOMContentLoaded') boot = fn; },
    getElementById(id) { return id === 'source-editor' ? {} : elements.get(id) || null; },
    createElement() { return { click() {} }; },
    body: { appendChild() {}, removeChild() {} },
  };
  if (sidebar) mountSidebar(document, elements);
  const window = {
    localStorage,
    FlatPPLFileTypes: fileTypes,
    __FLATPPL_CONFIG__: { allowEdit: editable },
    FlatPPLWebUserStore: store,
    addEventListener(name, fn) { if (name === 'beforeunload') beforeUnload = fn; },
    FlatPPLViewer: { mount() {} },
    FlatPPLWebManifest: { load: loadManifest || (async () => manifest) },
    FlatPPLWebResolver: { resolveBundle(path, override) {
      if (store?.has(path)) return Promise.resolve({ primarySource: override ?? store.getSource(path), sources: {} });
      const pending = Promise.withResolvers();
      requests.push({ path, ...pending });
      return pending.promise.then(bundle => ({ ...bundle, primarySource: override ?? bundle.primarySource }));
    } },
    FlatPPLWebRouter: {
      parseHash: () => route,
      onChange(fn) { onChange = fn; },
      emitInitial() {},
      navigateTo(state) { route = state; return onChange(state); },
    },
    FlatPPLWebEditor: { mountEditor: (_container, opts) => {
      editorOptions = opts;
      return { setSource(text) { source = text; }, getSource: () => source, setReadOnly(value) { readOnly = value; } };
    } },
    FlatPPLWebSurfaces: { createPaneController: () => ({
      update(text, target, opts) { renders.push({ text, target, path: opts?.path }); },
    }) },
  };
  runSource('app', { window, document, console: { error() {}, warn() {} }, setTimeout, clearTimeout, Blob,
    URL: { createObjectURL(blob) { download = blob; return 'blob:test'; }, revokeObjectURL() {} },
  });
  await boot();
  return {
    requests, renders, document, elements,
    source: () => source,
    readOnly: () => readOnly,
    edit(text) { source = text; editorOptions.onChange(text); },
    toggleEdit() { elements.get('edit-toggle').handlers.click(); },
    save() { editorOptions.onSave(); },
    async download() { await elements.get('download-btn').handlers.click(); return download.text(); },
    warnsOnUnload() { let warned = false; beforeUnload({ preventDefault() { warned = true; } }); return warned; },
    navigate(state) { route = state; return onChange(state); },
  };
}

/** Model the native DOM used by the real sidebar. Browser QA covers layout,
 * keyboard defaults, and touch behavior rather than mocking those here. */
function mountSidebar(document, elements) {
  function element(tag) {
    const el = { tagName: tag, children: [], dataset: {}, className: '', attributes: {}, handlers: {},
      style: { setProperty() {} }, scrollTop: 0, value: '',
      setAttribute(key, value) { this.attributes[key] = value; },
      getAttribute(key) { return key === 'data-key' ? this.dataset.key : this.attributes[key]; },
      addEventListener(key, fn) { this.handlers[key] = fn; },
      appendChild(child) { child.parentElement = this; this.children.push(child); return child; },
      append(...children) { children.forEach(child => this.appendChild(child)); },
      contains(node) { return node === this || this.children.some(child => child.contains(node)); },
      matches(selector) {
        return selector.split(',').some(part => {
          const s = part.trim();
          return s.startsWith('.') ? this.className.split(' ').includes(s.slice(1)) : this.tagName === s;
        });
      },
      closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector); },
      querySelectorAll(selector) {
        return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
      },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
      focus() { document.activeElement = this; },
      click() { this.focus(); return this.handlers.click?.({ currentTarget: this, preventDefault() {} }); },
      scrollIntoView() {},
    };
    el.classList = {
      add(name) { el.className += ' ' + name; },
      remove(name) { el.className = el.className.split(' ').filter(x => x !== name).join(' '); },
      contains(name) { return el.className.split(' ').includes(name); },
    };
    Object.defineProperty(el, 'innerHTML', { set() { this.children = []; } });
    return el;
  }
  document.createElement = element;
  elements.set('file-tree', element('div'));
  elements.set('file-filter', element('input'));
}

const collectionManifest = {
  collections: [{ path: 'examples', title: 'Examples' }, { path: 'posteriordb', title: 'PosteriorDB' }],
  entries: ['examples/model.flatppl', 'posteriordb/group/nested/model.flatppl', 'posteriordb/group/notes.md']
    .map(path => ({ path, title: path.split('/').at(-1) })),
};

function sidebar(g) {
  const tree = g.elements.get('file-tree');
  return {
    folder: key => tree.querySelectorAll('button').find(button => button.dataset.folder === key),
    paths: () => tree.querySelectorAll('a').map(link => link.dataset.path),
    filter(query) { const input = g.elements.get('file-filter'); input.value = query; input.handlers.input(); },
  };
}

test('nested sidebar search keeps folder counts, focus, and normal expansion choices', async () => {
  const ls = storage();
  const g = await gallery({ sidebar: true, manifest: collectionManifest, localStorage: ls });
  await g.navigate({ model: null });
  const tree = sidebar(g);
  assert.deepEqual(tree.paths(), ['examples/model.flatppl']);
  tree.folder('posteriordb').click();
  assert.equal(tree.folder('posteriordb/group').getAttribute('aria-expanded'), 'false');
  assert.equal(g.document.activeElement.dataset.folder, 'posteriordb');
  const normal = ls.getItem('flatppl-web:folder-open');
  tree.filter('model');
  assert.deepEqual(tree.paths(), ['examples/model.flatppl', 'posteriordb/group/nested/model.flatppl']);
  assert.equal(tree.folder('posteriordb').children.at(-1).textContent, '1');
  tree.folder('posteriordb/group').click();
  tree.filter('PosteriorDB');
  assert.deepEqual(tree.paths(), ['posteriordb/group/nested/model.flatppl', 'posteriordb/group/notes.md']);
  const selected = g.navigate({ model: 'posteriordb/group/nested/model.flatppl' });
  g.requests[0].resolve({ primarySource: 'x = 2', sources: {} });
  await selected;
  tree.filter('');
  assert.equal(ls.getItem('flatppl-web:folder-open'), normal);
  assert.equal(tree.folder('posteriordb/group').getAttribute('aria-expanded'), 'false');
  const reloaded = await gallery({ sidebar: true, manifest: collectionManifest, localStorage: ls });
  await reloaded.navigate({ model: null });
  assert.equal(sidebar(reloaded).folder('posteriordb').getAttribute('aria-expanded'), 'true');
  ls.fail = () => true;
  sidebar(reloaded).folder('posteriordb').click();
  assert.equal(sidebar(reloaded).folder('posteriordb').getAttribute('aria-expanded'), 'false');
});

test('nested navigation reveals ancestors and keeps equal-basename drafts distinct', async () => {
  const g = await gallery({ sidebar: true, manifest: collectionManifest, editable: true });
  const tree = sidebar(g);
  const firstPath = collectionManifest.entries[0].path;
  const secondPath = collectionManifest.entries[1].path;
  const first = g.navigate({ model: firstPath });
  g.requests[0].resolve({ primarySource: 'x = 1', sources: {} });
  await first;
  g.toggleEdit();
  g.edit('x = 42');
  tree.filter('examples');
  const second = g.navigate({ model: secondPath });
  g.requests[1].resolve({ primarySource: 'x = 2', sources: {} });
  await second;
  assert.equal(g.source(), 'x = 2');
  assert.equal(g.elements.get('file-filter').value, '');
  for (const key of ['posteriordb', 'posteriordb/group', 'posteriordb/group/nested']) {
    assert.equal(tree.folder(key).getAttribute('aria-expanded'), 'true');
  }
  tree.folder('posteriordb').click();
  g.edit('x = 7');
  assert.equal(tree.folder('posteriordb').getAttribute('aria-expanded'), 'false');
  const back = g.navigate({ model: firstPath });
  g.requests[2].resolve({ primarySource: 'x = 1', sources: {} });
  await back;
  assert.equal(g.source(), 'x = 42');
  const forward = g.navigate({ model: secondPath });
  g.requests[3].resolve({ primarySource: 'x = 2', sources: {} });
  await forward;
  assert.equal(g.source(), 'x = 7');
  assert.equal(tree.folder('posteriordb/group/nested').getAttribute('aria-expanded'), 'true');
});

function manifestLoader(raw) {
  const window = { FlatPPLFileTypes: fileTypes };
  runSource('manifest', { window, fetch: async () => ({ ok: true, text: async () => JSON.stringify(raw) }) });
  return window.FlatPPLWebManifest;
}

test('manifest loading preserves declared metadata and the minimal manifest contract', async () => {
  const declared = await manifestLoader(collectionManifest).load();
  assert.deepEqual(JSON.parse(JSON.stringify(declared.collections)), collectionManifest.collections);
  const g = await gallery({ sidebar: true, manifest: declared });
  await g.navigate({ model: null });
  assert.equal(sidebar(g).folder('posteriordb').getAttribute('aria-expanded'), 'false');
  const minimal = await manifestLoader({ entries: [{ path: 'nested/model.flatppl' }, { path: 'demo/test.flatppl' }] }).load();
  const legacy = await gallery({ sidebar: true, manifest: minimal });
  await legacy.navigate({ model: null });
  assert.deepEqual(sidebar(legacy).paths(), ['nested/model.flatppl']);
  sidebar(legacy).folder('test-cases').click();
  assert.deepEqual(sidebar(legacy).paths(), ['nested/model.flatppl', 'demo/test.flatppl']);
});

test('invalid collection metadata leaves saved files usable', async () => {
  const store = userStore(storage());
  store.save('user/model.flatppl', 'x = 9');
  const invalid = manifestLoader({ ...collectionManifest, entries: [{ path: 'posteriordb-extra/model.flatppl' }] });
  const g = await gallery({ sidebar: true, store, loadManifest: () => invalid.load() });
  await g.navigate({ model: 'user/model.flatppl' });
  assert.equal(g.source(), 'x = 9');
  assert.deepEqual(sidebar(g).paths(), ['user/model.flatppl']);
});

for (const result of ['success', 'failure']) {
  test('a stale gallery load ' + result + ' cannot replace the selected model', async () => {
    const g = await gallery();
    const old = g.navigate({ model: 'old.flatppl' });
    const fresh = g.navigate({ model: 'fresh.flatppl' });
    g.requests[1].resolve({ primarySource: 'fresh = 2', sources: {} });
    await fresh;
    if (result === 'success') g.requests[0].resolve({ primarySource: 'old = 1', sources: {} });
    else g.requests[0].reject(new Error('old request failed'));
    await old;
    assert.equal(g.source(), 'fresh = 2');
    assert.equal(g.renders.at(-1).path, 'fresh.flatppl');
    assert.equal(g.document.title, 'FlatPPL: fresh.flatppl');
  });
}

test('the fallback route supersedes a pending gallery load', async () => {
  const g = await gallery();
  const old = g.navigate({ model: 'old.flatppl' });
  await g.navigate({ model: null });
  const fallback = g.source();
  assert.match(fallback, /elementof/);
  g.requests[0].resolve({ primarySource: 'old = 1', sources: {} });
  await old;
  assert.equal(g.source(), fallback);
  assert.equal(g.document.title, 'FlatPPL');
});

test('edited files survive navigation and still warn in view mode', async () => {
  const g = await gallery({ editable: true });
  const first = g.navigate({ model: 'example.flatppl' });
  g.requests[0].resolve({ primarySource: 'x = 1', sources: {} });
  await first;
  g.toggleEdit();
  g.edit('x = 42');
  const away = g.navigate({ model: 'away.flatppl' });
  g.requests[1].resolve({ primarySource: 'y = 2', sources: {} });
  await away;
  assert.equal(g.warnsOnUnload(), true);
  const back = g.navigate({ model: 'example.flatppl' });
  g.requests[2].resolve({ primarySource: 'x = 1', sources: {} });
  await back;
  g.toggleEdit();
  assert.equal(g.source(), 'x = 42');
  assert.equal(g.warnsOnUnload(), true);
  assert.equal(await g.download(), 'x = 42');
  g.toggleEdit();
  g.edit('x = 1');
  assert.equal(g.warnsOnUnload(), false);
});

test('an edit awaiting debounce cannot render over a different file', async () => {
  const g = await gallery({ editable: true });
  const first = g.navigate({ model: 'old.flatppl' });
  g.requests[0].resolve({ primarySource: 'x = 1', sources: {} });
  await first;
  g.toggleEdit();
  g.edit('x = 42');
  const next = g.navigate({ model: 'new.flatppl' });
  g.requests[1].resolve({ primarySource: 'y = 2', sources: {} });
  await next;
  await new Promise(resolve => setTimeout(resolve, 300));
  assert.equal(g.renders.at(-1).text, 'y = 2');
});

test('Edit cannot unlock a loading or failed source buffer', async () => {
  const g = await gallery({ editable: true });
  const pending = g.navigate({ model: 'slow.flatppl' });
  g.toggleEdit();
  assert.equal(g.readOnly(), true);
  g.requests[0].resolve({ primarySource: 'x = 1', sources: {} });
  await pending;
  assert.equal(g.readOnly(), false);
  const failed = g.navigate({ model: 'missing.flatppl' });
  g.requests[1].reject(new Error('missing'));
  await failed;
  g.toggleEdit();
  g.toggleEdit();
  assert.equal(g.readOnly(), true);
});

test('a failed save stays recoverable across navigation and Retry clears the warning', async () => {
  const ls = storage();
  const store = userStore(ls);
  store.save('user/model.flatppl', 'x = 1');
  const g = await gallery({ editable: true, store });
  await g.navigate({ model: 'user/model.flatppl' });
  g.toggleEdit();
  g.edit('x = 42');
  ls.fail = () => true;
  g.save();
  assert.equal(g.elements.get('save-error').hidden, false);
  assert.equal(g.warnsOnUnload(), true);
  const away = g.navigate({ model: 'away.flatppl' });
  g.requests[0].resolve({ primarySource: 'y = 2', sources: {} });
  await away;
  assert.equal(g.warnsOnUnload(), true);
  await g.navigate({ model: 'user/model.flatppl' });
  assert.equal(g.source(), 'x = 42');
  assert.equal(await g.download(), 'x = 42');
  ls.fail = () => false;
  g.elements.get('retry-save-btn').handlers.click();
  assert.equal(g.elements.get('save-error').hidden, true);
  assert.equal(g.warnsOnUnload(), false);
  assert.equal(userStore(ls).getSource('user/model.flatppl'), 'x = 42');
});

test('returning to a loaded model replaces the intervening loading placeholder', async () => {
  const g = await gallery();
  const initial = g.navigate({ model: 'home.flatppl' });
  g.requests[0].resolve({ primarySource: 'home = 1', sources: {} });
  await initial;
  const away = g.navigate({ model: 'away.flatppl' });
  const back = g.navigate({ model: 'home.flatppl', target: 'home' });
  // A fresh load or retained source both satisfy the public navigation contract.
  g.requests[2]?.resolve({ primarySource: 'home = 1', sources: {} });
  await back;
  g.requests[1].resolve({ primarySource: 'away = 2', sources: {} });
  await away;
  assert.equal(g.source(), 'home = 1');
  assert.equal(g.renders.at(-1).target, 'home');
  const requestCount = g.requests.length;
  await g.navigate({ model: 'home.flatppl', target: null });
  assert.equal(g.requests.length, requestCount);
  assert.equal(g.source(), 'home = 1');
});

/** Model storage failure at one write boundary without touching browser state. */
function storage() {
  const values = new Map();
  return {
    fail: () => false,
    getItem: (key) => values.get(key) ?? null,
    setItem(key, value) {
      if (this.fail(key)) throw new Error('QuotaExceededError');
      values.set(key, value);
    },
    removeItem: (key) => values.delete(key),
  };
}

function userStore(localStorage) {
  const window = { localStorage };
  runSource('user-store', { window, console });
  return window.FlatPPLWebUserStore;
}

for (const write of ['entry', 'index']) {
  test('rename preserves the saved file when the ' + write + ' write fails', () => {
    const ls = storage();
    const store = userStore(ls);
    store.save('user/old.flatppl', 'saved = 42');
    ls.fail = (key) => write === 'entry'
      ? key === 'flatppl-web:user-file:user/new.flatppl'
      : key === 'flatppl-web:user-files';
    assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), false);
    assert.equal(store.getSource('user/old.flatppl'), 'saved = 42');
    assert.equal(userStore(ls).getSource('user/old.flatppl'), 'saved = 42');
  });
}

test('successful rename preserves source and the saved sidebar position', () => {
  const ls = storage();
  const store = userStore(ls);
  store.save('user/old.flatppl', 'saved = 42');
  store.save('user/other.flatppl', 'other = 1');
  assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), true);
  const reloaded = userStore(ls);
  assert.equal(reloaded.getSource('user/new.flatppl'), 'saved = 42');
  assert.equal(reloaded.getSource('user/old.flatppl'), null);
  assert.deepEqual(Array.from(reloaded.list(), (entry) => entry.path), ['user/new.flatppl', 'user/other.flatppl']);
});

test('session-only files still rename when storage is unavailable', () => {
  const ls = storage();
  ls.fail = () => true;
  const store = userStore(ls);
  store.save('user/old.flatppl', 'saved = 42');
  assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), true);
  assert.equal(store.getSource('user/new.flatppl'), 'saved = 42');
  assert.equal(store.getSource('user/old.flatppl'), null);
});

function resolver(localStorage, approve) {
  const requests = [];
  const prompts = [];
  const window = {
    localStorage, FlatPPLEngine: engine,
    confirm: approve == null ? undefined : (message) => { prompts.push(message); return approve; },
    FlatPPLWebEphemeral: {
      has: (path) => path === 'new/main.flatppl',
      get: () => 'remote = load_module("https://remote.invalid/model.flatppl")',
    },
  };
  runSource('resolver', {
    window, document: { baseURI: 'https://gallery.invalid/' }, URL,
    fetch: async (url) => { requests.push(url); return { ok: true, text: async () => 'x = 1' }; },
  });
  return { api: window.FlatPPLWebResolver, requests, prompts };
}

test('a refused remote dependency does not fetch, while same-origin sources load', async () => {
  const r = resolver(storage(), false);
  const bundle = await r.api.resolveBundle('new/main.flatppl');
  assert.deepEqual(r.requests, []);
  assert.deepEqual(Object.keys(bundle.sources), []);
  assert.equal(r.prompts.length, 1);
  assert.equal((await r.api.resolveBundle('demo/local.flatppl')).primarySource, 'x = 1');
  assert.equal((await r.api.resolveBundle('https://gallery.invalid/other.flatppl')).primarySource, 'x = 1');
  assert.equal(r.requests.length, 2);
  assert.equal(r.prompts.length, 1);
});

test('external sources cannot fetch when the host has no approval UI', async () => {
  const r = resolver(storage());
  const bundle = await r.api.resolveBundle('new/main.flatppl');
  assert.deepEqual(Object.keys(bundle.sources), []);
  await assert.rejects(r.api.resolveBundle('//remote.invalid/primary.flatppl'));
  assert.deepEqual(r.requests, []);
});

test('approved URLs retain trust across cache invalidation and page reload', async () => {
  const ls = storage();
  const r = resolver(ls, true);
  const url = 'https://remote.invalid/model.flatppl';
  const bundle = await r.api.resolveBundle('new/main.flatppl');
  assert.equal(bundle.sources[url], 'x = 1');
  r.api.invalidate(url);
  assert.equal((await r.api.resolveBundle(url)).primarySource, 'x = 1');
  assert.equal(r.prompts.length, 1);
  const reloaded = resolver(ls, false);
  assert.equal((await reloaded.api.resolveBundle(url + '#binding')).primarySource, 'x = 1');
  assert.equal(reloaded.prompts.length, 0);
});

test('approved URLs retain session trust when storage writes fail', async () => {
  const ls = storage();
  ls.fail = () => true;
  const r = resolver(ls, true);
  const url = 'https://remote.invalid/model.flatppl';
  await r.api.resolveBundle(url);
  r.api.invalidate(url);
  assert.equal((await r.api.resolveBundle(url)).primarySource, 'x = 1');
  assert.equal(r.prompts.length, 1);
  const reloaded = resolver(ls, false);
  await reloaded.api.resolveBundle('new/main.flatppl');
  assert.equal(reloaded.prompts.length, 1);
  assert.deepEqual(reloaded.requests, []);
});
