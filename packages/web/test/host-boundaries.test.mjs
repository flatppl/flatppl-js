import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');

function runSource(name, globals) {
  const file = new URL('../src/' + name + '.ts', import.meta.url);
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), globals, { filename: file.pathname });
}

/** Boot the real gallery over an inert editor and pane, then drive its router. */
async function gallery() {
  let boot, onChange, source = '';
  let route = { model: null };
  const requests = [];
  const renders = [];
  const document = {
    readyState: 'loading', title: '',
    addEventListener(name, fn) { if (name === 'DOMContentLoaded') boot = fn; },
    getElementById(id) { return id === 'source-editor' ? {} : null; },
  };
  const window = {
    addEventListener() {},
    FlatPPLViewer: { mount() {} },
    FlatPPLWebManifest: { load: async () => ({ entries: [] }) },
    FlatPPLWebResolver: { resolveBundle(path) {
      const pending = Promise.withResolvers();
      requests.push({ path, ...pending });
      return pending.promise;
    } },
    FlatPPLWebRouter: {
      parseHash: () => route,
      onChange(fn) { onChange = fn; },
      emitInitial() {},
    },
    FlatPPLWebEditor: { mountEditor: () => ({
      setSource(text) { source = text; }, getSource: () => source,
    }) },
    FlatPPLWebSurfaces: { createPaneController: () => ({
      update(text, target, opts) { renders.push({ text, target, path: opts?.path }); },
    }) },
  };
  runSource('app', { window, document, console: { error() {}, warn() {} }, setTimeout, clearTimeout });
  await boot();
  return {
    requests, renders, document,
    source: () => source,
    navigate(state) { route = state; return onChange(state); },
  };
}

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
