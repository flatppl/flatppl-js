const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { stripTypeScriptTypes } = require('node:module');
const path = require('node:path');
const vm = require('node:vm');
const engine = require('../../engine/index.ts');
const remote = require('../src/remoteModule.ts');

/** Use the real panel, HTML bootstrap, and dependency walk with inert host IO. */
function panelHost() {
  let dispose, receive;
  const pending = Promise.withResolvers();
  const started = Promise.withResolvers();
  const messages = [];
  const vscode = {
    Uri: { joinPath: (_base, ...parts) => ({ fsPath: path.join(...parts), toString: () => parts.join('/') }) },
    workspace: { fs: { readFile() { started.resolve(); return pending.promise; } } },
  };
  const module = { exports: {} };
  const file = path.join(__dirname, '../src/visualPanel.ts');
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), {
    module, Buffer, setTimeout, clearTimeout,
    require(name) {
      if (name === 'vscode') return vscode;
      if (name === 'fs') return { existsSync: () => false };
      if (name === '../lib/engine.min.js') return engine;
      if (name === '../lib/url-cache.cjs') return { isUrl: () => false };
      if (name === './remoteModule') return remote;
      throw new Error('Unexpected import: ' + name);
    },
  }, { filename: file });
  const panel = new module.exports.FlatPPLPanel({
    webview: {
      cspSource: 'https://webview.invalid', asWebviewUri: (uri) => uri,
      onDidReceiveMessage(fn) { receive = fn; }, postMessage(msg) { messages.push(msg); },
    },
    onDidDispose(fn) { dispose = fn; },
  }, { extensionUri: {} });
  receive({ type: 'webviewReady' });
  return { panel, messages, pending, started, dispose: () => dispose() };
}

for (const [next, fails] of [['embedded', false], ['embedded', true], ['disposed', false]]) {
  test('a pending dependency ' + (fails ? 'failure' : 'success') + ' cannot post after ' + next, async () => {
    const h = panelHost();
    h.panel.updateSource('old = load_module("dep.flatppl")', 'old', {
      scheme: 'file', path: '/old.flatppl', with: (opts) => opts,
    }, false);
    await h.started.promise;
    if (next === 'embedded') h.panel.updateSource('fresh = 2', 'fresh', null, false, { readOnly: true });
    else h.dispose();
    if (fails) h.pending.reject(new Error('dependency unavailable'));
    else h.pending.resolve(Buffer.from('x = 1'));
    await new Promise(setImmediate);
    assert.deepEqual(h.messages.map((msg) => msg.source), next === 'embedded' ? ['fresh = 2'] : []);
  });
}
