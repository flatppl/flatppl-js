// The webview's source highlighter as shipped: lib/textmate-highlight.js with
// lib/onig.wasm and lib/flatppl.tmLanguage.json (build-vendor.mjs output), and
// the panel HTML that loads it under the webview CSP.
const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, existsSync } = require('node:fs');
const { stripTypeScriptTypes } = require('node:module');
const path = require('node:path');
const vm = require('node:vm');

const LIB = path.join(__dirname, '..', 'lib');
const SHIPPED = ['textmate-highlight.js', 'onig.wasm', 'flatppl.tmLanguage.json'];

function loadHighlighter() {
  for (const name of SHIPPED) {
    assert.ok(existsSync(path.join(LIB, name)), `lib/${name} is missing; run npm run build:vendor`);
  }
  const fetched = [];
  const window = {};
  const sandbox = {
    window, WebAssembly, TextDecoder, TextEncoder, console,
    async fetch(url) {
      fetched.push(url);
      const bytes = readFileSync(path.join(LIB, path.basename(url)));
      return { ok: true, status: 200, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
        text: async () => bytes.toString('utf8') };
    },
  };
  sandbox.globalThis = sandbox;
  vm.runInNewContext(readFileSync(path.join(LIB, 'textmate-highlight.js'), 'utf8'), sandbox);
  return { tm: window.FlatPPLTextmate, fetched };
}

test('the shipped highlighter loads its assets from the given URLs and paints source', async () => {
  const { tm, fetched } = loadHighlighter();
  assert.equal(tm.highlightText('x'), null, 'no runs before init');
  await tm.init({ grammarUrl: 'https://webview.invalid/lib/flatppl.tmLanguage.json', wasmUrl: 'https://webview.invalid/lib/onig.wasm' });
  assert.deepEqual([...fetched], ['https://webview.invalid/lib/onig.wasm', 'https://webview.invalid/lib/flatppl.tmLanguage.json']);
  const runs = Array.from(tm.highlightText('p ~ Normal.(invlogit.(eta), 1.5)  # note'));
  const classed = runs.filter((r) => r.cls).map((r) => r.cls + ':' + r.text.trim());
  assert.deepEqual(classed, ['tok-ident:p', 'tok-op:~', 'tok-dist:Normal', 'tok-op:.', 'tok-func:invlogit', 'tok-op:.',
    'tok-ident:eta', 'tok-number:1.5', 'tok-comment:# note']);
  assert.equal(runs.map((r) => r.text).join(''), 'p ~ Normal.(invlogit.(eta), 1.5)  # note');
});

test('the panel loads the highlighter with its nonce and hands the viewer the asset URIs', () => {
  const vscode = { Uri: { joinPath: (_base, ...parts) => ({ fsPath: path.join(LIB, ...parts.slice(1)), toString: () => 'https://webview.invalid/' + parts.join('/') }) } };
  const module = { exports: {} };
  const file = path.join(__dirname, '../src/visualPanel.ts');
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), {
    module, Buffer, setTimeout, clearTimeout,
    require(name) {
      if (name === 'vscode') return vscode;
      if (name === 'fs') return { existsSync: (p) => SHIPPED.includes(path.basename(p)) };
      if (name === '../lib/engine.min.js' || name === '../lib/url-cache.cjs' || name === './remoteModule') return {};
      throw new Error('Unexpected import: ' + name);
    },
  }, { filename: file });
  const proto = module.exports.FlatPPLPanel.prototype;
  const html = proto._getHtml.call({
    _panel: { webview: { cspSource: 'https://webview.invalid', asWebviewUri: (uri) => uri } },
    _context: { extensionUri: {} },
  });
  const nonce = /'nonce-([^']+)'/.exec(html)[1];
  assert.match(html, new RegExp(`<script nonce="${nonce}" src="https://webview.invalid/lib/textmate-highlight.js"></script>`));
  assert.match(html, /textmateGrammarUrl: "https:\/\/webview.invalid\/lib\/flatppl.tmLanguage.json"/);
  assert.match(html, /textmateWasmUrl: "https:\/\/webview.invalid\/lib\/onig.wasm"/);
  // onig.wasm needs only 'wasm-unsafe-eval'; the CSP grants no 'unsafe-eval'.
  const csp = /Content-Security-Policy"\s+content="([^"]+)"/.exec(html)[1];
  assert.doesNotMatch(csp, /'unsafe-eval'/);
  assert.match(csp, /script-src [^;]*'wasm-unsafe-eval'/);
  assert.match(csp, /connect-src https:\/\/webview.invalid;/);
});
