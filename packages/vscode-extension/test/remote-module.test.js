'use strict';

// Unit tests for the remote-module URI helpers — the vscode-FREE half of the
// VS Code cross-URL drill-down. A remote (URL) load_module opens as a read-only
// virtual document under the `flatppl-remote:` scheme: the tab shows the URL's
// basename, the content is served from the on-disk cache, and the original URL
// rides the uri `query` so the content provider and the engine-path derivation
// recover it. These helpers operate on a plain { scheme, path, query } shape
// (which a real vscode.Uri satisfies), so they require directly with no vscode
// stub — same pattern as lspHelpers. (remoteModuleUri, which calls
// vscode.Uri.parse, stays in visualPanel and is smoke-tested.)

const test = require('node:test');
const assert = require('node:assert/strict');
const { REMOTE_SCHEME, urlFromRemoteUri, enginePathOf } = require('../src/remoteModule.js');

const URL = 'https://raw.githubusercontent.com/flatppl/flatppl-examples/refs/heads/main/examples/bayesian_inference_common.flatppl';
const remoteUri = { scheme: REMOTE_SCHEME, path: '/flatppl/.../bayesian_inference_common.flatppl', query: URL };

test('REMOTE_SCHEME is the flatppl-remote virtual scheme', () => {
  assert.equal(REMOTE_SCHEME, 'flatppl-remote');
});

test('urlFromRemoteUri recovers the original URL from the uri query', () => {
  assert.equal(urlFromRemoteUri(remoteUri), URL);
});

test('enginePathOf returns the URL (not the uri path) for a remote uri', () => {
  // The engine/viewer must resolve relative load_module deps against the URL,
  // not the virtual uri's path component (which drops scheme + authority).
  assert.equal(enginePathOf(remoteUri), URL);
});

test('enginePathOf returns the path for an ordinary (file) uri', () => {
  assert.equal(enginePathOf({ scheme: 'file', path: '/home/u/model.flatppl', query: '' }),
    '/home/u/model.flatppl');
});

test('enginePathOf tolerates a null uri', () => {
  assert.equal(enginePathOf(null), null);
});
