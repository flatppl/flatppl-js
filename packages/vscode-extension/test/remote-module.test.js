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
const { REMOTE_SCHEME, urlFromRemoteUri, enginePathOf, resolveBaseUri } = require('../src/remoteModule.js');

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

// resolveBaseUri picks the uri to resolve a cross-module back-target against.
// Drilling OUT of a remote (flatppl-remote) virtual doc back to a LOCAL module
// must resolve against the last LOCAL source uri — resolving against the remote
// uri leaks its scheme/authority/query into the file uri and the open fails (the
// back-button regression). A local current uri is used as-is.

const fileUri = { scheme: 'file', path: '/home/u/main.flatppl' };
const remoteUri2 = { scheme: REMOTE_SCHEME, path: '/ex/common.flatppl', query: 'https://h/ex/common.flatppl' };

test('resolveBaseUri uses the local base when the current module is remote', () => {
  assert.equal(resolveBaseUri(remoteUri2, fileUri), fileUri);
});

test('resolveBaseUri keeps the current uri when it is already local', () => {
  const other = { scheme: 'file', path: '/home/u/dep.flatppl' };
  assert.equal(resolveBaseUri(fileUri, other), fileUri);
});

test('resolveBaseUri falls back to the (remote) current uri when no local base is known', () => {
  assert.equal(resolveBaseUri(remoteUri2, undefined), remoteUri2);
});
