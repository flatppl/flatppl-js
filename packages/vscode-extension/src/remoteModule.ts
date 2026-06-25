'use strict';

// remoteModule.ts — the vscode-FREE half of VS Code cross-URL drill-down.
//
// A remote (URL) `load_module` (spec §04 #sec:url-cache) can't be opened as a
// file:// document, so it opens as a READ-ONLY virtual document under the
// `flatppl-remote:` scheme: the editor tab shows the URL's basename (like the
// web gallery), the content is served from the on-disk cache by a
// TextDocumentContentProvider (extension.ts), and the original URL rides the
// uri `query` so both the content provider and the engine-path derivation
// recover it exactly.
//
// These helpers operate on a plain { scheme, path, query } shape (which a real
// vscode.Uri satisfies), so they carry NO vscode import and unit-test directly
// (test/remote-module.test.js). The one vscode-coupled piece — building the uri
// via vscode.Uri.parse — stays in visualPanel.ts (smoke-tested).

const REMOTE_SCHEME = 'flatppl-remote';

/** The original http(s) URL behind a `flatppl-remote:` uri — stored verbatim in
 *  the uri `query` (vscode round-trips it through its own percent-encoding). */
function urlFromRemoteUri(uri: any): string {
  return uri && uri.query;
}

/** The logical path the engine/viewer resolves relative `load_module` deps
 *  against: the URL for a remote (virtual) module, the filesystem path
 *  otherwise. A `flatppl-remote:` uri's `.path` drops scheme + authority, so it
 *  must NOT be used for resolution — the URL (from `query`) must. Null uri ⇒
 *  null (a path-less surface clears the viewer's sticky module context). */
function enginePathOf(uri: any): string | null {
  if (!uri) return null;
  return uri.scheme === REMOTE_SCHEME ? urlFromRemoteUri(uri) : uri.path;
}

module.exports = { REMOTE_SCHEME, urlFromRemoteUri, enginePathOf };
