// @flatppl/web — async resolver for .flatppl sources.
//
// Step 1.3 (this file): one-call resolver that fetches a single
// `.flatppl` file by path/URL and returns its text. Caches by URL so
// repeat resolutions are free.
//
// Future-shape (lands with `load_module` in step 2): the same
// resolveBundle(rootPath) recursively walks the AST for
// `load_module(...)` calls and returns
//   { primaryPath, primarySource, sources: { [path]: text } }
// with every transitive .flatppl dependency populated. Until then the
// returned object has `sources: {}` and the caller treats the primary
// source as the entire bundle. The shape is forward-compatible — new
// fields land additively, no API break.
//
// Errors propagate as rejected promises with a friendly `.message`
// suitable for surfacing in the UI ("404 Not Found at <path>"). The
// resolver never silently fails — a missing file or network error
// should be visible.
//
// Lives on globalThis as window.FlatPPLWebResolver. Plain global is
// the simplest pattern for an IIFE-free build pipeline (no bundler,
// no module loader); when complexity warrants, this becomes an ES
// module.

'use strict';

(function (globalScope) {
  // URL → fetched text. Conservative cache: never expires within a
  // session (the user reloads the page if they edit a file on disk).
  // The browser's HTTP cache backs this for cross-page reuse.
  var sourceCache = new Map();

  /**
   * Fetch a single .flatppl source by URL.
   * Returns the text on success; throws on network/HTTP error.
   * Cached on first success.
   */
  async function fetchSource(url) {
    if (sourceCache.has(url)) return sourceCache.get(url);

    var response;
    try {
      response = await fetch(url);
    } catch (e) {
      throw new Error('Network error fetching ' + url + ': ' + (e && e.message || e));
    }
    if (!response.ok) {
      throw new Error(response.status + ' ' + response.statusText + ' at ' + url);
    }
    var text = await response.text();
    sourceCache.set(url, text);
    return text;
  }

  /**
   * Resolve a primary FlatPPL source by path. Path is treated as a
   * URL relative to the current document base. Returns a bundle
   * object whose shape is forward-compatible with the load_module
   * implementation (which will populate `sources` with transitive
   * `.flatppl` deps).
   */
  async function resolveBundle(primaryPath) {
    // Ephemeral (unsaved) sources live in an in-memory store the
    // user populates via the gallery's "+ New file" button. Consult
    // it before the network so an ephemeral path resolves without
    // ever 404'ing against the server. The store keeps the path
    // shape compatible with everything else (variantForPath picks
    // the right syntax, the router round-trips it through the
    // URL hash, the editor mounts on it).
    var eph = globalScope.FlatPPLWebEphemeral;
    if (eph && eph.has && eph.has(primaryPath)) {
      return {
        primaryPath: primaryPath,
        primaryUrl: primaryPath,
        primarySource: eph.get(primaryPath),
        sources: Object.create(null),
      };
    }
    var url = new URL(primaryPath, document.baseURI).href;
    var primarySource = await fetchSource(url);
    return {
      primaryPath: primaryPath,
      primaryUrl: url,
      primarySource: primarySource,
      // load_module deps will populate this; empty until step 2.
      sources: Object.create(null),
    };
  }

  /**
   * Drop the cached text for a path/URL so the next resolveBundle
   * re-fetches. Useful when something on the host signals that a
   * file has changed (e.g. a future "reload" button).
   */
  function invalidate(pathOrUrl) {
    var url;
    try { url = new URL(pathOrUrl, document.baseURI).href; }
    catch (_) { url = pathOrUrl; }
    sourceCache.delete(url);
  }

  globalScope.FlatPPLWebResolver = {
    resolveBundle: resolveBundle,
    invalidate: invalidate,
  };
})(typeof window !== 'undefined' ? window : globalThis);
