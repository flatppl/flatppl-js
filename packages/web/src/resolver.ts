// @flatppl/web — async resolver for .flatppl sources.
//
// `resolveBundle(primaryPath)` fetches the primary `.flatppl` file and
// then recursively walks its `load_module("dep")` dependencies (spec §04
// Module composition), pre-fetching every transitive `.flatppl` source
// into a bundle:
//   { primaryPath, primaryUrl, primarySource, sources: { [resolvedPath]: text } }
// The engine pipeline stays synchronous; this resolver does the async I/O
// (the host's job). Dependency paths resolve relative to their importer
// via the engine's `moduleResolve.resolveModulePath`, keyed by the same
// resolved path the engine's bundle compiler expects.
//
// Source resolution is three-tier per file (ephemeral in-memory → user
// store → network fetch), so a user-overridden module never falls through
// to its read-only origin. A missing/broken dependency is left out of the
// bundle (not fatal) — the engine then surfaces the precise "module source
// not found" diagnostic at the `load_module` call. A network error on the
// PRIMARY still rejects (nothing to show).
//
// Lives on globalThis as window.FlatPPLWebResolver. Plain global is
// the simplest pattern for an IIFE-free build pipeline (no bundler,
// no module loader); when complexity warrants, this becomes an ES
// module.

'use strict';

(function (globalScope: any) {
  // URL → fetched text. Conservative cache: never expires within a
  // session (the user reloads the page if they edit a file on disk).
  // The browser's HTTP cache backs this for cross-page reuse.
  const sourceCache = new Map();

  /**
   * Fetch a single .flatppl source by URL.
   * Returns the text on success; throws on network/HTTP error.
   * Cached on first success.
   */
  async function fetchSource(url: any) {
    if (sourceCache.has(url)) return sourceCache.get(url);

    let response;
    try {
      response = await fetch(url);
    } catch (e: any) {
      throw new Error('Network error fetching ' + url + ': ' + (e && e.message || e));
    }
    if (!response.ok) {
      throw new Error(response.status + ' ' + response.statusText + ' at ' + url);
    }
    const text = await response.text();
    sourceCache.set(url, text);
    return text;
  }

  /**
   * Resolve one `.flatppl` source by path through the three-tier order
   * (ephemeral in-memory → user store → network fetch). Returns
   * `{ source, url }` (url is the path itself for in-memory tiers).
   * Throws on a network/HTTP error.
   */
  async function resolveSource(path: any) {
    const eph = globalScope.FlatPPLWebEphemeral;
    if (eph && eph.has && eph.has(path)) return { source: eph.get(path), url: path };
    const userStore = globalScope.FlatPPLWebUserStore;
    if (userStore && userStore.has && userStore.has(path)) {
      return { source: userStore.getSource(path), url: path };
    }
    const url = new URL(path, document.baseURI).href;
    return { source: await fetchSource(url), url: url };
  }

  /**
   * Resolve a primary FlatPPL source by path AND every transitive
   * `.flatppl` dependency it loads via `load_module(...)` (spec §04).
   * Returns `{ primaryPath, primaryUrl, primarySource, sources }` where
   * `sources` maps each dependency's RESOLVED path to its text — the
   * canonical key the engine's bundle compiler expects.
   */
  async function resolveBundle(primaryPath: any) {
    const { source: primarySource, url: primaryUrl } = await resolveSource(primaryPath);
    const FE = globalScope.FlatPPLEngine;
    let sources = Object.create(null);
    if (FE && FE.resolveBundle) {
      // Host primitive only: resolve one source by path through the three-tier
      // order (ephemeral → user store → network fetch); a missing/broken dep
      // returns null and is left out (the engine reports it at the call). The
      // shared engine walk (FE.resolveBundle) owns the moduleDeps +
      // resolveModulePath + dedupe + cycle-safety, identical to the VS Code
      // host's — neither re-implements it.
      const readSource = async (resolved: any) => {
        try { return (await resolveSource(resolved)).source; }
        catch (_) { return null; }
      };
      ({ sources } = await FE.resolveBundle(primaryPath, primarySource, readSource));
    }
    return { primaryPath: primaryPath, primaryUrl: primaryUrl,
      primarySource: primarySource, sources: sources };
  }

  /**
   * Drop the cached text for a path/URL so the next resolveBundle
   * re-fetches. Useful when something on the host signals that a
   * file has changed (e.g. a future "reload" button).
   */
  function invalidate(pathOrUrl: any) {
    let url;
    try { url = new URL(pathOrUrl, document.baseURI).href; }
    catch (_) { url = pathOrUrl; }
    sourceCache.delete(url);
  }

  globalScope.FlatPPLWebResolver = {
    resolveBundle: resolveBundle,
    invalidate: invalidate,
  };
})(typeof window !== 'undefined' ? window : globalThis);
