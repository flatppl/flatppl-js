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
  const trustedUrls = new Set<string>();
  const imageDecisions = new Map<string, boolean>();

  function canonicalUrl(url: string): string {
    const parsed = new URL(url, document.baseURI);
    if (!/^https?:$/.test(parsed.protocol)) throw new Error('Browser sources require an http or https URL');
    parsed.hash = '';
    return parsed.href;
  }

  /** Spec §04 Remote file caching: approve an external URL before fetching.
   *  Browser trust markers use localStorage, with a session fallback when
   *  storage is unavailable. Same-origin gallery assets need no prompt. */
  function approveSource(url: string) {
    const parsed = new URL(url, document.baseURI);
    if (!/^https?:$/.test(parsed.protocol)
        || parsed.origin === new URL(document.baseURI).origin) return;
    const key = canonicalUrl(url);
    if (trustedUrls.has(key)) return;
    const storageKey = 'flatppl-web:trusted-url:' + key;
    let trusted = false;
    try { trusted = globalScope.localStorage.getItem(storageKey) === '1'; } catch (_) {}
    if (!trusted) {
      if (typeof globalScope.confirm !== 'function'
          || !globalScope.confirm('FlatPPL: fetch and trust the remote module?\n' + key)) {
        throw new Error('Remote module not trusted: ' + key);
      }
      try { globalScope.localStorage.setItem(storageKey, '1'); } catch (_) {}
    }
    trustedUrls.add(key);
  }

  /** Native images need no CORS permission. Ask before assigning their src,
   *  including the fact that the browser may follow redirects to other hosts. */
  function approveImage(url: string): boolean {
    const key = canonicalUrl(url);
    if (!imageDecisions.has(key)) imageDecisions.set(key, !!globalScope.confirm?.(
      'Load this document image? The server and any redirect destinations will receive a request.\n' + key));
    return imageDecisions.get(key)!;
  }

  /**
   * Fetch a single .flatppl source by URL.
   * Returns the text on success; throws on network/HTTP error.
   * Cached on first success.
   */
  async function fetchBytes(url: string, byteLimit?: number): Promise<Uint8Array> {
    url = canonicalUrl(url);
    const limits = globalScope.FlatPPLEngine?.DEFAULT_LIMITS;
    const maxBytes = byteLimit ?? (/\.zip(?:[?#]|$)/i.test(url) ? (limits?.maxArchiveBytes ?? 64 * 1024 ** 2)
      : (limits?.maxMemberBytes ?? 128 * 1024 ** 2));
    if (sourceCache.has(url)) {
      const bytes = sourceCache.get(url);
      if (bytes.byteLength > maxBytes) throw new Error('File exceeds the browser size limit: ' + url);
      return bytes;
    }
    approveSource(url);

    let response;
    try {
      // Browsers hide redirect destinations in manual mode. Refuse redirects
      // rather than fetch a hop whose scheme and trust cannot be checked.
      response = await fetch(url, { redirect: 'error' });
    } catch (e: any) {
      throw new Error('Network error fetching ' + url + ': ' + (e && e.message || e));
    }
    if (!response.ok) {
      throw new Error(response.status + ' ' + response.statusText + ' at ' + url);
    }
    const reader = response.body?.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    if (reader) {
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > maxBytes) throw new Error('File exceeds the browser size limit: ' + url);
          chunks.push(value);
        }
      } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
      } finally { reader.releaseLock(); }
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    sourceCache.set(url, bytes);
    return bytes;
  }

  /**
   * Resolve one `.flatppl` source by path through the three-tier order
   * (ephemeral in-memory → user store → network fetch). Returns
   * `{ source, url }` (url is the path itself for in-memory tiers).
   * Throws on a network/HTTP error.
   */
  async function readBytes(path: string, drafts: Record<string, string> = {}, byteLimit?: number): Promise<Uint8Array | null> {
    const bundles = globalScope.FlatPPLWebBundles;
    const display = bundles?.displayPath(path) || path;
    const draft = bundles?.draftFor ? await bundles.draftFor(path, drafts) : drafts[display];
    if (draft !== undefined) return new TextEncoder().encode(draft);
    if (bundles?.has(path)) return bundles.read(path);
    const eph = globalScope.FlatPPLWebEphemeral;
    if (eph?.has(path)) return new TextEncoder().encode(eph.get(path));
    const userStore = globalScope.FlatPPLWebUserStore;
    if (userStore?.has(path)) return new TextEncoder().encode(userStore.getSource(path));
    if (path.startsWith('flatppl-zip:')) return null;
    if (/^(?:new\/|user\/)/.test(path)) throw new Error('File is not available: ' + display);
    // An absolute http(s) URL dependency uses browser trust markers and the
    // HTTP/session caches. The shared on-disk cache is a Node-host concern.
    if (/^https?:\/\//i.test(path)) {
      return fetchBytes(path, byteLimit);
    }
    const url = new URL(path, document.baseURI).href;
    return fetchBytes(url, byteLimit);
  }

  /**
   * Resolve a primary FlatPPL source by path AND every transitive
   * `.flatppl` dependency it loads via `load_module(...)` (spec §04).
   * Returns `{ primaryPath, primaryUrl, primarySource, sources }` where
   * `sources` maps each dependency's RESOLVED path to its text — the
   * canonical key the engine's bundle compiler expects.
   */
  async function resolveBundle(primaryPath: string, sourceOverride?: string | null,
      options: { drafts?: Record<string, string>; strict?: boolean } = {}) {
    const drafts = { ...options.drafts };
    if (typeof sourceOverride === 'string') drafts[primaryPath] = sourceOverride;
    const bytes = await readBytes(primaryPath, drafts);
    if (!bytes) throw new Error('File is not available: ' + primaryPath);
    const path = globalScope.FlatPPLWebBundles?.canonical(primaryPath) || primaryPath;
    const FE = globalScope.FlatPPLEngine;
    const limits = globalScope.FlatPPLWebBundles?.limits?.(primaryPath);
    const fileType = globalScope.FlatPPLFileTypes?.typeForPath(primaryPath);
    if (FE?.resolveBundleBytes && (!fileType || fileType === 'flatppl' || /\.zip$/i.test(path))) {
      const graph = await FE.resolveBundleBytes(path, bytes,
        (dep: string) => readBytes(dep, drafts, limits?.maxMemberBytes),
        // The graph also captures whole ZIP containers as files. Individual
        // members remain bounded by the mount and external reads above.
        { strict: !!options.strict, limits: limits && { ...limits, maxMemberBytes: limits.maxArchiveBytes } });
      return { ...graph, primaryUrl: path };
    }
    let source: string, binary = false;
    try {
      source = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      if (source.includes('\0')) throw new Error('binary');
    } catch (_) {
      binary = true;
      source = 'Binary file · ' + bytes.byteLength.toLocaleString() + ' bytes\n\n'
        + Array.from(bytes.subarray(0, 256), byte => byte.toString(16).padStart(2, '0')).join(' ')
        + (bytes.length > 256 ? '\n…' : '');
    }
    return { primaryPath: path, primaryUrl: path, primarySource: source, binary,
      sources: Object.create(null), files: { [path]: bytes }, resolutions: Object.create(null) };
  }

  /**
   * Drop the cached text for a path/URL so the next resolveBundle
   * re-fetches. Useful when something on the host signals that a
   * file has changed (e.g. a future "reload" button).
   */
  function invalidate(pathOrUrl: any) {
    let url;
    try { url = canonicalUrl(pathOrUrl); }
    catch (_) { url = pathOrUrl; }
    sourceCache.delete(url);
  }

  globalScope.FlatPPLWebResolver = {
    resolveBundle: resolveBundle,
    readBytes: readBytes,
    invalidate: invalidate,
    approveImage: approveImage,
  };
})(typeof window !== 'undefined' ? window : globalThis);
