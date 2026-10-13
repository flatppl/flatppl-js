// @flatppl/web — manifest loader.
//
// Fetches `models.json` at startup and exposes the parsed manifest
// to the rest of the app. The manifest tells the gallery what models
// exist (file tree on the left) and which one is the default
// selection when the URL has no `#model=...`.
//
// Format (forward-compatible — new top-level fields land additively):
//
//   {
//     "title": "FlatPPL examples",
//     "collections": [{ "path": "examples", "title": "Examples" }],
//     "entries": [
//       { "path": "demo/foo.flatppl", "title": "Optional display name" },
//       { "path": "demo/bar.flatppl" }
//     ]
//   }
//
// Minimal form (acceptable; titles fall back to file basenames):
//
//   { "entries": [{ "path": "demo/foo.flatppl" }] }
//
// CI workflows that auto-generate `models.json` from
// `find -name '*.flatppl'` produce the minimal form. Hand-curated
// manifests can layer in titles, descriptions, groupings, etc.
//
// Lives on globalThis as window.FlatPPLWebManifest.

'use strict';

(function (globalScope: any) {
  // Default manifest URL. Lives at the gallery's deploy root.
  // Override at boot if a host wants a different filename
  // (e.g. for a multi-gallery deploy that selects between
  // `index/<topic>.json` files).
  const MANIFEST_URL = 'models.json';

  /** Strip everything before the last '/' from a path. */
  function basename(p: any) {
    const i = p.lastIndexOf('/');
    return i < 0 ? p : p.slice(i + 1);
  }

  /**
   * Normalize a raw manifest into the canonical shape:
   *   { title: string, collections?: [{ path, title }], entries: [{ path, title }] }
   * Missing titles fall back to the file basename. Missing entries
   * yields an empty array. An omitted collection list keeps the minimal
   * manifest's flat Examples/Test cases grouping. Declared roots are explicit.
   */
  function normalize(raw: any) {
    const out: { title: string; collections?: { path: string; title: string }[]; entries: any[] } = {
      title:   typeof raw?.title === 'string' ? raw.title : 'FlatPPL',
      entries: [],
    };
    if (raw?.collections !== undefined) {
      out.collections = globalScope.FlatPPLFileTypes.normalizeCollections(raw.collections, 'models.json: collections');
    }
    const roots = out.collections && new Set(out.collections.map(entry => entry.path));
    if (Array.isArray(raw?.entries)) {
      for (let i = 0; i < raw.entries.length; i++) {
        const e = raw.entries[i];
        if (!e || typeof e.path !== 'string') continue;
        if (roots) {
          const parts = e.path.split('/');
          const internal = parts[0] === 'demo' || parts[0] === 'test-cases';
          if ((!internal && !roots.has(parts[0])) || parts.length < 2
              || parts.some((part: string) => !part || part === '.' || part === '..') || e.path.includes('\\')) {
            throw new Error('models.json: entry outside a declared collection: ' + e.path);
          }
        }
        out.entries.push({
          path: e.path,
          title: typeof e.title === 'string' ? e.title : basename(e.path),
          ...(typeof e.description === 'string' ? { description: e.description } : {}),
          ...(typeof e.type === 'string' ? { type: e.type } : {}),
        });
      }
    }
    return out;
  }

  /**
   * Fetch and parse the manifest. Resolves to a normalized manifest
   * object. Rejects on network/HTTP/parse error with a friendly
   * message; the caller decides whether to surface or swallow it
   * (for the gallery, missing manifest is non-fatal — fall back to
   * the inline-source path or a "no models" placeholder).
   */
  async function load(url?: any) {
    url = url || MANIFEST_URL;
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
    let raw;
    try { raw = JSON.parse(text); }
    catch (e: any) { throw new Error('Manifest parse error at ' + url + ': ' + e.message); }
    return normalize(raw);
  }

  globalScope.FlatPPLWebManifest = {
    load: load,
    normalize: normalize,
  };
})(typeof window !== 'undefined' ? window : globalThis);
