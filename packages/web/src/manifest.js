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

(function (globalScope) {
  // Default manifest URL. Lives at the gallery's deploy root.
  // Override at boot if a host wants a different filename
  // (e.g. for a multi-gallery deploy that selects between
  // `index/<topic>.json` files).
  var MANIFEST_URL = 'models.json';

  /** Strip everything before the last '/' from a path. */
  function basename(p) {
    var i = p.lastIndexOf('/');
    return i < 0 ? p : p.slice(i + 1);
  }

  /**
   * Normalize a raw manifest into the canonical shape:
   *   { title: string, entries: [{ path, title }] }
   * Missing titles fall back to the file basename. Missing entries
   * yields an empty array. Unknown top-level fields are preserved
   * for forward compatibility.
   */
  function normalize(raw) {
    var out = {
      title:   typeof raw.title === 'string' ? raw.title : 'FlatPPL',
      entries: [],
    };
    if (Array.isArray(raw.entries)) {
      for (var i = 0; i < raw.entries.length; i++) {
        var e = raw.entries[i];
        if (!e || typeof e.path !== 'string') continue;
        out.entries.push({
          path: e.path,
          title: typeof e.title === 'string' ? e.title : basename(e.path),
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
  async function load(url) {
    url = url || MANIFEST_URL;
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
    var raw;
    try { raw = JSON.parse(text); }
    catch (e) { throw new Error('Manifest parse error at ' + url + ': ' + e.message); }
    return normalize(raw);
  }

  globalScope.FlatPPLWebManifest = {
    load: load,
    normalize: normalize,
  };
})(typeof window !== 'undefined' ? window : globalThis);
