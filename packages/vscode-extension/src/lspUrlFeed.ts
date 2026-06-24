'use strict';

// lspUrlFeed.ts — host-agnostic: gather the active model's transitive URL
// load_module closure from the on-disk cache, as the `{ sources: [{uri, text}] }`
// payload for the flatppl-rust LSP's `flatppl/urlSources` notification.
//
// The extension is the sole fetcher+truster of remote modules (spec §04
// #sec:url-cache); this hook only SHARES what is already cached — it never
// fetches and never prompts (it runs on open / save / server-start, not a user
// action). A URL not yet cached is simply skipped: the LSP reports "source not
// available" for it, exactly as before, until the user populates the cache
// (visualize / Download Dependencies), after which the next feed resolves it. So
// there is zero double-download and no second trust prompt — the LSP's
// `Location::join` resolution keys identically to the editor's url-cache.
//
// Mirrors dependencyPrefetch.ts: the SAME engine `resolveBundle` walk (so the
// editor and the LSP agree on "the same file"), every host dependency INJECTED
// (no vscode import), unit-tested headlessly (test/lsp-url-feed.test.js). Local
// deps are READ but NOT collected — the LSP reads local files from its own
// workspace FileSet, so only URL sources are fed; reading them here lets the
// walk recurse THROUGH a local dep to reach URL deps deeper in the graph. An
// unreadable local dep is tolerated (the engine reports it at the load_module
// call, exactly as the visualizer's readSource does).

async function collectUrlSources(opts: any): Promise<any> {
  const { primaryPath, primarySource, resolveBundle, isUrl, readUrlCached,
          readLocal } = opts;
  const sources: any[] = [];
  const seen = new Set();

  const readSource = async (resolved: string) => {
    if (isUrl(resolved)) {
      let text = null;
      try { text = await readUrlCached(resolved); } catch (_e) { text = null; }
      if (text != null && !seen.has(resolved)) {
        seen.add(resolved);
        sources.push({ uri: resolved, text });   // keyed by the resolved URL
      }
      return text;            // null (uncached) ⇒ the walk can't recurse here
    }
    try { return await readLocal(resolved); }
    catch (_e) { return null; }   // unreadable local dep — the engine reports it
  };

  await resolveBundle(primaryPath, primarySource, readSource);
  return { sources };
}

module.exports = { collectUrlSources };
