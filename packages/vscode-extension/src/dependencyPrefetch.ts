'use strict';

// dependencyPrefetch.ts — host-agnostic orchestration for the VS Code
// "FlatPPL: Download Dependencies" / "Update Dependencies" commands.
//
// Walks the active file's transitive load_module graph via the shared engine
// `resolveBundle` walk (the same one the visualizer uses) and pulls every
// http/https source into the on-disk cache (spec §04 #sec:url-cache), counting
// fresh fetches vs cache hits and collecting per-URL failures. `force` performs
// the spec's explicit update (re-fetch + atomic replace) so edits published at
// a URL are picked up.
//
// Local dependencies are READ (not fetched) so the walk still recurses through
// them to reach URL dependencies deeper in the graph; an unreadable local dep
// is tolerated — the engine reports it at the load_module call, exactly as the
// visualizer's readSource does, so it is NOT a `failed` entry (which is remote
// fetches only). A remote fetch failure IS collected (one entry per URL) and
// the dep is left out, so one bad URL never aborts the whole prefetch.
//
// Every host dependency — the engine walk, the fetch primitive, the trust
// prompt, the local reader — is INJECTED, so there is no `vscode` import and
// the counting / failure / force behavior unit-tests headlessly (see
// test/dependency-prefetch.test.js).

async function prefetchDependencies(opts: any): Promise<any> {
  const { primaryPath, primarySource, resolveBundle, isUrl, fetchToCache,
          approve, readLocal, force } = opts;
  let fetched = 0;
  let cached = 0;
  const failed: any[] = [];

  const readSource = async (resolved: string) => {
    if (isUrl(resolved)) {
      try {
        const r = await fetchToCache(resolved, { approve, force: !!force });
        if (r.fromCache) cached++; else fetched++;
        return r.content.toString('utf8');
      } catch (e: any) {
        failed.push({ url: resolved, error: String((e && e.message) || e) });
        return null;          // left out of the bundle; keep walking the rest
      }
    }
    try { return await readLocal(resolved); }
    catch (_e) { return null; }   // unreadable local dep — the engine reports it
  };

  await resolveBundle(primaryPath, primarySource, readSource);
  return { fetched, cached, failed };
}

module.exports = { prefetchDependencies };
