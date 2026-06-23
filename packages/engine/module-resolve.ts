'use strict';

// Spec §04 (Module composition — "Path resolution"): the pure path math
// behind `load_module(...)`. Relative module paths resolve against the
// DIRECTORY of the file containing the call; `/` is the mandatory
// separator on every platform; `..` traversal is allowed; absolute paths
// (leading `/`) are permitted (though discouraged for relocatable repos).
//
// This is intentionally a dependency-free leaf: the engine's bundle
// compiler keys its module registry by these resolved paths, and the host
// resolvers (web `fetch`, VS Code `workspace.fs`) must produce the SAME
// keys when they pre-fetch sources into `bundle.sources`. One owner for
// the rule keeps the engine and hosts from disagreeing about "the same
// file".

// Segments of the importer's *directory* — everything up to (not
// including) the final `/`. A bare filename ("model.flatppl") has an
// empty directory.
function _dirSegments(p: string): string[] {
  if (!p) return [];
  const idx = p.lastIndexOf('/');
  if (idx < 0) return [];
  return p.slice(0, idx).split('/').filter((s) => s.length > 0);
}

// Collapse `.` / `..` / empty segments. For a relative path, a leading
// `..` that cannot pop anything is PRESERVED (the path legitimately
// escapes upward); for an absolute path, `..` at the root is dropped
// (you cannot go above `/`).
function _normalize(segments: string[], isAbsolute: boolean): string[] {
  const out: string[] = [];
  for (const seg of segments) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..') {
      if (out.length > 0 && out[out.length - 1] !== '..') {
        out.pop();
      } else if (!isAbsolute) {
        out.push('..');
      }
      continue;
    }
    out.push(seg);
  }
  return out;
}

/**
 * Resolve a `load_module(relPath)` reference against the path of the file
 * that contains the call.
 *
 * @param importerPath the resolved path of the importing module (may be
 *        null/empty for a root-level entry — then `relPath` resolves
 *        against the root).
 * @param relPath the path literal as written in the `load_module(...)`
 *        call. A leading `/` marks it absolute; otherwise it is relative
 *        to the importer's directory.
 * @returns the normalised path, the canonical key into `bundle.sources`.
 */
function resolveModulePath(importerPath: string | null, relPath: string): string {
  relPath = String(relPath == null ? '' : relPath);
  if (relPath.startsWith('/')) {
    return '/' + _normalize(relPath.split('/'), true).join('/');
  }
  // An absolute importer (a VS Code URI path, `/proj/model.flatppl`)
  // resolves its siblings as absolute too — `_dirSegments` drops the
  // leading slash, so it must be restored.
  const importerAbsolute = !!(importerPath && String(importerPath).startsWith('/'));
  const base = _dirSegments(importerPath || '');
  const segs = _normalize(base.concat(relPath.split('/')), importerAbsolute);
  return (importerAbsolute ? '/' : '') + segs.join('/');
}

module.exports = { resolveModulePath };
