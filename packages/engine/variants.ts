'use strict';

// Surface syntax for FlatPPL (spec §05).
//
// As of flatppl-design cc81e4b there is ONE surface syntax: canonical
// FlatPPL (`.flatppl`). The former Python/Julia-AST-compatible variants
// FlatPPY/FlatPPJ were removed — host-language embedding is now canonical
// FlatPPL verbatim inside a raw string (`flatppl(r"""…""")` in Python,
// `flatppl"""…"""` in Julia), with editor support via grammar injection
// rather than variant grammars. The reserved-names rule
// (`and/or/not/True/False`) was a FlatPPY concern and was removed from
// §04, so nothing is reserved at binding position in canonical FlatPPL.
//
// This module is retained as a single frozen config object threaded
// through tokenize → parse so those stages keep one explicit place to
// read syntax flags from (and so a future genuinely-different surface
// form, should one ever be added, has an obvious seam). Everything
// downstream of the parser is, and always was, syntax-agnostic.
//
// Schema (all fields read by the tokenizer or parser only):
//   - id:                'flatppl'
//   - booleanLiterals:   ['true','false']
//   - logicalSyms:       { and:'&&', or:'||', not:'!' }
//   - tildeBindings:     accept `x ~ M` / `a, b ~ M`
//   - exponentOp:        accept `^`
//   - membershipOp:      accept `in` as a comparison operator
//   - chainedComparison: lower `a < b < c` to `land(a<b, b<c)`
//   - semiKwargs:        false (no `f(x; a=1)` form in canonical FlatPPL)
//   - indexingLowersTo:  'get' (1-based; the canonical convention)
//   - reservedAtBinding: {in,true,false} — keyword/literal tokens that
//                        are recognized before `Name` and so cannot be
//                        a binding LHS (spec §05 "Note on reserved
//                        words"). The former FlatPPY-only reservations
//                        (and/or/not/True/False) were removed by
//                        flatppl-design cc81e4b.

const FLATPPL = Object.freeze({
  id: 'flatppl',
  booleanLiterals: ['true', 'false'],
  logicalSyms: { and: '&&', or: '||', not: '!' },
  tildeBindings: true,
  exponentOp: true,
  membershipOp: true,
  chainedComparison: true,
  semiKwargs: false,
  indexingLowersTo: 'get',
  reservedAtBinding: new Set(['in', 'true', 'false']),
});

const EXTENSION_MAP = Object.freeze({ '.flatppl': FLATPPL });

/**
 * Always canonical FlatPPL. Retained for call-site compatibility;
 * `null`/empty path returns null (no path), any other path → FlatPPL.
 */
function variantForPath(path: string | null | undefined) {
  if (typeof path !== 'string' || path.length === 0) return null;
  return FLATPPL;
}

/**
 * Resolve the syntax config. There is only canonical FlatPPL now; an
 * explicit request for a removed variant is a clear error rather than
 * a silent fallback, so stale callers/tooling surface the migration.
 */
function resolveVariant(opts: any) {
  if (opts && opts.variant != null) {
    const id = typeof opts.variant === 'string'
      ? opts.variant
      : (opts.variant && opts.variant.id);
    if (id && id !== 'flatppl') {
      throw new Error(
        `Surface variant '${id}' was removed (flatppl-design cc81e4b): ` +
        'there is one canonical FlatPPL syntax. Embed FlatPPL in Python ' +
        'via flatppl(r"""…""") or in Julia via flatppl"""…""".');
    }
  }
  return FLATPPL;
}

module.exports = {
  FLATPPL,
  EXTENSION_MAP,
  variantForPath,
  resolveVariant,
};
