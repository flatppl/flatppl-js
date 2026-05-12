'use strict';

// Surface-syntax variants for FlatPPL (spec §05).
//
// The canonical syntax (.flatppl), the Python-embedding variant
// (.flatppy), and the Julia-embedding variant (.flatppj) share a base
// grammar and differ in additive ways: tilde bindings, exponent `^`,
// logical-operator spelling, boolean-literal spelling, indexing base,
// and semicolon kwargs (FlatPPJ only). Each variant lowers to the same
// FlatPIR — only the parser branches on the variant; analyzer,
// typeinfer, lowering, orchestrator, sampler, and dag are
// variant-agnostic.
//
// Variant config is a plain object so it can be passed by value through
// processSource → tokenize → parse without an import cycle. Fields are
// added as features land; the schema is documented in this file.
//
// Schema:
//   - id:           short identifier ('flatppl' | 'flatppy' | 'flatppj')
//   - booleanLiterals:     ['true','false'] or ['True','False']
//   - logicalSyms:         { and, or, not }  — surface spelling
//   - tildeBindings:       boolean — accept `x ~ M` / `a, b ~ M`
//   - exponentOp:          boolean — accept `^` as a binary operator
//   - membershipOp:        boolean — accept `in` as a comparison operator
//   - chainedComparison:   boolean — lower `a < b < c` to `land(a<b, b<c)`
//   - semiKwargs:          boolean — accept `f(x; a=1)` separator form
//   - indexingLowersTo:    'get' (1-based) or 'get0' (0-based)
//   - reservedAtBinding:   Set<string> — names that can't be a binding LHS
//
// All fields are read by the tokenizer or parser; nothing downstream
// inspects them. Adding a feature means adding a flag here, branching
// at the relevant parser site, and ensuring the lowered AST is the
// same across variants.

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
  reservedAtBinding: new Set(['and', 'or', 'not', 'True', 'False']),
});

const FLATPPY = Object.freeze({
  id: 'flatppy',
  booleanLiterals: ['True', 'False'],
  logicalSyms: { and: 'and', or: 'or', not: 'not' },
  tildeBindings: false,
  exponentOp: false,
  membershipOp: true,
  chainedComparison: true,
  semiKwargs: false,
  indexingLowersTo: 'get0',
  // FlatPPY can't bind `True`/`False` (boolean literals) or `and`/`or`/
  // `not` (logical keywords). Bare `true`/`false` are ordinary names in
  // FlatPPY but it's still good hygiene to keep them off the LHS so
  // round-tripping to FlatPPL doesn't surprise anyone.
  reservedAtBinding: new Set(['and', 'or', 'not', 'True', 'False', 'true', 'false']),
});

const FLATPPJ = Object.freeze({
  id: 'flatppj',
  booleanLiterals: ['true', 'false'],
  logicalSyms: { and: '&&', or: '||', not: '!' },
  tildeBindings: true,
  exponentOp: true,
  membershipOp: true,
  chainedComparison: true,
  semiKwargs: true,
  indexingLowersTo: 'get',
  reservedAtBinding: new Set(['and', 'or', 'not', 'True', 'False']),
});

const BY_ID = Object.freeze({
  flatppl: FLATPPL,
  flatppy: FLATPPY,
  flatppj: FLATPPJ,
});

const EXTENSION_MAP = Object.freeze({
  '.flatppl': FLATPPL,
  '.flatppy': FLATPPY,
  '.flatppj': FLATPPJ,
});

/**
 * Pick a variant from a file path or filename. Falls back to FlatPPL
 * when the extension isn't one of the three known forms (consistent
 * with treating canonical FlatPPL as the default lingua franca).
 * Case-insensitive on the extension. Returns null for an empty path.
 */
function variantForPath(path) {
  if (typeof path !== 'string' || path.length === 0) return null;
  const dot = path.lastIndexOf('.');
  if (dot < 0) return null;
  const ext = path.slice(dot).toLowerCase();
  return EXTENSION_MAP[ext] || null;
}

/**
 * Resolve a variant from explicit / implicit hints. Precedence:
 *   1. opts.variant (object with id field OR a string id)
 *   2. opts.path → variantForPath
 *   3. FLATPPL (canonical default)
 *
 * Unknown ids throw; unknown paths fall through to the default.
 */
function resolveVariant(opts) {
  if (opts && opts.variant != null) {
    if (typeof opts.variant === 'string') {
      const v = BY_ID[opts.variant];
      if (!v) throw new Error(`Unknown variant id: '${opts.variant}'`);
      return v;
    }
    if (opts.variant && opts.variant.id && BY_ID[opts.variant.id]) {
      return BY_ID[opts.variant.id];
    }
    throw new Error('opts.variant must be an id string or a known variant object');
  }
  if (opts && opts.path) {
    const fromPath = variantForPath(opts.path);
    if (fromPath) return fromPath;
  }
  return FLATPPL;
}

module.exports = {
  FLATPPL, FLATPPY, FLATPPJ,
  BY_ID, EXTENSION_MAP,
  variantForPath, resolveVariant,
};
