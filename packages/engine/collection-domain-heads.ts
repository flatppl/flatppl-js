'use strict';

// =====================================================================
// collection-domain-heads.ts — §07 heads whose domain never admits a scalar
// =====================================================================
//
// A `broadcast` applies its head to one ELEMENT at a time. Spec §04
// "Broadcasting":
//
//   > `broadcast(f_or_K, name = array, ...)` or `broadcast(f_or_K, array,
//   > ...)` maps a function or kernel elementwise over arrays (and row-wise
//   > over tables …)
//
// and §04 "Collection arguments" iterates every one of them:
//
//   > All collection arguments (arrays and tables) must have the same number
//   > of axes. Tables count as having one axis (the table's rows) here.
//
// while §04 "Non-collection inputs" pins which arguments escape iteration:
//
//   > Scalar values, functions, kernels, measures and likelihood objects are
//   > allowed as broadcasting inputs, they are simply not iterated over but
//   > held constant while collection arguments are iterated over.
//
// An array argument is a collection argument, so it is iterated, so the head
// receives an ELEMENT. §03 "Arrays" then settles what an element is:
//
//   > Arrays are fixed-size, ordered, n-dimensional collections of scalar
//   > values (real, integer, boolean and complex values) or arrays.
//
// A scalar is a MEMBER of an array, not an array, and §03 defines no rank-0
// array anywhere. So for a head whose §07 Domains cell admits only
// collections, a scalar element is a domain violation — a static error §04
// and §07 settle between them, with no new semantics. §04's dot-sugar closes
// the loophole:
//
//   > `f.(<args>)` lowers to `broadcast(f, <args>)`
//
// so `sum.(v)` and `broadcast(sum, v)` are the same call and both refuse.
//
// The other reading stays legal. §03:
//
//   > Vectors of vectors are not interpreted as matrices implicitly, but can
//   > be turned into matrices explicitly using `rowstack` or `colstack`.
//
// A vector of vectors has ONE axis, so §04's per-axis iteration hands the head
// an inner VECTOR, which IS in `sum`'s domain — `sum.(vv)` is the vector of
// per-inner-vector sums. The refusal therefore keys on a cell that IS a
// scalar, never on the absence of an answer.
//
// The table below names every head across the six §07 tables whose FIRST
// argument's domain is a collection and never a scalar, with its table and
// its Domains cell as §07 writes it (maths rendered as text; a
// multi-argument row carries the whole cell even though only the FIRST
// argument's domain drives the refusal).
//
// Two §07 rows were swept and deliberately REJECTED, both for the same
// reason — the row itself admits a scalar, so the rule does not reach it:
//
//   - `min` / `max`, §07 "Elementary functions", Domains `reals`. These are
//     the two-argument scalar ops, not the `minimum` / `maximum` reductions,
//     so `max.(v, 0.0)` is a well-formed per-element `max` of two scalars.
//   - `cat`, §07 "Array and table operations", Domains "scalars, vectors, or
//     records", whose entry states: "`cat(scalar1, scalar2, ...)` with all
//     scalar arguments produces a vector of those scalars. Equivalent to
//     `vector(scalar1, scalar2, ...)`." A per-element `cat` of two scalars is
//     well-formed where a per-element `sum` of one scalar is not.
//
// Consumed by three enforcement points, which must each keep their own half:
//
//   - `typeinfer._refuseBareCollectionDomainCall` — the located static error
//     over a scalar ARGUMENT, wherever the call is written. §07's domain does
//     not depend on a `broadcast` being present: `sum(2.0)` was 0, and fifteen
//     of these heads declare their slot `any()` in `types.ts` (the type AST
//     cannot say "array of any rank"), so the signature check admitted a
//     scalar and the runtime reduced over nothing.
//   - `typeinfer.inferBroadcast` — the same static error at a broadcast CELL,
//     and the only half that enforces `NESTED_CELL_HEADS` (below), since only
//     there is the head known to be receiving one element. This is the half
//     that was returning wrong numbers.
//   - `dissolver` — refuses to rewrite `broadcast(<head>, X)` to the bare
//     `<head>(X)`, or to fuse it into an `aggregate`. Without that half the
//     wrapper is discarded whenever inference is bypassed or deferred, which
//     is exactly how `cumsum.(vv)` came to emit a whole-collection scan.
//
// The bare half is what reaches a head DEEPER in a compound body.
// `broadcastHeadOpName` resolves only the OUTERMOST op, so `fn(sum(_) + 1.0)`
// resolves to `add` and the broadcast half never sees the `sum` — that is the
// same scope the Rust rule has. `inferBroadcast` infers the body per cell with
// each param bound to its cell type, so the bare gate fires on the inner `sum`
// with a scalar argument. §03's nested reading survives in both halves: an
// ARRAY cell is in `sum`'s domain, so `fn(sum(_) + 1.0).(vv)` answers
// `[7, 16]` over `[[1, 2, 3], [4, 5, 6]]`.
//
// FORWARD HAZARD. §07 carries four further tables with a collection-domain
// first argument that are NOT listed here — Convolution (`conv`,
// `crosscorr`), Binning (`bincounts`), Approximation functions
// (`polynomial`, `bernstein`, `stepwise`) and Array and table generation
// (`array`). They are absent because their dotted spellings refuse for an
// unrelated reason (an arity error, or a signature slot that rejects a
// scalar), not because of the domain reasoning above. `conv` and `crosscorr`
// are additionally in the dissolver's `DISSOLVE_AT_ANY_RANK_OPS`, so a change
// that makes either of them type over a scalar cell re-opens the dotted
// mislowering unless the head joins this table in the same change. Pinned by
// `the_unlowered_collection_heads_refuse_for_another_reason` in
// test/broadcast-collection-domain-heads.test.ts, which flips loudly rather
// than relying on this comment being read.

// head → { section: the §07 table it appears in,
//          domains: its Domains cell as §07 writes it }
const COLLECTION_DOMAIN_HEADS: Map<string, { section: string; domains: string }> = new Map([
  // §07 "Reductions" — 13 rows.
  ['sum', { section: 'Reductions', domains: 'real/complex arrays' }],
  ['mean', { section: 'Reductions', domains: 'real/complex arrays' }],
  ['var', { section: 'Reductions', domains: 'real arrays' }],
  ['std', { section: 'Reductions', domains: 'real arrays' }],
  ['prod', { section: 'Reductions', domains: 'real/complex arrays' }],
  ['maximum', { section: 'Reductions', domains: 'real arrays' }],
  ['minimum', { section: 'Reductions', domains: 'real arrays' }],
  ['median', { section: 'Reductions', domains: 'real arrays' }],
  ['quantile', { section: 'Reductions', domains: 'real arrays, `interval(0, 1)`' }],
  ['lengthof', { section: 'Reductions', domains: 'vectors, tables' }],
  ['sizeof', { section: 'Reductions', domains: 'vectors, arrays' }],
  ['indicesof', { section: 'Reductions', domains: 'vectors, arrays, tables' }],
  ['indicesof0', { section: 'Reductions', domains: 'vectors, arrays, tables' }],
  // §07 "Boolean reductions" — a bold table label inside §07 "Logic and
  // conditionals", not a `###` heading, but locatable by that name.
  ['lany', { section: 'Boolean reductions', domains: 'boolean arrays' }],
  ['lall', { section: 'Boolean reductions', domains: 'boolean arrays' }],
  // §07 "Norms and normalization" — 8 rows.
  ['l1norm', { section: 'Norms and normalization', domains: 'real/complex vectors' }],
  ['l2norm', { section: 'Norms and normalization', domains: 'real/complex vectors' }],
  ['linfnorm', { section: 'Norms and normalization', domains: 'real/complex vectors' }],
  ['l1unit', { section: 'Norms and normalization', domains: 'real/complex vectors' }],
  ['l2unit', { section: 'Norms and normalization', domains: 'real/complex vectors' }],
  ['logsumexp', { section: 'Norms and normalization', domains: 'real vectors' }],
  ['softmax', { section: 'Norms and normalization', domains: 'real vectors' }],
  ['logsoftmax', { section: 'Norms and normalization', domains: 'real vectors' }],
  // §07 "Cumulative operations" — 4 rows.
  ['cumsum', { section: 'Cumulative operations', domains: 'vectors' }],
  ['cumprod', { section: 'Cumulative operations', domains: 'vectors' }],
  ['cummax', { section: 'Cumulative operations', domains: 'real vectors' }],
  ['cummin', { section: 'Cumulative operations', domains: 'real vectors' }],
  // §07 "Linear algebra" — all 16 rows are collection-only in their first
  // argument.
  ['transpose', { section: 'Linear algebra', domains: 'vectors, matrices' }],
  ['adjoint', { section: 'Linear algebra', domains: 'vectors, matrices' }],
  ['det', { section: 'Linear algebra', domains: 'square matrices' }],
  ['logabsdet', { section: 'Linear algebra', domains: 'square matrices' }],
  ['inv', { section: 'Linear algebra', domains: 'square matrices' }],
  ['trace', { section: 'Linear algebra', domains: 'square matrices' }],
  ['linsolve', { section: 'Linear algebra', domains: 'square `A`, vector `b`' }],
  ['qr', { section: 'Linear algebra', domains: 'm x n, m >= n matrices' }],
  ['lower_cholesky', { section: 'Linear algebra', domains: 'positive definite `A`' }],
  ['row_gram', { section: 'Linear algebra', domains: 'matrices' }],
  ['col_gram', { section: 'Linear algebra', domains: 'matrices' }],
  ['self_outer', { section: 'Linear algebra', domains: 'vectors' }],
  ['cross', {
    section: 'Linear algebra',
    domains: 'real or complex vectors with `lengthof(a) == lengthof(b) == 3`',
  }],
  ['diagmat', { section: 'Linear algebra', domains: 'vectors' }],
  ['diag', { section: 'Linear algebra', domains: 'matrices, integer' }],
  ['quadform', { section: 'Linear algebra', domains: 'square `A`, vector `x`' }],
  // §07 "Array and table operations" — 10 of its 11 rows. `cat` is the row
  // swept and rejected (see the carve-out note above).
  ['rowstack', {
    section: 'Array and table operations', domains: 'vector of equal-length vectors',
  }],
  ['colstack', {
    section: 'Array and table operations', domains: 'vector of equal-length vectors',
  }],
  ['tile', {
    section: 'Array and table operations', domains: 'array, integer or integer vector',
  }],
  ['splitblocks', {
    section: 'Array and table operations', domains: 'array, integer or integer vector',
  }],
  ['joinblocks', {
    section: 'Array and table operations', domains: 'array of equal-shaped arrays',
  }],
  ['partition', {
    section: 'Array and table operations',
    domains: 'vector, positive integer or integer vector',
  }],
  ['reverse', { section: 'Array and table operations', domains: 'vectors, tables' }],
  ['addaxes', {
    section: 'Array and table operations',
    domains: 'array, non-negative integer, non-negative integer',
  }],
  ['blockdiagmat', {
    section: 'Array and table operations', domains: 'vector of matrices',
  }],
  ['bandedmat', {
    section: 'Array and table operations', domains: 'vector, positive integer',
  }],
]);

// The four heads whose §07 Domains cell demands a collection OF COLLECTIONS,
// so a FLAT cell — an array of scalars — is out of domain too, not only a
// scalar one:
//
//   `rowstack` / `colstack`  "vector of equal-length vectors"
//   `joinblocks`             "array of equal-shaped arrays"
//   `blockdiagmat`           "vector of matrices"
//
// §03 makes this a real distinction rather than a pedantic one — "Vectors of
// vectors are not interpreted as matrices implicitly" — and the engine already
// enforces it elsewhere (`valueLib.requireMatrix`, and typeinfer's §03
// vec-of-vec diagnostic). Without this, `rowstack.(vv)` handed `rowstack` a
// `[3]` vector of SCALARS per cell and answered with two empty 0x0 matrices.
//
// Not a general "cell out of domain" check: only the FIRST argument's domain
// drives the refusal, so a later argument out of domain — `linsolve.(MM, MM)`,
// where §07 gives `b` "vector" and it receives a matrix — is still not reached.
// Recorded in flatppl-dev/TODO-flatppl-js.md.
const NESTED_CELL_HEADS: Set<string> = new Set([
  'rowstack', 'colstack', 'joinblocks', 'blockdiagmat',
]);

// The ten built-ins §04 "Multi-axis aggregation" admits:
//
//   > The eligible built-ins are `sum`, `prod`, `mean`, `var`, `std`,
//   > `maximum`, `minimum`, `median`, `lany` and `lall`.
//
// Only these may be offered as an `aggregate` remedy. §07 "Cumulative
// operations" says the four scans "are not eligible reductions for multi-axis
// aggregation", so offering `aggregate(cumsum, …)` — or `aggregate(l2norm,
// …)`, which is in no §04 list at all — would send the reader into a second
// refusal.
const AGGREGATE_ELIGIBLE_HEADS: Set<string> = new Set([
  'sum', 'prod', 'mean', 'var', 'std', 'maximum', 'minimum', 'median',
  'lany', 'lall',
]);

// The bare-call spelling to offer as the remedy. Every head here takes its
// collection as the first argument; `quantile` is the one two-argument row in
// the table, and `quantile(v)` would be an arity error rather than a fix.
function bareFormFor(head: string): string {
  if (head === 'quantile') return 'quantile(xs, p)';
  return head + '(xs)';
}

// Is this head's domain a collection that never admits a scalar?
function isCollectionDomainHead(name: any): boolean {
  return typeof name === 'string' && COLLECTION_DOMAIN_HEADS.has(name);
}

function domainCellFor(name: string): { section: string; domains: string } | null {
  return COLLECTION_DOMAIN_HEADS.get(name) || null;
}

// Does this head's §07 cell demand a collection of collections, so that a flat
// array cell is out of domain as well as a scalar one?
function needsNestedCell(name: any): boolean {
  return typeof name === 'string' && NESTED_CELL_HEADS.has(name);
}

module.exports = {
  COLLECTION_DOMAIN_HEADS,
  AGGREGATE_ELIGIBLE_HEADS,
  NESTED_CELL_HEADS,
  isCollectionDomainHead,
  domainCellFor,
  bareFormFor,
  needsNestedCell,
};
