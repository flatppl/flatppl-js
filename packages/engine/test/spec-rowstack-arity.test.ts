'use strict';

// =====================================================================
// spec-rowstack-arity.test.ts — pin §07's ONE argument for rowstack /
// colstack
// =====================================================================
//
// Spec §07 "Array and table operations" gives both rows one argument and
// states its domain twice. The table:
//
//   | `rowstack` | `vs` | matrix with input vectors as rows    | vector of
//   |            |      |                                      | equal-length
//   |            |      |                                      | vectors |
//   | `colstack` | `vs` | matrix with input vectors as columns | vector of
//   |            |      |                                      | equal-length
//   |            |      |                                      | vectors |
//
// and the entries:
//
//   "**`rowstack(vs)`** constructs a matrix whose rows are the vectors in
//    `vs`. The argument `vs` is a vector of vectors, all of the same length."
//   "**`colstack(vs)`** constructs a matrix whose columns are the vectors in
//    `vs`. The argument `vs` is a vector of vectors, all of the same length."
//
// So there is no vararg form: `rowstack(a, b)` has no reading. It USED to be
// accepted — `inferRowstack` intercepts these two heads for their static-shape
// read and returned `deferred` on any count but one, so the call never reached
// `inferGenericCall`'s arity check, and the runtime `ARITH_OPS.rowstack` takes
// one JS parameter and ignores the rest. `rowstack([1, 2, 3], [4, 5, 6])`
// therefore answered with an EMPTY 0x0 matrix rather than refusing, and a §03
// matrix consumer went on to multiply by it.
//
// Every other fixed-arity row in the same §07 table already refuses its vararg
// form through `inferGenericCall`; the sibling matrix at the end of this file
// pins that, so a future change that routes one of them past that check the way
// `rowstack` was routed fails here.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function errors(src: string): any[] {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

/** The §07 stack-arity refusals in `src`. */
function arityRefusals(src: string): any[] {
  return errors(src).filter(
    (d: any) => /one argument, `vs`/.test(d.message));
}

/** The fixed value of binding `name`, for a source with no diagnostic. */
function fixedValue(src: string, name: string): any {
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `expected no diagnostic; got: ${errs.map((d: any) => d.message).join('; ')}`);
  return orchestrator.buildDerivations(ctx.bindings).fixedValues.get(name);
}

/**
 * Assert exactly one located §07 arity refusal naming `head`, quoting the
 * entry's own sentence and reporting `got` positional arguments.
 */
function assertRefused(src: string, head: string, got: number) {
  const ds = arityRefusals(src);
  assert.equal(ds.length, 1,
    `expected one arity refusal for ${JSON.stringify(src)}; got ${ds.length}: `
    + `${ds.map((d: any) => d.message).join('; ')}`);
  const d = ds[0];
  assert.match(d.message, new RegExp(`^${head}: spec §07 "Array and table operations"`));
  assert.match(d.message,
    /"The argument `vs` is a vector of vectors, all of the same length"/);
  assert.match(d.message, new RegExp(`got ${got} positional argument\\(s\\)`));
  assert.ok(d.loc && d.loc.start, 'refusal carries a location');
}

// =====================================================================
// 1. The spec spelling stays legal and keeps its value
// =====================================================================

test('rowstack of one vector-of-vectors is accepted, rows in order', () => {
  const v = fixedValue('M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])', 'M');
  assert.deepEqual(v.shape, [2, 3]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6]);
  assert.equal(v.outerRank, undefined, 'the lift produced a true matrix');
});

test('colstack of one vector-of-vectors is accepted, columns in order', () => {
  const v = fixedValue('M = colstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])', 'M');
  assert.deepEqual(v.shape, [3, 2]);
  assert.deepEqual(Array.from(v.data), [1, 4, 2, 5, 3, 6]);
});

test('rowstack of a named vector-of-vectors is accepted', () => {
  const v = fixedValue('vv = [[1.0, 2.0], [3.0, 4.0]]\nM = rowstack(vv)', 'M');
  assert.deepEqual(v.shape, [2, 2]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4]);
});

// =====================================================================
// 2. Every other arity is refused, located, and quotes §07
// =====================================================================

test('rowstack of two vectors is refused', () => {
  assertRefused('M = rowstack([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])', 'rowstack', 2);
});

test('rowstack of three vectors is refused', () => {
  assertRefused('M = rowstack([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])', 'rowstack', 3);
});

test('rowstack of two vectors of unequal length is refused on arity, not length', () => {
  assertRefused('M = rowstack([1.0, 2.0], [3.0, 4.0, 5.0])', 'rowstack', 2);
});

test('rowstack of a vector-of-vectors plus an extra vector is refused', () => {
  // The extra argument was silently DROPPED before: this answered with the
  // 2x2 matrix of the first argument alone.
  assertRefused('M = rowstack([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])', 'rowstack', 2);
});

test('rowstack of no arguments is refused', () => {
  assertRefused('M = rowstack()', 'rowstack', 0);
});

test('colstack of two vectors is refused', () => {
  assertRefused('M = colstack([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])', 'colstack', 2);
});

test('colstack of three vectors is refused', () => {
  assertRefused('M = colstack([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])', 'colstack', 3);
});

test('colstack of no arguments is refused', () => {
  assertRefused('M = colstack()', 'colstack', 0);
});

test('the refusal points at the call, not at the file head', () => {
  const ds = arityRefusals('a = [1.0, 2.0]\nb = [3.0, 4.0]\nM = rowstack(a, b)');
  assert.equal(ds.length, 1);
  // `loc` lines are 0-based here; the call is on source line 3.
  assert.equal(ds[0].loc.start.line, 2, 'located on the rowstack line');
  assert.equal(ds[0].loc.start.col, 4, 'located at the call, not the binding');
});

test('a vararg rowstack inside a larger expression is refused', () => {
  // A consumer that infers its argument twice reports twice — engine-wide and
  // pre-existing (the §07 `sizeof` and `quantile` refusals duplicate the same
  // way under `transpose(...)`), so this asserts at least one.
  const ds = arityRefusals('M = transpose(rowstack([1.0, 2.0], [3.0, 4.0]))');
  assert.ok(ds.length >= 1, 'the inner vararg call is refused');
  assert.equal(ds[0].loc.start.col, 14, 'located at the inner call');
});

// =====================================================================
// 3. The keyword spelling
// =====================================================================
//
// §04 says "All built-in ordinary callables have a defined input order and
// accept both positional and keyword arguments", so `rowstack(vs = …)` is a
// legal §07 spelling. No ordinary §07 row supports it in this engine yet —
// `zeros(dims = 3)`, `reverse(xs = …)`, `tile(A = …, size = …)` all refuse in
// `inferGenericCall` — and these two answered 0x0 instead of refusing. They
// refuse with their siblings now; the shared keyword gap is recorded in
// flatppl-dev/TODO-flatppl-js.md.

test('the keyword spelling is refused rather than answering an empty matrix', () => {
  assertRefused('M = rowstack(vs = [[1.0, 2.0], [3.0, 4.0]])', 'rowstack', 0);
  assertRefused('M = colstack(vs = [[1.0, 2.0], [3.0, 4.0]])', 'colstack', 0);
});

test('a sole positional record still splats into the vs argument (§04)', () => {
  // §04 auto-splatting rewrites this to `rowstack(vs = …)`'s positional
  // equivalent BEFORE the op switch, so it is a one-argument call here.
  const v = fixedValue(
    'r = record(vs = [[1.0, 2.0], [3.0, 4.0]])\nM = rowstack(r)', 'M');
  assert.deepEqual(v.shape, [2, 2]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4]);
});

// =====================================================================
// 4. A scalar first argument keeps the earlier §07 domain refusal
// =====================================================================

test('rowstack of scalars is refused by the §07 collection-domain rule', () => {
  // `_refuseBareCollectionDomainCall` is sited before the op switch because it
  // covers the whole §07 collection-domain table, so it reports first. It
  // cites the same §07 row, so the call is located and quoted either way.
  const errs = errors('M = rowstack(1.0, 2.0)');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message,
    /^rowstack: spec §07 "Array and table operations" gives `rowstack` the domain vector of equal-length vectors/);
  assert.ok(errs[0].loc && errs[0].loc.start, 'refusal carries a location');
});

// =====================================================================
// 5. Sibling matrix — the rest of §07 "Array and table operations"
// =====================================================================
//
// Each fixed-arity sibling with its §07 argument count, and a call carrying one
// argument too many. `cat(x, y, …)` is the table's one genuinely variadic row
// and is excluded. All of these refuse through `inferGenericCall`'s count
// check; pinning them here makes a future interception of one of them (the way
// `rowstack` was intercepted) fail rather than go silent.

const SIBLING_VARARGS: [string, number, string][] = [
  ['tile', 2, 'M = tile([1.0, 2.0], 2, 3)'],
  ['splitblocks', 2, 'M = splitblocks([1.0, 2.0, 3.0, 4.0], 2, 2)'],
  ['joinblocks', 1, 'M = joinblocks([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]])'],
  ['partition', 2, 'M = partition([1.0, 2.0, 3.0, 4.0], 2, 9)'],
  ['reverse', 1, 'M = reverse([1.0, 2.0], [3.0, 4.0])'],
  ['addaxes', 3, 'M = addaxes([1.0, 2.0], 1, 1, 1)'],
  ['blockdiagmat', 1,
    'M = blockdiagmat(rowstack([[1.0]]), rowstack([[2.0]]))'],
  ['bandedmat', 2, 'M = bandedmat([1.0, 2.0], 2, 3)'],
];

for (const [head, want, src] of SIBLING_VARARGS) {
  test(`${head} refuses its vararg form on §07's argument count`, () => {
    const errs = errors(src);
    assert.ok(errs.length >= 1, `${head} accepted ${src}`);
    assert.ok(
      errs.some((d: any) => new RegExp(
        `^${head} expects ${want} positional argument\\(s\\), got ${want + 1}$`,
      ).test(d.message)),
      `${head}: expected an arity refusal naming ${want}; got: `
      + errs.map((d: any) => d.message).join('; '));
  });
}

test('cat stays variadic — §07 gives it `x, y, ...`', () => {
  const v = fixedValue('M = cat([1.0, 2.0], [3.0, 4.0], [5.0])', 'M');
  assert.deepEqual(v.shape, [5]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5]);
});
