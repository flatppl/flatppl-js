'use strict';

// Positional joint of ALL-SCALAR components — shape-contract S1. Spec §06: the
// variate of a positional joint is the `cat` of the component variates, so
// joint(M1, M2) over scalar components is a measure over a VECTOR [x1, x2], not
// an (empty) record. This closes the canonical three-way drift for the common
// all-scalar case: the type (array), the drawn value [x1,x2], and the density
// (which already consumes the cat array) all agree. Keyword joint stays a
// record. (Vector/record/heterogeneous positional joint still need density
// work — tracked in the shape-contract refactor; not claimed here.)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const T = require('../types.ts');

const normalLogpdf = (x: number, mu: number, sd: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mu) / sd) ** 2;
const expLogpdf = (x: number, rate: number) => (x < 0 ? -Infinity : Math.log(rate) - rate * x);

test('positional joint of scalars infers a measure over a vector (not an empty record)', () => {
  const r = processSource('m = joint(Normal(0.0, 1.0), Exponential(1.0))');
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const t = r.bindings.get('m').inferredType;
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array', 'domain is a vector, got ' + T.show(t.domain));
  assert.deepEqual(t.domain.shape, [2]);
  assert.ok(T.equal(t.domain.elem, T.REAL));
});

test('keyword joint still infers a record domain (unchanged)', () => {
  const r = processSource('m = joint(a = Normal(0.0, 1.0), b = Exponential(1.0))');
  const t = r.bindings.get('m').inferredType;
  assert.equal(t.domain.kind, 'record');
  assert.deepEqual(Object.keys(t.domain.fields), ['a', 'b']);
});

test('positional joint: density consumes the cat-vector variate (type≡density)', async () => {
  // logdensityof(joint(N,E), [a,b]) = logphi(a;0,1) + logExp(b;1) — the variate
  // is the vector the type now advertises.
  const { ctx } = ctxFor(
    'm = joint(Normal(0.0, 1.0), Exponential(1.0))\nlp = logdensityof(m, [0.3, 0.7])', 1);
  const lp = await ctx.getMeasure('lp');
  const expected = normalLogpdf(0.3, 0, 1) + expLogpdf(0.7, 1);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    `positional-joint density: got ${lp.samples[0]}, expected ${expected}`);
});

test('positional joint: drawn values are the cat-vector [x1, x2] (sampler≡type)', async () => {
  const { ctx } = ctxFor('m = joint(Normal(0.0, 1.0), Normal(5.0, 1.0))', 4096);
  const m = await ctx.getMeasure('m');
  // The two component draws marginalise to N(0,1) and N(5,1); whatever the
  // internal measure tag, the per-component sample columns are recoverable.
  const comp = m.elems || (m.fields && Object.values(m.fields));
  assert.ok(comp && comp.length === 2, 'two components');
  const mean = (xs: any) => { let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i]; return s / xs.length; };
  assert.ok(Math.abs(mean(comp[0].samples) - 0.0) < 0.06, 'component 0 ~ N(0,1)');
  assert.ok(Math.abs(mean(comp[1].samples) - 5.0) < 0.06, 'component 1 ~ N(5,1)');
});

// --- S1-complete: route the FULL cat rule (not just all-scalar) through
//     shape-contract.catShape. §06: a positional joint's variate is the `cat`
//     of the component variates — all scalars → a vector, all VECTORS → a
//     concatenated vector, all RECORDS → a merged record; mixing shape classes
//     is a static error. Before S1-complete every non-all-scalar positional
//     joint typed as an empty `record({})` (the density already consumes the
//     right flat/record variate — only the advertised type was wrong).

test('positional joint of vectors → measure over the concatenated vector', () => {
  const r = processSource('m = joint(iid(Normal(0.0, 1.0), 3), iid(Normal(0.0, 1.0), 2))');
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const t = r.bindings.get('m').inferredType;
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array', 'domain is a concatenated vector, got ' + T.show(t.domain));
  assert.deepEqual(t.domain.shape, [5]);
  assert.ok(T.equal(t.domain.elem, T.REAL));
});

test('positional joint of vectors: density consumes the flat cat-vector (type≡density)', async () => {
  const { ctx } = ctxFor(
    'm = joint(iid(Normal(0.0, 1.0), 3), iid(Normal(0.0, 1.0), 2))\n'
    + 'lp = logdensityof(m, [0.1, 0.2, 0.3, 0.4, 0.5])', 1);
  const lp = await ctx.getMeasure('lp');
  let expected = 0;
  for (const x of [0.1, 0.2, 0.3, 0.4, 0.5]) expected += normalLogpdf(x, 0, 1);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    `positional-vector-joint density: got ${lp.samples[0]}, expected ${expected}`);
});

test('positional joint of records → measure over the merged record', () => {
  const r = processSource(
    'ra = joint(a = Normal(0.0, 1.0), b = Normal(0.0, 1.0))\n'
    + 'rb = joint(c = Normal(0.0, 1.0), d = Normal(0.0, 1.0))\n'
    + 'm = joint(ra, rb)');
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const t = r.bindings.get('m').inferredType;
  assert.equal(t.domain.kind, 'record', 'domain is a merged record, got ' + T.show(t.domain));
  assert.deepEqual(Object.keys(t.domain.fields), ['a', 'b', 'c', 'd']);
});

test('positional joint mixing shape classes (scalar + vector) is a static error (§06)', () => {
  const r = processSource('m = joint(Normal(0.0, 1.0), iid(Normal(0.0, 1.0), 3))');
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.ok(errs.some((e: any) => /mixing shape classes|all scalar.*all vector|distinct variate/i.test(e.message)),
    'expected a mixed-shape-class static error; got ' + JSON.stringify(errs.map((e: any) => e.message)));
});
