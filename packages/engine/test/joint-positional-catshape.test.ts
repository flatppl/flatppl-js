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
