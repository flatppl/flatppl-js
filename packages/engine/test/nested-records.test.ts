'use strict';

// Nested records — spec §03/§06 (commit "Allow nested records and nested
// tuples"). A record field may itself be a record; field access chains
// (`r.a.b`). A measure over a nested record (built with the keyword
// `joint(name = M, ...)` form, where a record-valued component nests UNDER
// its name — §06: "the name adds a level, it does not merge the inner
// fields") samples and scores by recursion through the record structure.
//
// The engine type system, materialiser, and density walker were already
// recursive by kind, so these are regression guards pinning behaviour the
// commit newly blesses end to end (value, sampling, and density).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const T = require('../types.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function valueOf(name: string, src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    throw new Error('source had errors: ' + errs.map((e: any) => e.message).join(' | '));
  }
  const ds = buildDerivations(r.bindings);
  return ds.fixedValues && ds.fixedValues.get(name);
}
function typeOf(src: string, name: string) {
  const r = processSource(src);
  const b = r.bindings.get(name);
  return b && b.inferredType;
}
const normalLogpdf = (x: number, mu: number, sd: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mu) / sd) ** 2;

// =====================================================================
// Nested record values
// =====================================================================

test('nested record value: type nests and chained access resolves', () => {
  const src = `
    r = record(a = record(p = 1.0, q = 2.0), b = 3.0)
    x = r.a.q
  `;
  const tr = typeOf(src, 'r');
  assert.equal(tr.kind, 'record');
  assert.equal(tr.fields.a.kind, 'record');
  assert.ok(T.equal(tr.fields.a.fields.p, T.REAL));
  assert.ok(T.equal(typeOf(src, 'x'), T.REAL));
  assert.equal(valueOf('x', src), 2);
});

test('nested record value: the whole record materialises as nested JS objects', () => {
  const r = valueOf('r', `r = record(a = record(p = 1.0, q = 2.0), b = 3.0)`);
  assert.deepEqual(r, { a: { p: 1, q: 2 }, b: 3 });
});

// =====================================================================
// Nested record measures — sampling
// =====================================================================

test('nested record measure: joint(a = joint(p, q), b) samples with nested fields', async () => {
  const { ctx } = ctxFor(`
m = joint(a = joint(p = Normal(0.0, 1.0), q = Normal(2.0, 1.0)), b = Normal(5.0, 1.0))
`, 4096);
  const m = await ctx.getMeasure('m');
  assert.equal(m.shape, 'record');
  // §06: the keyword name 'a' adds a LEVEL — fields are { a, b }, not the
  // flattened { p, q, b }.
  assert.deepEqual(Object.keys(m.fields).sort(), ['a', 'b']);
  assert.equal(m.fields.a.shape, 'record');
  assert.deepEqual(Object.keys(m.fields.a.fields).sort(), ['p', 'q']);
  const mean = (xs: any) => { let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i]; return s / xs.length; };
  assert.ok(Math.abs(mean(m.fields.a.fields.p.samples) - 0.0) < 0.06, 'E[a.p] ≈ 0');
  assert.ok(Math.abs(mean(m.fields.a.fields.q.samples) - 2.0) < 0.06, 'E[a.q] ≈ 2');
  assert.ok(Math.abs(mean(m.fields.b.samples) - 5.0) < 0.06, 'E[b] ≈ 5');
});

// =====================================================================
// Nested record measures — density
// =====================================================================

test('nested record measure: logdensityof scores the nested record by recursion', async () => {
  const { ctx } = ctxFor(`
m = joint(a = joint(p = Normal(0.0, 1.0), q = Normal(2.0, 1.0)), b = Normal(5.0, 1.0))
lp = logdensityof(m, record(a = record(p = 0.1, q = 2.2), b = 5.3))
`, 1);
  const lp = await ctx.getMeasure('lp');
  const expected = normalLogpdf(0.1, 0, 1) + normalLogpdf(2.2, 2, 1) + normalLogpdf(5.3, 5, 1);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    `nested-record density: got ${lp.samples[0]}, expected ${expected}`);
});
