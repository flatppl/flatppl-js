'use strict';

// `in` over a Cartesian product (§03 "Cartesian product"). A record lies in
// `cartprod(name = S, …)` when each field lies in the set of the same name,
// in any field order. A positional member is the `cat` of one element per
// component, so a vector component spans its own width. The determiniser's
// record truncation emits `record(s = p) in cartprod(s = cartpow(posreals, 2))`
// (rust #531), which this engine refused with "unsupported set shape".

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function fixed(src: string): Record<string, any> {
  const r = processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  const fv = orchestrator.buildDerivations(r.linkedBindings || r.bindings).fixedValues;
  return new Proxy({}, { get: (_t, k: string) => fv.get(k) });
}

test('a record is in a record set when each field is in its set, in any order', () => {
  const v = fixed(
    'a = record(s = [0.5, 1.5]) in cartprod(s = cartpow(posreals, 2))\n'
    + 'b = record(s = [-0.5, 1.5]) in cartprod(s = cartpow(posreals, 2))\n'
    + 'c = record(y = 2.0, x = [0.2, 0.7]) in cartprod(x = cartpow(unitinterval, 2), y = posreals)\n'
    + 'd = record(y = -2.0, x = [0.2, 0.7]) in cartprod(x = cartpow(unitinterval, 2), y = posreals)\n'
    + 'e = record(r = record(u = 1.0), v = 3) in cartprod(r = cartprod(u = posreals), v = posintegers)\n'
    + 'f = record(a = 1.0, b = [0.1, 0.2]) in cartprod(cartprod(a = posreals), cartprod(b = cartpow(unitinterval, 2)))\n'
    + 'g = record(b = [0.1, 2.0], a = 1.0) in cartprod(cartprod(a = posreals), cartprod(b = cartpow(unitinterval, 2)))\n');
  assert.deepEqual([v.a, v.b, v.c, v.d, v.e, v.f, v.g], [true, false, true, false, true, true, false]);
});

test('a record with other fields than the set is refused, not answered', () => {
  const r = processSource('e = record(s = [0.5, 1.5], t = 1.0) in cartprod(s = cartpow(posreals, 2))\n');
  const fv = orchestrator.buildDerivations(r.linkedBindings || r.bindings).fixedValues;
  assert.throws(() => fv.get('e'), /expects a record with exactly those fields, got fields s, t/);
});

test('a positional member splits into its components, vector components by width', () => {
  const v = fixed(
    'a = [1.0, -2.0] in cartprod(posreals, reals)\n'
    + 'b = [1.0, -2.0] in cartprod(posreals, posreals)\n'
    + 'c = [0.5, 1.5, 1.0, 2.0, 3.0] in cartprod(ordered(2), cartpow(posreals, 3))\n'
    + 'd = [1.5, 0.5, 1.0, 2.0, 3.0] in cartprod(ordered(2), cartpow(posreals, 3))\n');
  assert.deepEqual([v.a, v.b, v.c, v.d], [true, false, true, false]);
});

// `logweighted(s -> -sum(s), truncate(Lebesgue(ordered(2)), cartpow(posreals, 2)))`
// over a record: the positive ordered pairs, with density exp(-s1 - s2) there.
const NATIVE = (s: string) => `log_prior(r) = -sum(r.s)
M = logweighted(log_prior, truncate(Lebesgue(cartprod(s = ordered(2))), cartprod(s = cartpow(posreals, 2))))
lp = logdensityof(M, record(s = ${s}))
`;
// \`flatppl determinize\` of NATIVE('[0.5, 1.5]') --keep lp, rust 7b335b8.
const DETERMINISED = (s: string) => `lp = add(
  -sum(${s}),
  ifelse(
    record(s = ${s}) in cartprod(s = cartpow(posreals, 2)),
    ifelse(true && ${s} in ordered(2), 0.0, -inf),
    -inf,
  ),
)
`;

for (const [label, model] of [['native', NATIVE], ['determinised', DETERMINISED]] as [string, (s: string) => string][]) {
  test(`the record truncation scores the closed form (${label})`, async () => {
    for (const [s, expected] of [['[0.5, 1.5]', -2.0], ['[1.5, 0.5]', -Infinity], ['[-0.5, 1.5]', -Infinity]] as [string, number][]) {
      const { ctx } = ctxFor(model(s), 1);
      const m = await ctx.getMeasure('lp');
      assert.equal(m.samples ? m.samples[0] : m.value.data[0], expected, s);
    }
  });
}
