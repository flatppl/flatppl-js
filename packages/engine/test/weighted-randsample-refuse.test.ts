'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const RNGSEED_SRC = `
rngseed = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
rstate = rnginit(rngseed)
`;

test('rand rejects measures with unproven unit mass during inference', () => {
  for (const bindings of [
    'source = weighted(2.0, Normal(0, 1))',
    'source = logweighted(0.5, Normal(0, 1))',
    'source = weighted(fn(abs(_) + 0.1), Normal(0, 1))',
    'source = logweighted(fn(-abs(_)), Normal(0, 1))',
    'w = weighted(2.0, Normal(0, 1))\nsource = iid(w, 5)',
    'w = weighted(2.0, Normal(0, 1))\nsource = w',
  ]) {
    const { diagnostics } = processSource(RNGSEED_SRC + bindings + '\ns, next = rand(source, rstate)\n');
    assert.ok(diagnostics.some((d: any) => d.severity === 'error' &&
      d.message.startsWith('rand requires a probability measure')), bindings);
  }
});

test('randsample: a non-weighted composite still samples (no regression)', async () => {
  // Sanity check pinning that this fix is scoped to weighted/logweighted —
  // an ordinary composite named binding (a deterministic transform of a
  // stochastic draw) still routes through matRandSample and materialises.
  const src = RNGSEED_SRC + `
x ~ Normal(0, 1)
y = 2.0 * x
law = lawof(y)
s, rstate2 = rand(law, rstate)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  const m = await ctx.getMeasure('s');
  assert.equal(typeof m.value.data[0], 'number');
  assert.ok(Number.isFinite(m.value.data[0]));
});
