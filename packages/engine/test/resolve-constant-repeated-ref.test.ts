'use strict';
// resolveConstant's cycle guard was one set shared by sibling arguments, so a
// fixed ref read twice (`n + n`) looked like a cycle and `iid(M, n + n)` got no
// derivation, while `2 * n` worked. The guard now covers the current path only.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { resolveConstant } = require('../ir-shared.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

test('a fixed size read twice folds', async () => {
  for (const size of ['n + n', 'n * n', 'k + n - 1']) {
    const { ctx } = makeMatCtx(`n = 2\nk = n + 1\nm = iid(Normal(0.0, 1.0), ${size})\nlp = logdensityof(m, [0.1, 0.2, 0.3, 0.4])\n`, { sampleCount: 1 });
    const got = (await ctx.getMeasure('lp')).value.data[0];
    const want = lnN(0.1) + lnN(0.2) + lnN(0.3) + lnN(0.4);
    assert.ok(Math.abs(got - want) < 1e-12, `${size}: ${got}`);
  }
});

test('a repeated ref folds and a real cycle still does not', () => {
  const lifted = orchestrator.buildDerivations(processSource('n = 2\nr = n + n\na = b + 1\nb = a + 1\n').bindings).bindings;
  assert.equal(resolveConstant(lifted.get('r').ir, lifted, new Set()), 4);
  assert.equal(resolveConstant(lifted.get('a').ir, lifted, new Set()), null);
});
