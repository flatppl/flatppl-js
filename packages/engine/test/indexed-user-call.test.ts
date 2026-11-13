'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { resolveBundle } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// §05 Postfix permits calls beneath indexing; §04 functions are objects that
// may be passed to other functions. Both the array and index need lifting.
test('an indexed imported solver call scores with its live callback parameter', async () => {
  const source = `
lib = load_module("lib.flatppl")
solver = load_module("solver.flatppl")
a ~ Normal(0.0, 1.0)
z = lib.apply(solver.solve, a)
y ~ Normal(z, 1.0)
L = likelihoodof(kernelof(y, a = a), 1.0)
P = bayesupdate(L, lawof(record(a = a)))
lp = logdensityof(P, record(a = 0.3))
`;
  const sources: Record<string, string> = {
    'lib.flatppl': `
rhs(y, p) = y .* p
position(i) = i + 1
apply(solver, p) = solver(rhs, [2.0, 3.0], p)[position(0)]
`,
    'solver.flatppl': 'solve(f, y, p) = f(y, p)',
  };
  const bundle = await resolveBundle('entry.flatppl', source,
    async (path: string) => sources[path]);
  const { ctx } = makeMatCtx(source, { path: bundle.primaryPath, bundle, sampleCount: 1 });
  // z = 2a, so the unnormalized posterior is N(a; 0, 1) N(1; 2a, 1).
  const expected = -Math.log(2 * Math.PI) - 0.3 ** 2 / 2 - (1 - 2 * 0.3) ** 2 / 2;
  assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] - expected) < 1e-12);
});
