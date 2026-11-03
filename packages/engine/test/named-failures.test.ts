'use strict';

// A run-time failure names the user binding where it happened. Rejections
// from the materialiser once carried only the bare cause, and a failure inside
// a lift anon named the anon (`__anon0`) instead of the binding that reads it.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { ENGINE_LIMITATION } = require('../limitations.ts');

async function rejection(ctx: any, name: string): Promise<any> {
  try {
    await ctx.getMeasure(name);
  } catch (e: any) {
    return e;
  }
  throw new Error(`'${name}' materialised, expected a rejection`);
}

const model = (index: number) => `flatppl_compat = "0.1"
a = [1.0, 2.0]
z ~ Normal(mu = a[${index}], sigma = 1.0)
m = Normal(mu = a[${index}] * 2.0, sigma = 1.0)
s = logdensityof(m, 0.5)
q ~ Normal(mu = z, sigma = 1.0)
`;

test('a materialiser rejection names the innermost user binding', async () => {
  const { ctx } = makeMatCtx(model(5), { sampleCount: 4 });
  const cause = 'evaluateExpr: get index 5 out of bounds for length 2';
  for (const [read, named] of [['z', 'z'], ['m', 'm'], ['s', 's'], ['q', 'z']]) {
    const err = await rejection(ctx, read);
    assert.equal(err.message, `Binding '${named}' failed to evaluate: ${cause}`, read);
  }
  // Oracle for the valid sibling: log N(0.5; mu = 2 * 2, sigma = 1).
  const valid = makeMatCtx(model(2), { sampleCount: 4 }).ctx;
  const s = (await valid.getMeasure('s')).samples[0];
  assert.ok(Math.abs(s - (-0.5 * Math.log(2 * Math.PI) - 0.5 * 3.5 ** 2)) < 1e-12, String(s));
});

test('a named rejection keeps its tags', async () => {
  const { ctx } = ctxFor('flatppl_compat = "0.1"\n'
    + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'm = normalize(weighted(fn(exp(_)), Normal(theta, 1.0)))\n'
    + 'y ~ m\nK = kernelof(record(y = y), theta = theta)\n'
    + 'L = likelihoodof(K, record(y = 0.5))\n'
    + 'lp = logdensityof(L, record(theta = 0.5))\n', 32);
  const err = await rejection(ctx, 'lp');
  assert.equal(err.code, ENGINE_LIMITATION);
  assert.match(err.message, /^Binding 'lp' failed to evaluate: engine limitation: /);
});

test('a failure in a lift anon names the user binding that reads it', () => {
  const build = (g: string) => {
    const r = engine.processSource(`flatppl_compat = "0.1"
g = rowstack(${g})
p = [3.0, 2.0]
g: norm[] := p[.i^] * p[.i_]
`);
    return orchestrator.buildDerivations(r.bindings);
  };
  const built = build('[[1.0, 0.5], [0.7, -1.0]]');
  const want = "Fixed-phase binding 'norm' failed to evaluate: "
    + 'metricsum: metric is not symmetric: A[0,1] = 0.5 but A[1,0] = 0.7';
  assert.throws(() => built.fixedValues.get('norm'), { message: want });
  const errors = built.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors.map((d: any) => d.message), [want]);
  assert.ok(errors[0].loc, 'the diagnostic is located at norm');
  // Oracle for the symmetric sibling. Spec §04 stores p and g all-upper, so
  // p[.i_] = inv(g) p, and inv([[1, 0.5], [0.5, -1]]) = [[0.8, 0.4], [0.4, -0.8]]:
  // 9 * 0.8 + 2 * (3 * 2 * 0.4) + 4 * (-0.8) = 8.8.
  const norm = build('[[1.0, 0.5], [0.5, -1.0]]').fixedValues.get('norm');
  assert.ok(Math.abs(norm - 8.8) < 1e-12, String(norm));
});
