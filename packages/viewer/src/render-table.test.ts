// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { tableDiagnostics } = await import('./render-table.ts');

test('tableDiagnostics: MCMC perParam supplies per-variate ess + rhat', () => {
  const measure = { diagnostics: { perParam: { sigma: { rHat: 1.012, essBulk: 842 } } } };
  const d = tableDiagnostics(measure, 'sigma', 999);
  assert.equal(d.ess, 842);
  assert.equal(d.rhat, 1.012);
});

test('tableDiagnostics: no perParam entry falls back to Kish ESS, rhat NaN', () => {
  const measure = { diagnostics: { perParam: {} } };
  const d = tableDiagnostics(measure, 'sigma', 999);
  assert.equal(d.ess, 999);
  assert.ok(Number.isNaN(d.rhat));
});

test('tableDiagnostics: no diagnostics at all (IS) → fallback ESS, rhat NaN', () => {
  const d = tableDiagnostics({}, 'mu', 12345);
  assert.equal(d.ess, 12345);
  assert.ok(Number.isNaN(d.rhat));
});

test('tableDiagnostics: non-finite perParam values fall back', () => {
  const measure = { diagnostics: { perParam: { mu: { rHat: NaN, essBulk: NaN } } } };
  const d = tableDiagnostics(measure, 'mu', 500);
  assert.equal(d.ess, 500);
  assert.ok(Number.isNaN(d.rhat));
});

test('tableDiagnostics: a vector latent row reads its own coordinate, not the next one', async () => {
  // The samplers key perParam by the model view's coordinate names; the table
  // labels come from listScalarAxes. Both are built by the real code here.
  const { createRequire } = await import('node:module');
  const require_ = createRequire(import.meta.url);
  const { ctxFor } = require_('@flatppl/engine/test/_ctx-factory.ts');
  const MV = require_('@flatppl/engine/model-view.ts');
  const { listScalarAxes } = await import('./util.ts');
  const { ctx } = ctxFor(`
theta ~ iid(Normal(0.0, 1.0), 3)
y ~ Normal(sum(theta), 1.0)
prior = lawof(record(theta = theta))
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
posterior = bayesupdate(L, prior)
`, 50);
  const mv = await MV.buildModelViewFromCtx(ctx, ctx.derivations['posterior']);
  assert.equal(mv.names.length, 3);
  // Coordinate j (0-based slot in the flat draw) carries ESS 100·(j+1).
  const perParam = {};
  mv.names.forEach((nm, j) => { perParam[nm] = { rHat: 1 + j / 100, essBulk: 100 * (j + 1) }; });
  const N = 4;
  const samples = new Float64Array(N * 3);
  const measure = { diagnostics: { perParam }, fields: { theta: { shape: 'array', dims: [3], samples } } };
  const axes = listScalarAxes(measure);
  assert.deepEqual(axes.map((a) => a.label), ['theta[1]', 'theta[2]', 'theta[3]']);
  axes.forEach((a, j) => {
    const d = tableDiagnostics(measure, a.label, 7);
    assert.equal(d.ess, 100 * (j + 1), `${a.label} is slot ${j}`);
    assert.equal(d.rhat, 1 + j / 100);
  });
});
