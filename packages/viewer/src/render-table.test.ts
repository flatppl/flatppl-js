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
