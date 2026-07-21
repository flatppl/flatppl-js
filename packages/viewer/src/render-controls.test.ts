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

const { SAMPLING_BACKENDS, shouldDeferAutoSample } = await import('./render-controls.ts');

// shouldDeferAutoSample is the gating predicate that keeps sampling
// (MCMC / nested / AMIS / SMC / ESS) starting only on the explicit Sample
// button, not on an automatic re-render triggered by a model/editor edit.
// Cheap non-sampling plots (IS / forward / tractable / array / matrix) must
// keep updating live regardless.

test('SAMPLING_BACKENDS is exactly the stateful-sampler set (excludes IS and forward)', () => {
  assert.deepEqual(
    SAMPLING_BACKENDS,
    ['mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested', 'elliptical-slice-sampler'],
  );
  assert.equal(SAMPLING_BACKENDS.indexOf('is'), -1);
  assert.equal(SAMPLING_BACKENDS.indexOf('forward'), -1);
});

for (const backend of ['mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested', 'elliptical-slice-sampler']) {
  test(`an editor/model-change auto-trigger defers a "${backend}" posterior (no auto sample run)`, () => {
    assert.equal(
      shouldDeferAutoSample({ autoTrigger: true, sampling: true, effectiveBackend: backend }),
      true,
    );
  });
}

test('an explicit trigger (Sample button / DAG click) is never deferred, even for a sampling backend', () => {
  assert.equal(
    shouldDeferAutoSample({ autoTrigger: false, sampling: true, effectiveBackend: 'nested' }),
    false,
  );
});

test('an auto-trigger on a non-sampling backend (IS) is never deferred — cheap plots stay live on edit', () => {
  assert.equal(
    shouldDeferAutoSample({ autoTrigger: true, sampling: true, effectiveBackend: 'is' }),
    false,
  );
});

test('an auto-trigger on a non-posterior plot (effectiveBackend "forward") is never deferred', () => {
  assert.equal(
    shouldDeferAutoSample({ autoTrigger: true, sampling: true, effectiveBackend: 'forward' }),
    false,
  );
});

test('array/matrix mode (sampling=false) is never deferred, even with a stale sampling-backend selection', () => {
  // Regression guard: ctx.inferenceOpts is a single shared object, so a
  // leftover 'nested'/'mh' selection from a previously-viewed posterior must
  // not gate an unrelated array/matrix binding that loads synchronously.
  assert.equal(
    shouldDeferAutoSample({ autoTrigger: true, sampling: false, effectiveBackend: 'nested' }),
    false,
  );
});
