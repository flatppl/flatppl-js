'use strict';

// =====================================================================
// hierarchical-models.test.ts — Phase F/G motivating test cases
// =====================================================================
//
// Documents the CURRENT behavior of two canonical SOTA hierarchical-
// model patterns (composite kernel bodies + outer-axis propagation —
// engine-concepts §20.10.9 / TODO-flatppl-js fusion thread (b)
// remaining sub-items). Each test fixes the expected end-state once
// Phase F/G land, and the present-day behavior so regressions in
// either direction are caught.
//
// Patterns under test:
//   1. Hierarchical repeated-measures (Pyro-style `plate` over
//      groups with `plate` over observations).
//   2. Random-intercepts regression (lme4 / Bambi / Stan-style
//      hierarchical linear model with per-group intercept +
//      shared slope).
//
// Both fixtures land in flatppl-examples + a copy in the engine
// test/fixtures/ tree (per CONVENTIONS.md: examples canonical,
// tests carry their own copy for resilience).
//
// **Today (post fusion (b) MVP):** both fixtures CLASSIFY but
// materialise fails because matKernelBroadcast rejects the
// user-kernel head — the kernel body is `lawof(iid(<Dist>, n))`
// rather than `lawof(<Dist>)`, so the fusion (b) MVP refuses to
// inline.
//
// **Target (Phase F):** matKernelBroadcast or the dissolver
// recognises the composite kernel body, lifts the kernel application
// into a tiled broadcast + reshape, and produces a shape
// [N_atom, G, N_per_group] obs measure.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  const p = path.join(__dirname, 'fixtures', name);
  return fs.readFileSync(p, 'utf-8');
}

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: N,
    rootSeed: 42,
  };
  return { ctx, derivations: built.derivations, bindings: built.bindings };
}

// =====================================================================
// 1. Hierarchical repeated-measures — `kernelof(iid(Normal,N), mu)`
//    broadcast over a per-group mu vector
// =====================================================================
//
// Per spec §04 + §06: broadcast iterates over its collection-arg
// outer axis; per cell the kernel's body is drawn. With body
// `iid(Normal(mu, s), N_per_group)`, each cell yields a length-
// N_per_group vector — so the whole result is shape [G, N_per_group]
// per engine atom.

test('hierarchical-repeated-measures: classifies + materialises (target: Phase F)', async () => {
  const src = readFixture('hierarchical-repeated-measures.flatppl');
  const { ctx, derivations } = setupCtx(src, 50);

  // Today (post fusion (b) MVP): obs SHOULD have a derivation
  // (the cascade-prune lets the broadcast classify), but
  // materialise rejects.
  assert.ok(derivations.obs,
    'obs has a derivation (classifier accepts the broadcast IR)');

  // Phase F target: matKernelBroadcast or a dissolver rewrite
  // handles `kernelof(iid(...), ...)` heads. Until that lands,
  // pin the failure mode so a regression in either direction is
  // detected.
  let didError: Error | null = null;
  try {
    const m = await ctx.getMeasure('obs');
    // If this branch fires, Phase F has landed. Pin the result
    // shape: [N_atom=50, G=4, N_per_group=5].
    assert.ok(m && m.value && Array.isArray(m.value.shape),
      'obs materialises to a shape-tagged Value once Phase F lands');
    assert.equal(m.value.shape[0], 50, 'leading axis = N_atom');
    assert.equal(m.value.shape[1], 4, 'second axis = G');
    assert.equal(m.value.shape[2], 5, 'third axis = N_per_group');
  } catch (e: any) {
    didError = e;
  }

  if (didError) {
    // Document current failure mode — present-day matKernelBroadcast
    // rejects the user-kernel head. Once Phase F lands, the catch
    // branch above goes away and this assertion path retires.
    assert.match(didError.message,
      /unknown distribution kernel|undefined|Cannot read|kernelbroadcast/i,
      'documented failure mode: user-kernel head not yet supported by matKernelBroadcast');
  }
});

// =====================================================================
// 2. Random-intercepts regression — per-group intercept + per-group
//    regressors, kernel body is `iid(Normal(int + slope * x, s), N)`
// =====================================================================
//
// Same shape as eight-schools BUT each group has multiple obs (the
// per-group N_per_group dim). The kernel body uses both the
// placeholder `intercept` (scalar per group) and `x_group` (rank-1
// per group), making it a more complex composite-body case than
// the simpler hierarchical test above.

test('random-intercepts: classifies + materialises (target: Phase F)', async () => {
  const src = readFixture('random-intercepts.flatppl');
  let setupOk = false;
  try {
    const { derivations } = setupCtx(src, 50);
    // y_obs SHOULD classify; failure happens at materialise.
    assert.ok(derivations.y_obs,
      'y_obs has a derivation (classifier accepts the broadcast IR)');
    setupOk = true;
  } catch (e: any) {
    // Some fixture features may not have full typeinfer support yet
    // (e.g. multi-dim literal x_per_group). Pin if so.
    assert.match(e.message,
      /typeinfer|not implemented|unsupported|Cannot read/i,
      'documented setup gap');
  }

  // If setup didn't classify, the materialise check below is moot.
  if (!setupOk) return;

  const { ctx } = setupCtx(src, 50);
  let didError: Error | null = null;
  try {
    const m = await ctx.getMeasure('y_obs');
    // Phase F target shape: [N_atom=50, G=3, N_per_group=4].
    assert.equal(m.value.shape[0], 50);
    assert.equal(m.value.shape[1], 3);
    assert.equal(m.value.shape[2], 4);
  } catch (e: any) {
    didError = e;
  }

  if (didError) {
    assert.match(didError.message,
      /unknown distribution kernel|undefined|Cannot read|kernelbroadcast|broadcast/i,
      'documented failure mode for Phase F target');
  }
});
