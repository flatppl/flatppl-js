'use strict';

// Shared helpers for dual-mode equivalence tests against the
// optimization registry (engine/perf-config.ts).
//
// Usage:
//
//   const { inBothModes } = require('./_perf-helpers.ts');
//
//   inBothModes('matmul via aggregate matches mul', 'aggregate', () => {
//     // ... assertions
//   });
//
// Each call registers TWO `node:test` cases — one with the named
// optimization enabled (the specialised path), one with it disabled
// (the general / reference path). Both must agree, which is the
// whole correctness contract for any specialiser.

const { test } = require('node:test');
const perfConfig = require('../perf-config.ts');

/**
 * Register a test under BOTH the `optKey` optimization-on and
 * optimization-off settings. The optimization is restored to its
 * previous value via try/finally, so test order doesn't leak state.
 *
 * `fn` may be sync or async (returning a Promise); the wrapper
 * awaits transparently so assertion failures land on the correct
 * variant.
 */
function inBothModes(name: string, optKey: string, fn: () => any) {
  test(`${name} [opt:${optKey}=on]`, async () => {
    const prev = perfConfig.setOptimization(optKey, true);
    try { await fn(); } finally { perfConfig.setOptimization(optKey, prev); }
  });
  test(`${name} [opt:${optKey}=off]`, async () => {
    const prev = perfConfig.setOptimization(optKey, false);
    try { await fn(); } finally { perfConfig.setOptimization(optKey, prev); }
  });
}

/**
 * Same idea but takes an array of optKeys and toggles them as a
 * set. Useful when a test exercises multiple specialisers at once
 * (e.g. aggregate + density's closed-form path) — the test must
 * hold under every combination of on/off across the set.
 *
 * For N keys this produces 2^N test cases, so use sparingly.
 */
function inAllOptModes(name: string, optKeys: string[], fn: () => any) {
  const combos = 1 << optKeys.length;
  for (let mask = 0; mask < combos; mask++) {
    const settings: Record<string, boolean> = {};
    const tag: string[] = [];
    for (let i = 0; i < optKeys.length; i++) {
      const on = !!(mask & (1 << i));
      settings[optKeys[i]] = on;
      tag.push(`${optKeys[i]}=${on ? 'on' : 'off'}`);
    }
    test(`${name} [opt: ${tag.join(', ')}]`, async () => {
      const snap = perfConfig.snapshotOptimizations();
      perfConfig.setOptimizations(settings);
      try { await fn(); } finally { perfConfig.restoreOptimizations(snap); }
    });
  }
}

module.exports = { inBothModes, inAllOptModes };
