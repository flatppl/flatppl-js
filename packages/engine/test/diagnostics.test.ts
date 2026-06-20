const { test } = require('node:test');
const assert = require('node:assert/strict');
const D = require('../diagnostics.ts');

test('splitRHat ~ 1 for well-mixed iid normal chains', () => {
  // Deterministic pseudo-normal via Box-Muller on a seeded LCG (test-local).
  let s = 12345;
  const u = () => (s = (1103515245 * s + 12345) % 2147483648) / 2147483648;
  const norm = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  const chains = [0, 1, 2, 3].map(() => Float64Array.from({ length: 2000 }, norm));
  const rhat = D.splitRHat(chains);
  assert.ok(rhat > 0.98 && rhat < 1.05, `rhat ${rhat} not ~1`);
});

test('splitRHat large for chains stuck at different locations', () => {
  const chains = [
    Float64Array.from({ length: 1000 }, () => 0),
    Float64Array.from({ length: 1000 }, () => 10),
  ];
  assert.ok(D.splitRHat(chains) > 2, 'separated chains -> large rhat');
});

test('essBulk <= total draws and positive for iid chains', () => {
  let s = 999;
  const u = () => (s = (1103515245 * s + 12345) % 2147483648) / 2147483648;
  const chains = [0, 1].map(() => Float64Array.from({ length: 1000 }, u));
  const ess = D.essBulk(chains);
  assert.ok(ess > 0 && ess <= 2000, `ess ${ess} out of range`);
  assert.ok(ess > 500, `iid ess ${ess} unexpectedly low`);
});
