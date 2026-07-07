'use strict';

// =====================================================================
// Value-level broadcast over a bare BUILTIN head.
// =====================================================================
//
// The determiniser lowers `logdensityof(broadcast(K, params), obs)` to the
// axis-native FlatPDL density form (spec §04 value-level broadcast; §07
// builtin_logdensityof). The actual `flatppl determinize` emission for
// `logdensityof(broadcast(Poisson, rates), obs)` is (surface):
//
//   score = sum(builtin_logdensityof.(Poisson, broadcast(record, rate = rates), obs))
//
// which lowers to two nested value-level broadcasts whose heads are
// builtins the engine could not previously apply:
//
//   - inner `broadcast(record, rate = rates)`  — a bare `record` head
//     → array of `{rate: rᵢ}` records.
//   - outer `builtin_logdensityof.(Poisson, <records>, obs)` — a
//     functionof head over `builtin_logdensityof`, whose FIRST arg is a
//     bare kernel NAME (`Poisson`) held constant across cells. That name
//     must reach `_resolveKernelName` as IR, never as an evaluated value.
//
// These exercise the general mechanism (any builtin head, held kernel
// name inlined) — not a Poisson special case.
//
// Oracle (computed independently, Poisson log-pmf = k·ln λ − λ − ln k!):
//   λ = [2.0, 3.5, 1.0], obs = [1, 4, 2]
//   per-cell = [-1.3068528194400546, -1.6670019563664722, -1.693147180559945]
//   Σ        = -4.667001956366471

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function fixedValue(src: string, name: string): any {
  const r = engine.processSource(src);
  for (const d of r.diagnostics || []) {
    if (d.severity === 'error') {
      throw new Error('diagnostic: ' + d.message);
    }
  }
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  return derivs.fixedValues.get(name);
}

// Independent Poisson log-pmf oracle.
function poissonLogpmf(k: number, lam: number): number {
  const stdlibGammaln = require('@stdlib/math-base-special-gammaln');
  return k * Math.log(lam) - lam - stdlibGammaln(k + 1);
}

// ---------------------------------------------------------------------
// Focused: bare `record` head → array of records.
// ---------------------------------------------------------------------

test('broadcast(record, rate = [..]) → array of {rate} records', () => {
  const recs = fixedValue(
    'flatppl_compat = "0.1"\n' +
    'recs = broadcast(record, rate = [2.0, 3.0])\n',
    'recs');
  assert.ok(Array.isArray(recs), `expected array, got ${JSON.stringify(recs)}`);
  assert.equal(recs.length, 2);
  assert.equal(recs[0].rate, 2.0);
  assert.equal(recs[1].rate, 3.0);
});

// ---------------------------------------------------------------------
// Outer builtin_logdensityof broadcast, unsummed — per-cell vector.
// ---------------------------------------------------------------------

test('builtin_logdensityof.(Poisson, records, obs) → per-cell log-density vector', () => {
  const lam = [2.0, 3.5, 1.0];
  const obs = [1, 4, 2];
  const lps = fixedValue(
    'flatppl_compat = "0.1"\n' +
    'rates = [2.0, 3.5, 1.0]\n' +
    'obs = [1, 4, 2]\n' +
    'lps = builtin_logdensityof.(Poisson, broadcast(record, rate = rates), obs)\n',
    'lps');
  // The value-level broadcast packs finite scalar cells into a
  // shape-explicit Value (engine-concepts §2.1).
  const data = (lps && lps.data) ? lps.data : lps;
  assert.ok(data && typeof data.length === 'number' && data.length === 3,
    `expected a length-3 vector, got ${JSON.stringify(lps)}`);
  for (let i = 0; i < 3; i++) {
    const expected = poissonLogpmf(obs[i], lam[i]);
    assert.ok(Math.abs(data[i] - expected) < 1e-12,
      `cell ${i}: got ${data[i]}, expected ${expected}`);
  }
});

// ---------------------------------------------------------------------
// Full determinized density form: sum of per-cell log-densities.
// ---------------------------------------------------------------------

test('sum(builtin_logdensityof.(Poisson, broadcast(record, rate=rates), obs)) ≡ Σ Poisson.logpmf', () => {
  const lam = [2.0, 3.5, 1.0];
  const obs = [1, 4, 2];
  const oracle = lam.reduce((acc, l, i) => acc + poissonLogpmf(obs[i], l), 0);
  // sanity: the independently-computed oracle constant.
  assert.ok(Math.abs(oracle - (-4.667001956366471)) < 1e-12);

  const lp = fixedValue(
    'flatppl_compat = "0.1"\n' +
    'rates = [2.0, 3.5, 1.0]\n' +
    'obs = [1, 4, 2]\n' +
    'lp = sum(builtin_logdensityof.(Poisson, broadcast(record, rate = rates), obs))\n',
    'lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  assert.ok(Math.abs(lp - oracle) < 1e-9,
    `lp = ${lp}, oracle = ${oracle}, diff = ${Math.abs(lp - oracle)}`);
});
