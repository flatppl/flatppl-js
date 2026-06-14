'use strict';

// PoissonProcess core (spec §08; engine-concepts §2.3) — the ragged
// sampling assembly + per-atom density math, oracle-pinned. Oracle
// values cross-checked in Julia against the spec density
// `(∏ λ(tᵢ))·exp(−M)`, λ(t)=M·shape_pdf(t) — NOT the engine's own output.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const P = require('../mat-poisson.ts');
const R = require('../ragged.ts');

const arr = (a: any) => Float64Array.from(a);
const normlogpdf = (t: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * t * t;

// =====================================================================
// Sampling assembly: counts + flat pool → ragged
// =====================================================================

test('assemble: counts + flat point pool → ragged with cumsum offsets', () => {
  // 3 atoms drawing 2, 0, 3 points; pool laid out atom-major.
  const v = P.assemblePoissonRagged([2, 0, 3], arr([0.1, 0.2, 0.3, 0.4, 0.5]));
  assert.ok(R.isRagged(v));
  assert.equal(R.raggedCount(v), 3);
  assert.deepEqual(Array.from(v.offsets), [0, 2, 2, 5]);
  assert.deepEqual(R.raggedToNested(v).map((a: any) => Array.from(a)),
    [[0.1, 0.2], [], [0.3, 0.4, 0.5]]);
});

test('assemble: all-empty process (every atom drew 0) is a valid ragged', () => {
  const v = P.assemblePoissonRagged([0, 0], arr([]));
  assert.deepEqual(Array.from(v.offsets), [0, 0, 0]);
  assert.equal(v.data.length, 0);
});

test('assemble: pool/count mismatch throws', () => {
  assert.throws(() => P.assemblePoissonRagged([2, 1], arr([1, 2])), /≠ Σcounts/);
});

// =====================================================================
// Density: logp_i = k_i·log(M_i) + Σ shape_logpdf(t) − M_i
// =====================================================================

test('density: inhomogeneous (Normal shape) matches the Julia oracle', () => {
  // Case A: M=5, Normal(0,1) shape, atom points [0.0, 1.0] → −4.119001.
  const obs = R.raggedFromArrays([[0.0, 1.0]]);
  const lp = P.poissonProcessLogDensity(obs, 5.0, normlogpdf);
  assert.deepEqual(lp.shape, [1]);
  assert.ok(Math.abs(lp.data[0] - (-4.119001)) < 1e-5, 'got ' + lp.data[0]);
});

test('density: homogeneous (uniform shape) matches the closed form k·logλ − λL', () => {
  // λ=2 on [0,3]: M=λL=6, shape=Uniform[0,3] (logpdf=−log3); points [0.5,1.5,2.0].
  const obs = R.raggedFromArrays([[0.5, 1.5, 2.0]]);
  const lp = P.poissonProcessLogDensity(obs, 6.0, () => -Math.log(3.0));
  const closed = 3 * Math.log(2.0) - 2.0 * 3.0;   // −3.920558
  assert.ok(Math.abs(lp.data[0] - closed) < 1e-6, 'got ' + lp.data[0]);
});

test('density: empty observation scores −M (no points)', () => {
  const obs = R.raggedFromArrays([[]]);
  const lp = P.poissonProcessLogDensity(obs, 4.0, normlogpdf);
  assert.ok(Math.abs(lp.data[0] - (-4.0)) < 1e-12);
});

test('density: per-atom M (varying expected count) scores each atom independently', () => {
  const obs = R.raggedFromArrays([[0.0, 1.0], [], [0.5]]);
  const M = arr([5.0, 4.0, 3.0]);
  const lp = P.poissonProcessLogDensity(obs, M, normlogpdf);
  // atom 0: case A = −4.119001; atom 1: −M = −4.0; atom 2: 1·log3 + Nlp(0.5) − 3
  assert.ok(Math.abs(lp.data[0] - (-4.119001)) < 1e-5);
  assert.ok(Math.abs(lp.data[1] - (-4.0)) < 1e-12);
  const a2 = Math.log(3.0) + normlogpdf(0.5) - 3.0;
  assert.ok(Math.abs(lp.data[2] - a2) < 1e-12, 'got ' + lp.data[2]);
});
