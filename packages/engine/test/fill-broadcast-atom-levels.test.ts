'use strict';

// A density query tiles each input point under a new atom axis. The tiling
// copied the point's outerRank, so the atom axis took the place of the point's
// own outer level: an atom of `theta`, a vector of 2-vectors (§03), sliced back
// as a 3x2 matrix, and `mix.(theta, fill(rowstack(phi), 3))` met three outer
// axes against one ("all collection arguments must have the same number of
// axes"). The atom axis is now a new level above the point's own.
//
// The model is a three-document, two-topic LDA. Its closed form is
//   log p = Σ_m log Dir(theta_m; 1, 1) + Σ_k log Dir(phi_k; 1, 1, 1)
//           + Σ_n log Σ_k theta[d_n, k] phi[k, w_n],
// with Dir(·; 1, 1) = 1 and Dir(·; 1, 1, 1) = 2.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const valueLib = require('../value.ts');

const DOCS = [1, 2, 3, 1];
const WORDS = [1, 3, 2, 2];
const MODEL = `
theta ~ iid(Dirichlet([1.0, 1.0]), 3)
phi ~ iid(Dirichlet([1.0, 1.0, 1.0]), 2)
mix(t, P) = transpose(P) * t
document_words = mix.(theta, fill(rowstack(phi), 3))
w ~ Categorical.(document_words[${JSON.stringify(DOCS)}])
Pi_prior = lawof(record(theta = theta, phi = phi))
K_model = kernelof(record(w = w), theta = theta, phi = phi)
L = likelihoodof(K_model, record(w = ${JSON.stringify(WORDS)}))
Pi_post = bayesupdate(L, Pi_prior)
`;

function closedForm(theta: number[][], phi: number[][]): number {
  let lp = 2 * Math.log(2);
  DOCS.forEach((d, n) => {
    lp += Math.log(theta[d - 1][0] * phi[0][WORDS[n] - 1] + theta[d - 1][1] * phi[1][WORDS[n] - 1]);
  });
  return lp;
}

async function score(theta: number[][], phi: number[][]): Promise<number> {
  const { ctx, proc } = ctxFor(MODEL + `lp = logdensityof(Pi_post, record(theta = ${JSON.stringify(theta)}, phi = ${JSON.stringify(phi)}))\n`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return (await ctx.getMeasure('lp')).samples[0];
}

const THETA = [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]];
const PHI = [[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]];

test('a broadcast over a vector of vectors and a fill of a parametric matrix scores the closed form', async () => {
  const got = await score(THETA, PHI);
  assert.ok(Math.abs(got - (-2.8102158258635175)) < 1e-12, `${got}`);
  assert.ok(Math.abs(closedForm(THETA, PHI) - (-2.8102158258635175)) < 1e-12);
});

test('the density moves along a simplex tangent as the closed form does', async () => {
  // Central difference of the engine density along a direction that keeps
  // every row on its simplex, against the closed form's derivative.
  const dTheta = [[0.1, -0.1], [-0.2, 0.2], [0.05, -0.05]];
  const dPhi = [[0.1, -0.05, -0.05], [-0.1, 0.0, 0.1]];
  const step = (s: number) => [THETA.map((r, i) => r.map((x, k) => x + s * dTheta[i][k])),
    PHI.map((r, i) => r.map((x, k) => x + s * dPhi[i][k]))] as [number[][], number[][]];
  const h = 1e-5;
  const engine = (await score(...step(h)) - await score(...step(-h))) / (2 * h);
  const exact = (closedForm(...step(h)) - closedForm(...step(-h))) / (2 * h);
  const analytic = DOCS.reduce((acc, d, n) => {
    const w = WORDS[n] - 1;
    const p = THETA[d - 1][0] * PHI[0][w] + THETA[d - 1][1] * PHI[1][w];
    const dp = dTheta[d - 1][0] * PHI[0][w] + THETA[d - 1][0] * dPhi[0][w]
      + dTheta[d - 1][1] * PHI[1][w] + THETA[d - 1][1] * dPhi[1][w];
    return acc + dp / p;
  }, 0);
  assert.ok(Math.abs(exact - analytic) < 1e-8, `closed form ${exact} vs analytic ${analytic}`);
  assert.ok(Math.abs(engine - analytic) < 1e-7, `engine ${engine} vs analytic ${analytic}`);
});

test('withAtomAxis puts the atom axis above the cell levels', () => {
  const nested = valueLib.withAtomAxis({ shape: [4, 3, 2], data: new Float64Array(24) },
    { shape: [3, 2], data: new Float64Array(6), outerRank: 1 });
  assert.deepEqual(valueLib.levelAxesOf(nested), [1, 1]);
  assert.equal(valueLib._sliceRow(nested, 2).outerRank, 1);
  const grid = valueLib.withAtomAxis({ shape: [4, 2, 3, 5], data: new Float64Array(120) },
    { shape: [2, 3, 5], data: new Float64Array(30), outerRank: 2, levelAxes: [2] });
  assert.deepEqual(valueLib.levelAxesOf(grid), [1, 2]);
  // A plain array cell still gets the atom level, so a slice returns the cell.
  const flat = valueLib.withAtomAxis({ shape: [4, 3, 2], data: new Float64Array(24) },
    { shape: [3, 2], data: new Float64Array(6) });
  assert.deepEqual(valueLib.levelAxesOf(flat), [1]);
  assert.equal(valueLib._sliceRow(flat, 1).outerRank, undefined);
  // Scalar cells leave the batch untagged.
  const scalars = valueLib.withAtomAxis({ shape: [4], data: new Float64Array(4) }, 2.0);
  assert.equal(scalars.outerRank, undefined);
});
