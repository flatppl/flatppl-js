'use strict';

// MC-marginalised log-density (mat-density.mcMarginalLogDensity) for a
// generative model whose observation is a bijective pushforward of one
// retained innovation but marginalises a hidden draw. We validate the
// estimator's per-event log p(z=d | θ) against an INDEPENDENT oracle:
// a fine forward-sample histogram (pure simulation, no inverse/LADJ).
// Agreement confirms the inverse + LADJ conditional and the MC
// marginalisation are jointly correct.
//
// Model (the transport driver, per-event, θ = glob_pars baked):
//   x ~ Normal(1.1, 0.2)            (marginalised gun draw)
//   u ~ Uniform(0, 1)               (retained innovation)
//   δ = (2u + 1)·0.1
//   z = ((x + δ)^3 · exp(x − 0.3)) / 2

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { mcMarginalLogDensity } = require(ENG + 'mat-density.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const lit = (v: number) => ({ kind: 'lit', value: v, numType: Number.isInteger(v) ? 'integer' : 'real' });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const C = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

// z(x, u) = ((x + (2u+1)·0.1)^3 · exp(x − 0.3)) / 2, θ baked as lits.
const delta = C('mul', C('add', C('mul', lit(2), ref('u')), lit(1)), lit(0.1));
const y = C('mul', C('pow', C('add', ref('x'), delta), lit(3)), C('exp', C('sub', ref('x'), lit(0.3))));
const recipeIR = C('divide', y, lit(2));
const marginalDistIR = C('Normal', lit(1.1), lit(0.2));   // positional [mu, sigma]

// Independent oracle: forward-simulate z with a seeded PRNG, histogram,
// read density at d. No inverse/LADJ — pure simulation.
function oracleLogP(d: number, N: number): number {
  let s = 0x2545f491 >>> 0;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s & 0x7fffffff) / 0x7fffffff; };
  const lo = 0, binW = 0.02, nb = 1200;   // [0, 24)
  const counts = new Float64Array(nb);
  for (let i = 0; i < N; i++) {
    const u1 = rnd(), u2 = rnd();
    const xstd = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    const x = 1.1 + 0.2 * xstd;
    const u = rnd();
    const dlt = (2 * u + 1) * 0.1;
    const z = (Math.pow(x + dlt, 3) * Math.exp(x - 0.3)) / 2;
    const b = Math.floor((z - lo) / binW);
    if (b >= 0 && b < nb) counts[b]++;
  }
  const bin = Math.floor((d - lo) / binW);
  return Math.log(counts[bin] / (N * binW));
}

function makeCtx() {
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 4242 });
  return {
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    rootKey: 4242,
  };
}

test('mcMarginalLogDensity: per-event log p(z=d|θ) matches a forward-sim oracle', async () => {
  const ctx = makeCtx();
  // Data points spanning the bulk of m_z(θ): mode ≈ 2.4, so test below,
  // at, and above the mode.
  const points = [1.8, 2.4, 3.2, 4.0];
  const N_ORACLE = 2_000_000;
  for (const d of points) {
    const est = await mcMarginalLogDensity({
      recipeIR, retainedRef: { name: 'u' }, retainedInterval: [0, 1],
      marginalRef: { name: 'x' }, marginalDistIR,
      data: [d], M: 8000, ctx, seedTag: 'test:d=' + d,
    });
    assert.ok(est != null, `estimator returned null for d=${d} (recipe should be invertible in u)`);
    const ora = oracleLogP(d, N_ORACLE);
    assert.ok(Number.isFinite(est) && Number.isFinite(ora),
      `non-finite: est=${est} oracle=${ora} at d=${d}`);
    // Tolerance absorbs MC (M=8000) + histogram (binW=0.02) noise; a
    // wrong inverse/LADJ would be off by >> this.
    assert.ok(Math.abs(est - ora) < 0.15,
      `d=${d}: estimator log p=${est.toFixed(4)} vs oracle=${ora.toFixed(4)} (Δ=${(est - ora).toFixed(4)})`);
  }
});

test('mcMarginalLogDensity: iid total = sum of per-event log-densities', async () => {
  const ctx = makeCtx();
  const data = [2.2, 2.9, 3.5];
  const total = await mcMarginalLogDensity({
    recipeIR, retainedRef: { name: 'u' }, retainedInterval: [0, 1],
    marginalRef: { name: 'x' }, marginalDistIR, data, M: 4000, ctx, seedTag: 'iid',
  });
  // Same draws/seed per call ⇒ per-point calls sum to the joint (the
  // estimator reuses one marginalised sample set across the D points).
  let sum = 0;
  for (const d of data) {
    sum += (await mcMarginalLogDensity({
      recipeIR, retainedRef: { name: 'u' }, retainedInterval: [0, 1],
      marginalRef: { name: 'x' }, marginalDistIR, data: [d], M: 4000, ctx, seedTag: 'iid',
    }))!;
  }
  assert.ok(Math.abs(total - sum) < 1e-9,
    `iid factorisation broken: total=${total} vs Σ per-event=${sum}`);
});

test('mcMarginalLogDensity: returns null when the recipe is not bijective in the retained draw', async () => {
  const ctx = makeCtx();
  // z = x·u — appears in u AND (if we asked for x) … here ask to retain
  // `x`, which occurs in two places of the real recipe ⇒ not invertible.
  const res = await mcMarginalLogDensity({
    recipeIR, retainedRef: { name: 'x' }, retainedInterval: [0, 1],
    marginalRef: { name: 'u' }, marginalDistIR: C('Uniform', lit(0), lit(1)),
    data: [3.0], M: 100, ctx, seedTag: 'nonbij',
  });
  assert.strictEqual(res, null);
});
