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

test('deriveMcLikelihoodRecipe: auto-derives transport recipe from model bindings, matches oracle', async () => {
  const { processSource, orchestrator } = require(ENG + 'index.ts');
  const { deriveMcLikelihoodRecipe } = require(ENG + 'mat-density.ts');
  const fs = require('node:fs');
  const src = fs.readFileSync(
    '/homedir/Data/Science/Projects/BAT/Projects/FlatPPL/flatppl-examples/examples/tmp_transport_model.flatppl', 'utf8');
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);

  // Auto-derive from the scalar per-event variate `z` over boundary `pars`
  // — no hand-built recipe.
  const r = deriveMcLikelihoodRecipe('z', ['pars'], built.bindings);
  assert.ok(r, 'deriveMcLikelihoodRecipe returned null');
  assert.ok(/^__anon/.test(r.retainedRef.name),
    `retained should be the internal uniform draw, got ${r.retainedRef.name}`);
  assert.strictEqual(r.marginalRef.name, 'x', 'marginalised latent should be the gun draw x');
  assert.deepStrictEqual(r.retainedInterval, [0, 1], 'retained Uniform support');
  assert.strictEqual(r.marginalDistIR.op, 'Normal', 'marginal dist is the gun Normal');

  // Feed the AUTO-DERIVED recipe to the estimator at θ = glob_pars; the
  // frozen point + sigma flow through the worker session env. Compare to
  // the same independent forward-sim oracle.
  const ctx = makeCtx();
  const frozenEnv = { pars: { a: 0.1, b: 0.3, mu: 1.1 }, sigma: 0.2 };
  for (const d of [2.0, 3.0]) {
    const est = await mcMarginalLogDensity({
      ...r, data: [d], M: 8000, ctx, frozenEnv, seedTag: 'derived:d=' + d,
    });
    assert.ok(est != null && Number.isFinite(est), `non-finite est at d=${d}: ${est}`);
    const ora = oracleLogP(d, 2_000_000);
    assert.ok(Math.abs(est - ora) < 0.15,
      `d=${d}: auto-derived estimator log p=${est.toFixed(4)} vs oracle=${ora.toFixed(4)} (Δ=${(est - ora).toFixed(4)})`);
  }
});

test('mcMarginalLogDensity: full likelihood profile over pars.mu is peaked in the interior', async () => {
  // The acid test: a correct likelihood profile over the data must be a
  // PEAKED curve (a proper MLE), not monotone/flat. We sweep pars.mu with
  // a/b at glob_pars, score the model's 20 observations, and require the
  // maximum to fall in the interior of the sweep (a real optimum).
  const { processSource, orchestrator } = require(ENG + 'index.ts');
  const { deriveMcLikelihoodRecipe } = require(ENG + 'mat-density.ts');
  const fs = require('node:fs');
  const src = fs.readFileSync(
    '/homedir/Data/Science/Projects/BAT/Projects/FlatPPL/flatppl-examples/examples/tmp_transport_model.flatppl', 'utf8');
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const r = deriveMcLikelihoodRecipe('z', ['pars'], built.bindings);
  assert.ok(r, 'derivation failed');
  const data = [3.81359, 2.91195, 3.20085, 3.09185, 3.34005, 4.96067, 0.842412,
    2.34128, 3.06224, 2.59162, 2.5017, 5.39892, 1.19806, 1.60855, 1.44647,
    0.771489, 0.26153, 1.56184, 0.561171, 4.4823];
  const ctx = makeCtx();
  const mus = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
  const curve: number[] = [];
  for (const mu of mus) {
    const ll = await mcMarginalLogDensity({
      ...r, data, M: 6000, ctx,
      frozenEnv: { pars: { a: 0.1, b: 0.3, mu }, sigma: 0.2 },
      seedTag: 'profile:mu=' + mu,
    });
    // −inf is VALID and correct: where mu pushes m_z's (hard, deterministic)
    // support off the data, the likelihood is genuinely zero. A proper
    // peaked profile has −inf tails flanking a finite interior optimum.
    assert.ok(ll != null, `log L(mu=${mu}) returned null`);
    curve.push(ll!);
  }
  // Diagnostic: the profile curve (relative to its max), for the record.
  const top = Math.max(...curve);
  process.stderr.write('  log-likelihood profile over pars.mu:\n');
  for (let i = 0; i < mus.length; i++) {
    process.stderr.write(`    mu=${mus[i].toFixed(1)}  logL=${curve[i].toFixed(2)}  Δ=${(curve[i] - top).toFixed(2)}\n`);
  }
  assert.ok(Number.isFinite(top), 'no finite likelihood anywhere in the sweep');
  const argmax = curve.indexOf(top);
  assert.ok(argmax > 0 && argmax < mus.length - 1,
    `profile peak at the sweep boundary (mu=${mus[argmax]}), not an interior MLE`);
  // Peaked, not flat: the endpoints must be clearly below the max (−inf
  // tails trivially satisfy this — a proper hard-support likelihood).
  assert.ok(top - curve[0] > 1 && top - curve[mus.length - 1] > 1,
    `profile too flat to be a real likelihood (Δ ends: ${(top - curve[0]).toFixed(2)}, ${(top - curve[mus.length - 1]).toFixed(2)})`);
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
