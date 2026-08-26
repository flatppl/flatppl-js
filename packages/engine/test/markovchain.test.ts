'use strict';

// =====================================================================
// markovchain.test.ts — spec §06 `markovchain(kernel, init, n)`
// =====================================================================
//
// §06 dependent composition: "Step $i$ is $\text{traj}_i \sim
// \kappa(\text{traj}_{i-1})$ with $\text{traj}_0 = \text{init}$. The initial
// value is not part of the trajectory. The resulting measure is a measure
// over arrays `[traj[1], ..., traj[n]]`, excluding the initial state."
//
// Every number here is pinned against a CLOSED FORM, never against an engine
// output. The step kernel throughout is the AR-1 random walk
// x_k = x_{k-1} + N(0, sigma_step), whose moments are exact:
//
//   Var(traj[k])            = Var(init) + k·sigma_step^2
//   Cov(traj[j], traj[k])   = Var(init) + min(j,k)·sigma_step^2
//
// (Var(init) = 0 for a literal init; sigma_init^2 when init is a draw.) That
// second form is the SAME calibration `hierarchical-models.test.ts` pins for
// `test/fixtures/hierarchical-state-space.flatppl`, which spells the chain
// with `jointchain` because markovchain did not exist — see
// TODO-flatppl-js.md. The fixture is left alone; this is the markovchain
// spelling of the same law, so the two can be compared before any migration.
//
// The density is exact (no MC): §06 makes `init` a VALUE, so there is no base
// term and logdensityof is the sum of exactly n Gaussian transition
// log-densities, hand-summed below.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 60000;
const ROOT_SEED    = 0xABCD;

function makeCtx(source: any, sampleCount?: number) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    diagnostics: lifted.diagnostics || [],
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: sampleCount || SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

/**
 * Error diagnostics from analysis + derivation building, no materialisation.
 * Both channels: typeinfer reports the argument types, `buildDerivations`
 * reports the step-kernel shapes it will not lower.
 */
function errorsOf(source: any): string[] {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  return (lifted.diagnostics || []).concat(built.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

/** Column j (1-based) of an atom-major array measure with per-atom dims [n]. */
function column(m: any, n: number, j: number): Float64Array {
  const N = m.samples.length / n;
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) out[i] = m.samples[i * n + (j - 1)];
  return out;
}

function mean(xs: any) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function variance(xs: any) {
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - m) * (xs[i] - m);
  return s / xs.length;
}

function covariance(xs: any, ys: any) {
  const mx = mean(xs);
  const my = mean(ys);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
  return s / xs.length;
}

/** Normal log-pdf, closed form — the density oracle. */
function normLogpdf(x: number, mu: number, sigma: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) * ((x - mu) / sigma);
}

// =====================================================================
// Variate shape — §06's "measure over arrays […], excluding the initial state"
// =====================================================================

test('markovchain: the variate is an ARRAY of n states, init excluded', async () => {
  // n = 4 with a literal init. Two claims, both structural:
  //   - the measure is array-shaped with per-atom dims [4], not a tuple and
  //     not [5] — the initial state is not a trajectory element;
  //   - traj[1] is one STEP from init, not init itself, so its variance is
  //     sigma_step^2 rather than zero.
  const ctx = makeCtx(`
f = fn(Normal(mu = _, sigma = 0.5))
traj = markovchain(f, 0.0, 4)
`);
  const m = await ctx.getMeasure('traj');
  assert.equal(m.shape, 'array', 'markovchain materialises an array measure');
  assert.deepEqual(m.dims, [4], 'per-atom dims are [n], not [n+1]');
  assert.equal(m.samples.length, SAMPLE_COUNT * 4);
  const v1 = variance(column(m, 4, 1));
  assert.ok(Math.abs(v1 - 0.25) < 0.02,
    'Var(traj[1]) = sigma_step^2 = 0.25 (traj[1] is a step FROM init, not '
    + 'init, whose variance would be 0), got ' + v1);
});

// =====================================================================
// Sampling moments — AR-1 random walk, literal init (Var(init) = 0)
// =====================================================================

test('markovchain: AR-1 moments Var(traj[k]) = k·sigma_step^2 (closed form)',
  async () => {
  // f_step = x -> Normal(x, 0.5): the increments are independent
  // N(0, 0.5), so traj[k] = sum of k of them and
  //   Var(traj[k])          = k · 0.25
  //   Cov(traj[j], traj[k]) = min(j,k) · 0.25
  // Tolerances: sd(s^2) ~ sigma^2·sqrt(2/N) = 0.58% of the target at
  // N = 60000, so 0.02 absolute is > 3 sd on the largest target.
  const ctx = makeCtx(`
sigma_step = 0.5
f = fn(Normal(mu = _, sigma = sigma_step))
traj = markovchain(f, 0.0, 4)
`);
  const m = await ctx.getMeasure('traj');
  const c = [1, 2, 3, 4].map((j) => column(m, 4, j));
  for (let k = 1; k <= 4; k++) {
    const got = variance(c[k - 1]);
    assert.ok(Math.abs(got - 0.25 * k) < 0.02,
      `Var(traj[${k}]) = ${0.25 * k}, got ${got}`);
    // E[traj[k]] = init = 0 for every k (a driftless walk).
    assert.ok(Math.abs(mean(c[k - 1])) < 0.02,
      `E[traj[${k}]] = 0, got ${mean(c[k - 1])}`);
  }
  // Cov(traj[j], traj[k]) = min(j,k)·sigma_step^2 — the walk's correlation
  // structure, which an independent product would get wrong (it would give 0).
  const cov14 = covariance(c[0], c[3]);
  assert.ok(Math.abs(cov14 - 0.25) < 0.02,
    'Cov(traj[1], traj[4]) = 1·0.25, got ' + cov14);
  const cov23 = covariance(c[1], c[2]);
  assert.ok(Math.abs(cov23 - 0.50) < 0.02,
    'Cov(traj[2], traj[3]) = 2·0.25, got ' + cov23);
});

// =====================================================================
// Sampling moments — the AR-1 calibration the jointchain fixture pins
// =====================================================================

test('markovchain: a drawn init reproduces sigma_init^2 + k·sigma_step^2',
  async () => {
  // The calibration `hierarchical-models.test.ts` pins for
  // `test/fixtures/hierarchical-state-space.flatppl`, written with the op §06
  // gives the Markov reading. §06 makes `init` a VALUE, so a random initial
  // condition is a draw fed as that value: Var(init) = sigma_init^2 rides
  // into every trajectory position.
  //   Var(traj[k])          = sigma_init^2 + k·sigma_step^2
  //   Cov(traj[j], traj[k]) = sigma_init^2 + min(j,k)·sigma_step^2
  const ctx = makeCtx(`
sigma_init = 0.1
sigma_step = 0.5
step_kernel = kernelof(Normal(mu = prev, sigma = sigma_step), prev = prev)
x0 = draw(Normal(mu = 0.0, sigma = sigma_init))
traj = markovchain(step_kernel, x0, 3)
`);
  const m = await ctx.getMeasure('traj');
  assert.deepEqual(m.dims, [3]);
  const c = [1, 2, 3].map((j) => column(m, 3, j));
  for (let k = 1; k <= 3; k++) {
    const want = 0.01 + 0.25 * k;
    const got = variance(c[k - 1]);
    assert.ok(Math.abs(got - want) < 0.02,
      `Var(traj[${k}]) = sigma_init^2 + ${k}·sigma_step^2 = ${want}, got ${got}`);
  }
  const cov13 = covariance(c[0], c[2]);
  assert.ok(Math.abs(cov13 - 0.26) < 0.02,
    'Cov(traj[1], traj[3]) = sigma_init^2 + 1·sigma_step^2 = 0.26, got ' + cov13);
});

// =====================================================================
// Density — exact, n transition terms and no base term
// =====================================================================

test('markovchain: logdensityof is the sum of the n transitions (exact)',
  async () => {
  // §06 gives the constituent conditionals; with init a value there is no
  // base term, so the n = 4 trajectory [0.3, 0.5, 0.9, 1.4] from init 0
  // scores logN(0.3;0,1) + logN(0.5;0.3,1) + logN(0.9;0.5,1) + logN(1.4;0.9,1).
  // Exact — no Monte Carlo anywhere on this path.
  const ctx = makeCtx(`
f = fn(Normal(mu = _, sigma = 1.0))
traj = markovchain(f, 0.0, 4)
lp = logdensityof(traj, [0.3, 0.5, 0.9, 1.4])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.3, 0.0, 1.0) + normLogpdf(0.5, 0.3, 1.0)
    + normLogpdf(0.9, 0.5, 1.0) + normLogpdf(1.4, 0.9, 1.0);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `markovchain logp ${m.samples[0]} vs closed form ${want}`);
});

test('markovchain: a non-zero init enters step 1 and only step 1', async () => {
  // Shifting init changes exactly the FIRST transition's mean. Pinning it
  // separately catches an off-by-one that fed init to every step (or to none).
  const ctx = makeCtx(`
f = fn(Normal(mu = _, sigma = 2.0))
traj = markovchain(f, -1.5, 3)
lp = logdensityof(traj, [0.4, 1.1, 0.2])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.4, -1.5, 2.0) + normLogpdf(1.1, 0.4, 2.0)
    + normLogpdf(0.2, 1.1, 2.0);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `markovchain logp ${m.samples[0]} vs closed form ${want}`);
});

// =====================================================================
// n = 1 — §06 admits any positive n
// =====================================================================

test('markovchain: n = 1 is one step from init, density is one transition',
  async () => {
  // The edge §06 allows and `jointchain` cannot spell (it requires ≥ 2
  // components). The trajectory is [traj[1]] alone; init is still excluded.
  const ctx = makeCtx(`
f = fn(Normal(mu = _, sigma = 0.5))
traj = markovchain(f, 2.0, 1)
lp = logdensityof(traj, [2.5])
`, 40000);
  const lp = await ctx.getMeasure('lp');
  const want = normLogpdf(2.5, 2.0, 0.5);
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `n=1 logp ${lp.samples[0]} vs closed form ${want}`);

  const m = await ctx.getMeasure('traj');
  assert.equal(m.shape, 'array');
  assert.deepEqual(m.dims, [1], 'a length-1 trajectory is still an array');
  const c = column(m, 1, 1);
  assert.ok(Math.abs(mean(c) - 2.0) < 0.02, 'E[traj[1]] = init = 2.0, got ' + mean(c));
  assert.ok(Math.abs(variance(c) - 0.25) < 0.02,
    'Var(traj[1]) = sigma_step^2 = 0.25, got ' + variance(c));
});

// =====================================================================
// Spellings §06 and §05 admit for the kernel and for n
// =====================================================================

test('markovchain: the §06 Brownian-motion example spelling samples',
  async () => {
  // §06's own example, verbatim in shape: a `->` lambda whose body passes the
  // distribution's parameters POSITIONALLY (§05). Var(traj[k]) = k·2·D·dt.
  // 20 steps rather than §06's 100 keeps the estimator tight enough to pin.
  const ctx = makeCtx(`
D = 4.1
dt = 0.01
f_step = x -> Normal(x, sqrt(2*D * dt))
traj = markovchain(f_step, 0.0, 20)
`);
  const m = await ctx.getMeasure('traj');
  assert.deepEqual(m.dims, [20]);
  const step2 = 2 * 4.1 * 0.01;                     // 0.082
  for (const k of [1, 10, 20]) {
    const got = variance(column(m, 20, k));
    const want = k * step2;
    assert.ok(Math.abs(got - want) < 0.05 * want + 0.005,
      `Var(traj[${k}]) = ${k}·2·D·dt = ${want}, got ${got}`);
  }
});

test('markovchain: n may be a named integer binding, not only a literal',
  async () => {
  const ctx = makeCtx(`
N_STEPS = 3
f = fn(Normal(mu = _, sigma = 0.5))
traj = markovchain(f, 0.0, N_STEPS)
`);
  const m = await ctx.getMeasure('traj');
  assert.deepEqual(m.dims, [3]);
  assert.ok(Math.abs(variance(column(m, 3, 3)) - 0.75) < 0.02,
    'Var(traj[3]) = 3·0.25 = 0.75');
});

// =====================================================================
// Located refusals
// =====================================================================

test('markovchain: a non-kernel first argument is a located error', () => {
  // §06: arg 1 is "a Markov kernel `(state) -> measure_over_state`". A MEASURE
  // there is not one — the uniform kernel extension makes a measure a NULLARY
  // kernel, which has no state to step from.
  const msgs = errorsOf('traj = markovchain(Normal(mu = 0.0, sigma = 1.0), 0.0, 3)\n');
  assert.ok(msgs.some((m) => /markovchain: arg 1 expects a Markov kernel/.test(m)),
    'expected a located non-kernel diagnostic, got ' + JSON.stringify(msgs));
});

test('markovchain: a multi-input kernel is a located error', () => {
  // Two inputs is `kscan`'s shape (§06: "a Markov kernel `(state, x) ->
  // measure_over_state`"), not markovchain's.
  const msgs = errorsOf(`
k2 = kernelof(Normal(mu = a, sigma = b), a = a, b = b)
traj = markovchain(k2, 0.0, 3)
`);
  assert.ok(msgs.some((m) => /markovchain: .*has 2 inputs/.test(m)),
    'expected a located arity diagnostic, got ' + JSON.stringify(msgs));
});

test('markovchain: a non-scalar-distribution step kernel is refused, not silent',
  () => {
  // §06 also admits record states (whose trajectories are tables) and
  // composite step bodies. This engine lowers a scalar-distribution step only,
  // and says so rather than leaving the binding derivation-less and quiet.
  const msgs = errorsOf(`
inner = fn(lawof(iid(Normal(mu = _, sigma = 1.0), 3)))
traj = markovchain(inner, 0.0, 2)
`);
  assert.ok(msgs.some((m) => /markovchain: .*not a sampleable distribution/.test(m)),
    'expected a located scope refusal, got ' + JSON.stringify(msgs));
});

test('markovchain: a non-integer n is a located error', () => {
  const msgs = errorsOf(`
f = fn(Normal(mu = _, sigma = 1.0))
traj = markovchain(f, 0.0, 2.5)
`);
  assert.ok(msgs.some((m) => /markovchain: arg 3 .*positive integer/.test(m)),
    'expected a located n diagnostic, got ' + JSON.stringify(msgs));
});
