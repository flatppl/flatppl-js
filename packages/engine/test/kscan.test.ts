'use strict';

// =====================================================================
// kscan.test.ts — spec §06 `kscan(kernel, init, xs)`
// =====================================================================
//
// §06 dependent composition, verbatim: "`kernel` is a Markov kernel
// `(state, x) -> measure_over_state`; step $i$ is $\text{traj}_i \sim
// \kappa(\text{traj}_{i-1}, \text{xs}_i)$ with $\text{traj}_0 = \text{init}$.
// Trajectories have length `lengthof(xs)`. As with `markovchain`, `init` is a
// value in the state space and not part of the trajectory."
//
// Every number here is pinned against a CLOSED FORM, never against an engine
// output and never against `markovchain`'s own output. The step kernel is
// §06's own example — Brownian motion with variable timesteps,
// x_k = x_{k-1} + N(0, sqrt(2 D dt_k)) — whose moments are exact:
//
//   Var(traj[k])          = Σ_{j ≤ k} 2 D dt_j
//   Cov(traj[j], traj[k]) = Var(traj[min(j,k)])
//
// The per-step VARIANCE differs from step to step, which is the whole point of
// the op: a `markovchain` cannot express it, and an implementation that fed
// `xs` in the wrong order, or fed the same element every step, would miss
// these numbers. The density is exact (no MC): §06 makes `init` a value, so
// logdensityof is the sum of exactly `lengthof(xs)` Gaussian transition
// log-densities, hand-summed below with a DIFFERENT sigma per term.

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

/** Error diagnostics from analysis + derivation building, no materialisation. */
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
// Variate shape and length — §06's `lengthof(xs)`, init excluded
// =====================================================================

test('kscan: the variate is an ARRAY of lengthof(xs) states, init excluded',
  async () => {
  // Three exogenous inputs, so three trajectory elements — not four, and not
  // one per something else. traj[1] is one STEP from init, so its variance is
  // the FIRST timestep's, not zero.
  const ctx = makeCtx(`
dts = [0.5, 2.0, 1.0]
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, dts)
`);
  const m = await ctx.getMeasure('traj');
  assert.equal(m.shape, 'array', 'kscan materialises an array measure');
  assert.deepEqual(m.dims, [3], 'per-atom dims are [lengthof(xs)]');
  const c1 = column(m, 3, 1);
  assert.ok(Math.abs(variance(c1) - 0.25) < 0.02,
    `traj[1] var ${variance(c1)} vs one step of sigma = 0.5`);
});

test('kscan: the derivation is its own kind, not a markovchain', () => {
  // The two ops share every execution path, so the descriptor is the only
  // place the exogenous column is recorded — a classification that fell
  // through to `markovchain` would silently drop `xs`.
  const lifted = processSource(`
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, [0.5, 1.0])
`);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const d: any = built.derivations.traj;
  assert.equal(d.kind, 'kscan');
  assert.equal(d.n, 2, 'the length is lengthof(xs)');
  assert.ok(d.xsIR, 'the exogenous array rides on the derivation');
  // The lowerer renames a lambda's params, so match the source name inside it.
  assert.match(d.step.xParam, /dt/, 'the second kernel input is the exogenous one');
});

// =====================================================================
// Moments — §06's own variable-timestep Brownian example, closed form
// =====================================================================

test('kscan: variable-timestep Brownian variance is the CUMULATIVE 2·D·dt sum',
  async () => {
  // §06's example spelling verbatim. Var(traj[k]) = Σ_{j ≤ k} 2·D·dt_j: five
  // different steps, five different increments, so the whole exogenous column
  // is under test in order. Feeding one element to every step, or reversing
  // the column, breaks these five numbers together.
  const D = 4.1;
  const dts = [0.01, 0.02, 0.015, 0.018, 0.012];
  const ctx = makeCtx(`
D = 4.1    % Diffusion constant
dts = [0.01, 0.02, 0.015, 0.018, 0.012]  % Time steps
f_step = (x, dt) -> Normal(x, sqrt(2*D * dt))
traj = kscan(f_step, 0.0, dts)
`);
  const m = await ctx.getMeasure('traj');
  assert.deepEqual(m.dims, [5]);
  let want = 0;
  for (let k = 1; k <= 5; k++) {
    want += 2 * D * dts[k - 1];
    const got = variance(column(m, 5, k));
    assert.ok(Math.abs(got - want) / want < 0.03,
      `Var(traj[${k}]) ${got} vs closed form ${want}`);
  }
});

test('kscan: Cov(traj[j], traj[k]) is Var(traj[min(j,k)]) (closed form)',
  async () => {
  // The increments are independent, so covariance is the shared prefix's
  // variance. This is what makes the trajectory a SCAN rather than an
  // independent product: an iid lowering would give zero off-diagonal.
  const dts = [0.5, 2.0, 1.0];
  const ctx = makeCtx(`
dts = [0.5, 2.0, 1.0]
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, dts)
`);
  const m = await ctx.getMeasure('traj');
  const cols = [1, 2, 3].map((j) => column(m, 3, j));
  const prefix = [0.25, 0.25 + 4.0, 0.25 + 4.0 + 1.0];
  for (let j = 1; j <= 3; j++) {
    for (let k = j + 1; k <= 3; k++) {
      const got = covariance(cols[j - 1], cols[k - 1]);
      const want = prefix[j - 1];
      assert.ok(Math.abs(got - want) / want < 0.05,
        `Cov(traj[${j}], traj[${k}]) ${got} vs closed form ${want}`);
    }
  }
});

test('kscan: a drawn init contributes its own variance to every element',
  async () => {
  // init is a VALUE in the state space (§06), which here is a drawn one. It
  // shifts the whole trajectory, so Var(traj[k]) = sigma_init² + Σ 2·D·dt_j.
  const ctx = makeCtx(`
x0 ~ Normal(mu = 0.0, sigma = 1.5)
dts = [0.5, 2.0]
step_kernel = (x, dt) -> Normal(x, dt)
traj = kscan(step_kernel, x0, dts)
`);
  const m = await ctx.getMeasure('traj');
  const want = [1.5 * 1.5 + 0.25, 1.5 * 1.5 + 0.25 + 4.0];
  for (let k = 1; k <= 2; k++) {
    const got = variance(column(m, 2, k));
    assert.ok(Math.abs(got - want[k - 1]) / want[k - 1] < 0.04,
      `Var(traj[${k}]) ${got} vs closed form ${want[k - 1]}`);
  }
});

// =====================================================================
// Density — §06's constituent conditionals, exact
// =====================================================================

test('kscan: logdensityof sums the transitions, one sigma per exogenous input',
  async () => {
  // Three DISTINCT timesteps, so three distinct transition sigmas. Only the
  // right element at the right step reproduces this sum, and `init` adds no
  // base term because §06 makes it a value.
  const ctx = makeCtx(`
dts = [0.5, 2.0, 1.0]
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, dts)
lp = logdensityof(traj, [0.3, 0.5, 0.9])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.3, 0.0, 0.5) + normLogpdf(0.5, 0.3, 2.0)
    + normLogpdf(0.9, 0.5, 1.0);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `kscan logp ${m.samples[0]} vs closed form ${want}`);
});

test('kscan: reversing xs reverses the transition sigmas', async () => {
  // The same trajectory scored under the reversed exogenous column must give
  // the reversed sum, and the two must DIFFER — an implementation that ignored
  // the index would pass the previous test and fail this one.
  const forward = makeCtx(`
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, [0.5, 2.0, 1.0])
lp = logdensityof(traj, [0.3, 0.5, 0.9])
`, 1000);
  const reverse = makeCtx(`
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 0.0, [1.0, 2.0, 0.5])
lp = logdensityof(traj, [0.3, 0.5, 0.9])
`, 1000);
  const fwd = (await forward.getMeasure('lp')).samples[0];
  const rev = (await reverse.getMeasure('lp')).samples[0];
  const wantFwd = normLogpdf(0.3, 0.0, 0.5) + normLogpdf(0.5, 0.3, 2.0)
    + normLogpdf(0.9, 0.5, 1.0);
  const wantRev = normLogpdf(0.3, 0.0, 1.0) + normLogpdf(0.5, 0.3, 2.0)
    + normLogpdf(0.9, 0.5, 0.5);
  assert.ok(Math.abs(fwd - wantFwd) < 1e-12, `forward ${fwd} vs ${wantFwd}`);
  assert.ok(Math.abs(rev - wantRev) < 1e-12, `reversed ${rev} vs ${wantRev}`);
  assert.ok(Math.abs(fwd - rev) > 0.1, 'the two orders must not coincide');
});

test('kscan: a non-zero init enters step 1 and only step 1', async () => {
  // Shifting init changes exactly the FIRST transition's mean. Pins the same
  // off-by-one markovchain pins, now with the exogenous column present.
  const ctx = makeCtx(`
f = kernelof(Normal(mu = p, sigma = d), p = p, d = d)
traj = kscan(f, -1.5, [0.5, 2.0])
lp = logdensityof(traj, [0.4, 1.1])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.4, -1.5, 0.5) + normLogpdf(1.1, 0.4, 2.0);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `kscan logp ${m.samples[0]} vs closed form ${want}`);
});

test('kscan: lengthof(xs) = 1 is one step from init, density is one transition',
  async () => {
  // §06 fixes the length as `lengthof(xs)` with no lower bound above one.
  const ctx = makeCtx(`
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 2.0, [0.75])
lp = logdensityof(traj, [2.5])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(2.5, 2.0, 0.75);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `kscan logp ${m.samples[0]} vs closed form ${want}`);
  const sampled = await makeCtx(`
f = (x, dt) -> Normal(x, dt)
traj = kscan(f, 2.0, [0.75])
`).getMeasure('traj');
  assert.deepEqual(sampled.dims, [1]);
});

test('kscan: a trajectory inside a record law scores exactly', async () => {
  // `x ~ kscan(…)` makes `x` a variate, and the density inlines into the
  // enclosing law — the indirection that made `selfThreaded` ride on the node
  // rather than on the derivation kind. Exact, no MC.
  const ctx = makeCtx(`
step = (p, d) -> Normal(p, d)
x ~ kscan(step, 0.0, [0.5, 2.0])
model = lawof(record(x = x))
lp = logdensityof(model, record(x = [0.3, 0.5]))
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.3, 0.0, 0.5) + normLogpdf(0.5, 0.3, 2.0);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `kscan-in-record logp ${m.samples[0]} vs closed form ${want}`);
});

test('kscan: the exogenous column reaches a parameter EXPRESSION, not just a param',
  async () => {
  // §06's own example puts `dt` inside `sqrt(2*D * dt)`, so the substitution
  // has to reach into an arithmetic subexpression, and the sampler and the
  // density have to agree about the value it produces there.
  const D = 4.1;
  const dts = [0.01, 0.02];
  const ctx = makeCtx(`
D = 4.1
f_step = (x, dt) -> Normal(x, sqrt(2*D * dt))
traj = kscan(f_step, 0.0, [0.01, 0.02])
lp = logdensityof(traj, [0.1, 0.05])
`, 1000);
  const m = await ctx.getMeasure('lp');
  const want = normLogpdf(0.1, 0.0, Math.sqrt(2 * D * dts[0]))
    + normLogpdf(0.05, 0.1, Math.sqrt(2 * D * dts[1]));
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `kscan logp ${m.samples[0]} vs closed form ${want}`);
});

// =====================================================================
// Static refusals — §06 shapes this engine does not lower
// =====================================================================

test('kscan: a single-input kernel is a located error', () => {
  // One input is `markovchain`'s shape (§06: "a Markov kernel `(state) ->
  // measure_over_state`"), not kscan's.
  const msgs = errorsOf(`
f = fn(Normal(mu = _, sigma = 1.0))
traj = kscan(f, 0.0, [0.5, 1.0])
`);
  assert.ok(msgs.some((m) => /kscan: .*has 1 inputs/.test(m)),
    'expected a 1-input refusal, got ' + JSON.stringify(msgs));
});

test('kscan: a non-kernel first argument is a located error', () => {
  const msgs = errorsOf('traj = kscan(Normal(mu = 0.0, sigma = 1.0), 0.0, [0.5])\n');
  assert.ok(msgs.some((m) => /kscan: arg 1 expects a Markov kernel/.test(m)),
    'expected a kernel-arg type error, got ' + JSON.stringify(msgs));
});

test('kscan: a scalar `xs` is a located error, not a length-1 trajectory', () => {
  // §06 gives the length as `lengthof(xs)`, so a scalar third argument is
  // markovchain's `n` spelling and belongs to that op.
  const msgs = errorsOf(`
f = (x, d) -> Normal(x, d)
traj = kscan(f, 0.0, 3)
`);
  assert.ok(msgs.some((m) => /kscan: arg 3 \(`xs`\) expects an array/.test(m)),
    'expected an xs type error, got ' + JSON.stringify(msgs));
});

test('kscan: a RANDOM `xs` is refused — exogenous inputs are data', () => {
  const msgs = errorsOf(`
f = (x, d) -> Normal(x, d)
xs = iid(Normal(mu = 1.0, sigma = 0.1), 3)
traj = kscan(f, 0.0, xs)
`);
  assert.ok(msgs.some((m) => /kscan: arg 3 \(`xs`\) expects an array/.test(m)),
    'expected an xs type error, got ' + JSON.stringify(msgs));
  assert.ok(msgs.some((m) => /kscan: `xs` length does not resolve/.test(m)),
    'and the length refusal names what is missing, got ' + JSON.stringify(msgs));
});

test('kscan: a non-scalar-distribution step kernel is refused, not silent', () => {
  // §06 admits composite step bodies; this engine lowers a scalar step only,
  // and says so rather than leaving the binding without a derivation.
  const msgs = errorsOf(`
inner = (x, d) -> joint(a = Normal(x, d), b = Normal(x, d))
traj = kscan(inner, 0.0, [0.5])
`);
  assert.ok(msgs.some((m) => /kscan: .*not a sampleable distribution/.test(m)),
    'expected a composite-body refusal, got ' + JSON.stringify(msgs));
});

test('kscan: a kernel argument naming no binding is a located error', () => {
  const msgs = errorsOf('traj = kscan(nope, 0.0, [0.5])\n');
  assert.ok(msgs.some((m) => /kscan: `nope` names no binding/.test(m)),
    'expected a no-binding refusal, got ' + JSON.stringify(msgs));
});

test('kscan: the wrong argument count is a located error', () => {
  const msgs = errorsOf('f = (x, d) -> Normal(x, d)\ntraj = kscan(f, 0.0)\n');
  assert.ok(msgs.some((m) => /kscan expects 3 positional argument/.test(m)),
    'expected an arity error, got ' + JSON.stringify(msgs));
});

// =====================================================================
// Runtime refusals — §06 state shapes this engine does not lower
// =====================================================================

test('kscan: a RECORD init — §06\'s table-trajectory case — is refused',
  async () => {
  // §06: "If `init` and `traj[i]` are records, then the trajectories are
  // tables, not arrays" (stated for markovchain, and kscan is that scan with
  // an exogenous input). This engine lowers a scalar state, and refuses the
  // record state with a message instead of reading a field.
  const ctx = makeCtx(`
f = (x, d) -> Normal(x, d)
traj = kscan(f, record(a = 1.0), [0.5, 1.0])
`, 500);
  await assert.rejects(() => ctx.getMeasure('traj'),
    /kscan: `init` resolved to object \(expected a scalar or one value per atom\)/);
});

test('kscan: an ARRAY-valued init is refused, not read element-wise', async () => {
  // A length-1 array must not be silently read as its element, and a longer
  // one is a vector state §06 admits and this engine does not lower.
  for (const init of ['[2.0]', '[1.0, 2.0]']) {
    const ctx = makeCtx(`
f = (x, d) -> Normal(x, d)
traj = kscan(f, ${init}, [0.5, 1.0])
`, 500);
    await assert.rejects(() => ctx.getMeasure('traj'),
      /kscan: `init` resolved to \[\d+\] \(expected a scalar or one value per atom\)/,
      `init = ${init} must be refused`);
  }
});

test('kscan: a vector step parameter is refused at the step it breaks',
  async () => {
  const ctx = makeCtx(`
f = (x, d) -> Normal(x, [1.0, 2.0])
traj = kscan(f, 0.0, [0.5, 1.0])
`, 500);
  await assert.rejects(() => ctx.getMeasure('traj'),
    /kscan: step 1 param 'sigma' resolved to \[2\] \(expected a scalar or \[500\]\)/);
});
