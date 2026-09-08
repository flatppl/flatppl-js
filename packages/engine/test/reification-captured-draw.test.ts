'use strict';

// Spec §04 *Captured draws*: a reified callable may reference `draw` nodes of
// the enclosing graph. Such a captured draw "remains a shared ancestor of the
// reification and of the enclosing graph, with a single realisation", and the
// callable is "conditional on that realisation: it is neither resampled per
// call nor marginalized, so any number of calls uses the one realisation".
//
// The phase is where that costs determinism: "The phase of a reification
// follows the reified sub-graph by the same ancestor rule as any other binding
// [...] it is stochastic when the sub-graph holds a captured `draw` whose phase
// no `lawof` node absorbs."
//
// Before this the analyzer typed EVERY reification `fixed`, so a callable over
// a drawn ancestor claimed determinism it does not have. These tests pin both
// halves: the phase, and the one-realisation numbers the phase describes.
//
// Every number here is a closed form. Nothing is compared against the engine's
// own earlier output.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

/** Inferred phase of one binding. */
function phaseOf(src: string, name: string): string {
  const { bindings, diagnostics } = processSource(src);
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [], 'source analyses clean');
  const b = bindings.get(name);
  assert.ok(b, `no binding named ${name}`);
  return b.phase;
}

const mean = (a: any) => Array.from(a as number[]).reduce((s, x) => s + x, 0) / a.length;

// ---------------------------------------------------------------------
// The §02 overview witness: a boundary-less `functionof` over a value
// descending from a drawn systematic.
// ---------------------------------------------------------------------

const WITNESS = `
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
resolution = 2.5 * exp(0.12 * raw_syst)
f = functionof(resolution)
y = f()
`;

test('a functionof that captures a draw is stochastic, and so is its call', () => {
  const { bindings } = processSource(WITNESS);
  assert.equal(bindings.get('f').phase, 'stochastic');
  assert.equal(bindings.get('y').phase, 'stochastic');
});

test('a captured draw has ONE realisation per replicate', async () => {
  const n = 20000;
  const { ctx } = makeMatCtx(WITNESS, { sampleCount: n, rootSeed: 4242 });
  const y = (await ctx.getMeasure('y')).samples;
  const raw = (await ctx.getMeasure('raw_syst')).samples;
  assert.equal(y.length, n);
  // The identity that "conditional on that realisation" means: within one
  // joint realisation `y` is the deterministic function of THAT replicate's
  // `raw_syst`. Exact, atom by atom — a resample per call could not hold it.
  let worst = 0;
  for (let i = 0; i < n; i++) {
    worst = Math.max(worst, Math.abs(y[i] - 2.5 * Math.exp(0.12 * raw[i])));
  }
  assert.equal(worst, 0, 'y_i = 2.5 * exp(0.12 * raw_syst_i) exactly');
  // E[y] = 2.5 * exp(0.12^2 / 2) for raw_syst ~ Normal(0, 1) (lognormal mean).
  // sd(y) = 2.5 * sqrt(exp(2 s^2) - exp(s^2)) = 0.30327 with s = 0.12, so the
  // standard error over n replicates is 0.00214 — the bound is ~5 of those.
  assert.ok(Math.abs(mean(y) - 2.5 * Math.exp(0.0072)) < 0.011,
    `E[y] = ${mean(y)}, oracle ${2.5 * Math.exp(0.0072)}`);
});

test('every call of a capturing callable sees the same realisation', async () => {
  const n = 4096;
  const { ctx } = makeMatCtx(`
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
f = functionof(2.5 * exp(0.12 * raw_syst))
a = f()
b = f()
d = a - b
`, { sampleCount: n, rootSeed: 606 });
  const d = (await ctx.getMeasure('d')).samples;
  // Two calls, one realisation: the difference is identically zero. Were the
  // callable resampled per call, d would be the difference of two independent
  // lognormals with sd 0.30327 * sqrt(2) = 0.429.
  for (let i = 0; i < n; i++) assert.equal(d[i], 0, `call ${i} differs`);
});

// ---------------------------------------------------------------------
// The cut rules: what does NOT capture.
// ---------------------------------------------------------------------

test('a boundary-named draw is not captured, so the callable stays fixed', () => {
  // §04: `functionof` "substitutes each boundary node `a` with an input node
  // `elementof(valueset(a))`", and the substitution precedes the ancestor
  // trace — so `h` is the conditional kernel over a declared input.
  assert.equal(phaseOf(`
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
resolution = 2.5 * exp(0.12 * raw_syst)
h = functionof(resolution, raw_syst = raw_syst)
`, 'h'), 'fixed');
});

test('functionof(lawof(...)) stays fixed and marginal', async () => {
  // §04 "Phase of the reified law": `lawof` "absorbs stochasticity into the
  // reified law rather than propagating it outward", so nothing is captured.
  const src = `
raw ~ Normal(mu = 0.0, sigma = 1.0)
g = functionof(lawof(2.5 * exp(0.12 * raw)))
mm = g()
`;
  const { bindings } = processSource(src);
  assert.equal(bindings.get('g').phase, 'fixed');
  assert.equal(bindings.get('mm').phase, 'fixed');
  // And the reified law is the MARGINAL pushforward of raw's law, not a
  // point mass at one realisation: y = 2.5 exp(0.12 Z) with Z ~ Normal(0, 1)
  // has mean 2.5 exp(s^2/2) and variance 6.25 (exp(2 s^2) - exp(s^2)),
  // s = 0.12 — mean 2.518065, variance 0.0919686, sd 0.303263.
  const n = 20000;
  const { ctx } = makeMatCtx(src, { sampleCount: n, rootSeed: 1234 });
  const s = (await ctx.getMeasure('mm')).samples;
  const m = mean(s);
  const varS = mean(Float64Array.from(s, (x: number) => (x - m) ** 2));
  assert.ok(Math.abs(m - 2.5 * Math.exp(0.0072)) < 0.011, `mean ${m}`);
  // Standard error of a variance estimate here is about 0.0009; allow ~5.
  assert.ok(Math.abs(varS - 6.25 * (Math.exp(0.0288) - Math.exp(0.0144))) < 0.005,
    `var ${varS}, oracle ${6.25 * (Math.exp(0.0288) - Math.exp(0.0144))}`);
});

test('kernelof is never stochastic: its body sits under an implicit lawof', () => {
  // §04 *Kernels and `kernelof`* makes `kernelof(x, kwargs…)` equivalent to
  // `functionof(lawof(x), kwargs…)`, so every draw it reaches is absorbed.
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
x ~ Normal(mu = raw, sigma = 1.0)
k = kernelof(record(x = x))
`, 'k'), 'fixed');
  // Nested inside a functionof body, the same absorption applies.
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
x ~ Normal(mu = raw, sigma = 1.0)
kk = functionof(kernelof(record(x = x)))
`, 'kk'), 'fixed');
});

test('a parameterized ancestor is traced to an input, not captured', () => {
  // §04: "a parameterized ancestor of the reified sub-graph is traced back to
  // an input", so an `elementof` leaf leaves the callable fixed.
  assert.equal(phaseOf('p = functionof(elementof(reals))', 'p'), 'fixed');
  assert.equal(phaseOf('c = 3.0\nq = functionof(c + c)', 'q'), 'fixed');
});

// ---------------------------------------------------------------------
// Spellings that reach the same draw.
// ---------------------------------------------------------------------

test('a draw written inline in the body is captured', () => {
  assert.equal(phaseOf('w = functionof(2.0 * draw(Normal(mu = 0.0, sigma = 1.0)))', 'w'),
    'stochastic');
});

test('a nested reification is walked into under both boundaries', () => {
  // The inner boundary cuts what it names, and the outer's formals ride in.
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
outer = functionof(functionof(2.0 * raw))
`, 'outer'), 'stochastic');
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
closed = functionof(functionof(2.0 * raw, raw = raw))
`, 'closed'), 'fixed');
});

test('applying a capturing callable inside a body captures too', () => {
  // §04 gives a reification "the same ancestor rule as any other binding", and
  // outside a reification `z = inner()` is stochastic for this reason.
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
inner = functionof(2.0 * raw)
outer = functionof(inner())
`, 'outer'), 'stochastic');
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
inner = functionof(2.0 * raw, raw = raw)
outer = functionof(inner(1.0))
`, 'outer'), 'fixed');
});

test('a name cut by an inner boundary is still captured at the outer level', () => {
  // `o` names `other`, so the `other` path is substituted away — but nothing
  // names `raw`, and `o` reaches it through `mid`. The inner `inner` DOES name
  // `raw`, so a walk that remembers "mid captures nothing" from inside `inner`
  // and reuses it at `o`'s own level answers `fixed` here, which is wrong.
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
other ~ Normal(mu = 0.0, sigma = 1.0)
mid = 2.0 * raw
inner = functionof(mid + other, raw = raw)
o = functionof(inner(1.0) + mid, other = other)
`, 'o'), 'stochastic');
});

test('an fn body captures the same way a functionof body does', () => {
  assert.equal(phaseOf(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
u = fn(2.0 * _ + raw)
`, 'u'), 'stochastic');
});

// ---------------------------------------------------------------------
// Lambdas: the shipped shapes.
// ---------------------------------------------------------------------

test('a lambda capturing a draw shares one realisation across a broadcast', async () => {
  const n = 4096;
  const src = `
lam ~ Uniform(interval(1.0, 2.0))
g = x -> x * lam
v = g.([1.0, 2.0, 3.0])
`;
  assert.equal(phaseOf(src, 'g'), 'stochastic');
  const { ctx } = makeMatCtx(src, { sampleCount: n, rootSeed: 99 });
  const lam = (await ctx.getMeasure('lam')).samples;
  const v = (await ctx.getMeasure('v')).samples;
  assert.equal(v.length, 3 * n);
  // One realisation of `lam` per replicate, shared by all three broadcast
  // members: v[i, k] = (k + 1) * lam_i exactly. A per-call resample would
  // break the ratio v[i, 2] / v[i, 1] = 2.
  let worst = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      worst = Math.max(worst, Math.abs(v[3 * i + k] - (k + 1) * lam[i]));
    }
  }
  assert.equal(worst, 0, 'v[i, k] = (k + 1) * lam_i exactly');
  // E[lam] = 1.5 for Uniform(1, 2); sd 1/sqrt(12) = 0.2887, so the standard
  // error over 4096 replicates is 0.0045 — the bound is ~5 of those.
  assert.ok(Math.abs(mean(lam) - 1.5) < 0.023, `E[lam] = ${mean(lam)}`);
});

// ---------------------------------------------------------------------
// The AR-1 shape: `markovchain` over a step kernel with a drawn scale.
// This is `flatppl-examples/examples/ar1-noise-estimation.flatppl`.
// ---------------------------------------------------------------------

test('an AR-1 step kernel capturing a drawn scale is stochastic', () => {
  assert.equal(phaseOf(`
drift = 0.4
sigma_step ~ normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf)))
step_kernel = prev -> Normal(prev + drift, sigma_step)
x ~ markovchain(step_kernel, 0.0, 3)
`, 'step_kernel'), 'stochastic');
});

test('an AR-1 chain scores n step terms at ONE shared scale plus one prior', async () => {
  const { ctx } = makeMatCtx(`
sigma_step ~ normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf)))
step_kernel = prev -> Normal(prev + 0.0, sigma_step)
x ~ markovchain(step_kernel, 0.0, 3)
prior = lawof(record(sigma_step = sigma_step))
fk = kernelof(record(x = x), sigma_step = sigma_step)
L = likelihoodof(fk, record(x = [0.1, 0.2, 0.3]))
post = bayesupdate(L, prior)
lp = logdensityof(post, record(sigma_step = 0.7))
`, { sampleCount: 1, rootSeed: 1 });
  const got = (await ctx.getMeasure('lp')).value.data[0];
  // Closed form. The trajectory [0.1, 0.2, 0.3] off x_init = 0 with zero
  // drift gives three increments of 0.1, each scored Normal(0, s):
  //   3 (-log sqrt(2 pi) - log s) - 3 (0.1)^2 / (2 s^2)
  // at ONE s, plus the half-Cauchy prior 2 / (pi (1 + s^2)) scored ONCE.
  const s = 0.7;
  const steps = 3 * (-0.5 * Math.log(2 * Math.PI) - Math.log(s))
    - 3 * 0.01 / (2 * s * s);
  const prior = Math.log(2 / (Math.PI * (1 + s * s)));
  assert.ok(Math.abs(got - (steps + prior)) < 1e-12,
    `lp = ${got}, oracle ${steps + prior}`);
});

test('an AR-1 trajectory samples every step at the replicate\'s own scale', async () => {
  const n = 6000;
  const { ctx } = makeMatCtx(`
sigma_step ~ Uniform(interval(0.5, 1.5))
step_kernel = prev -> Normal(prev + 0.0, sigma_step)
x ~ markovchain(step_kernel, 0.0, 3)
`, { sampleCount: n, rootSeed: 31337 });
  const xs = (await ctx.getMeasure('x')).samples;
  const sig = (await ctx.getMeasure('sigma_step')).samples;
  assert.equal(xs.length, 3 * n);
  // Conditional on the replicate's own sigma_step, each increment is
  // Normal(0, sigma_step), so ((x_k - x_{k-1}) / sigma_step)^2 has mean 1.
  // A per-step resample of sigma_step would leave the standardised squares
  // with mean E[sigma'^2] / sigma^2 averaged over an independent sigma',
  // which for Uniform(0.5, 1.5) is E[sigma'^2] E[1/sigma^2] = 1.0833 *
  // 1.3333 = 1.4444 — far outside the band below.
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let prev = 0;
    for (let k = 0; k < 3; k++) {
      sum += ((xs[3 * i + k] - prev) / sig[i]) ** 2;
      prev = xs[3 * i + k];
    }
  }
  const stat = sum / (3 * n);
  // Var of a squared standard normal is 2, so the standard error over 18000
  // terms is sqrt(2 / 18000) = 0.0105 — the bound is ~5 of those.
  assert.ok(Math.abs(stat - 1) < 0.055,
    `mean standardised square = ${stat}, oracle 1`);
});
