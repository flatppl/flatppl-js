'use strict';

// =====================================================================
// markovchain-posterior.test.ts — scoring a markovchain trajectory when its
// density is INLINED into an enclosing measure
// =====================================================================
//
// `markovchain`'s density lowers to a positional joint whose step j reads
// `s{j-1}`, the previous OBSERVED element that
// `walkJointFieldsOrPositional` threads as it consumes the value. Those names
// are self-threaded: there is nothing to feed them, so clm's `_enumerateInputs`
// must exclude them from the ⊆ (every-body-self-ref-is-declared) set.
//
// The exclusion was keyed on the ENCLOSING derivation's kind, which only
// matches when the trajectory binding is lowered under its own name. Any
// indirection defeats it — and `x ~ markovchain(…)` already introduces one,
// since `~` makes `x` an alias to an anon markovchain binding. So
// `lawof(record(x = x))` and a `likelihoodof` over it both threw
// CLM_SUBSET_VIOLATION on [s0 … s{n-2}], which `derivationRefsValid` turned
// into a pruned derivation and the flat "produced no value" diagnostic.
//
// Every density number here is a closed form. For x_k ~ Normal(x_{k-1} + d, s)
// with x_0 = 0 the increments are n independent Normal(0, s) draws, so
//
//   logdensityof(traj, xs) = -n/2·log(2π) - n·log s - S/(2 s²),
//   S = Σ_k (x_k - x_{k-1} - d)²
//
// which is exact — no Monte Carlo on the likelihood path.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED11;

function makeCtx(source: any, sampleCount: number) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    errors: (lifted.diagnostics || []).concat(built.diagnostics || [])
      .filter((d: any) => d.severity === 'error').map((d: any) => d.message),
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
    sampleCount, rootSeed: ROOT_SEED,
  };
  return ctx;
}

/** -n/2·log(2π) - n·log s - S/(2 s²) for the AR-1 chain (spec §06). */
function ar1Logdensity(xs: number[], init: number, drift: number, sigma: number) {
  let S = 0;
  let prev = init;
  for (const x of xs) { const r = x - prev - drift; S += r * r; prev = x; }
  const n = xs.length;
  return -0.5 * n * Math.log(2 * Math.PI) - n * Math.log(sigma) - S / (2 * sigma * sigma);
}

const X_DATA = [0.4, 0.9, 1.1, 1.9, 2.2];
const DATA_SRC = 'x_data = [0.4, 0.9, 1.1, 1.9, 2.2]\n';

// =====================================================================
// The inlined density — the seam that was refusing
// =====================================================================

test('markovchain: a trajectory inside lawof(record(...)) scores exactly',
  async () => {
  // `x ~ markovchain(...)` makes `x` a variate, and `lawof(record(x = x))` is
  // its record law (spec §06 "Equivalent record law"). Scoring it inlines the
  // trajectory density, so the `s{j}` positions appear in a body lowered under
  // a DIFFERENT name than the markovchain binding's. Exact, no MC.
  const ctx = makeCtx(DATA_SRC + `
sig = 0.75
step = prev -> Normal(prev, sig)
x ~ markovchain(step, 0.0, 5)
j = lawof(record(x = x))
lp = logdensityof(j, record(x = x_data))
`, 1000);
  assert.deepEqual(ctx.errors, [], 'no diagnostics: the body must lower');
  const m = await ctx.getMeasure('lp');
  const want = ar1Logdensity(X_DATA, 0.0, 0.0, 0.75);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `inlined trajectory logp ${m.samples[0]} vs closed form ${want}`);
});

test('markovchain: a likelihood over a trajectory scores at 3 sigma grid points',
  async () => {
  // `logdensityof(likelihoodof(K, obs), θ)` is the standalone
  // likelihood_density path (audit H2): the trajectory density is inlined into
  // the kernel body and scored at a given θ. Three points across the relevant
  // range of the scale, each against the closed form — one point could agree
  // by accident on a constant, three cannot.
  //
  // The MLE for this data is sqrt(S/n), so the grid straddles it.
  for (const sigma of [0.35, 0.6, 1.25]) {
    const ctx = makeCtx(DATA_SRC + `
sig ~ normalize(truncate(Cauchy(0, 1), interval(0, inf)))
step = prev -> Normal(prev, sig)
x ~ markovchain(step, 0.0, 5)
fk = kernelof(record(x = x), sig = sig)
L = likelihoodof(fk, record(x = x_data))
lp = logdensityof(L, record(sig = ${sigma}))
`, 1000);
    assert.deepEqual(ctx.errors, [], `no diagnostics at sigma = ${sigma}`);
    const m = await ctx.getMeasure('lp');
    const want = ar1Logdensity(X_DATA, 0.0, 0.0, sigma);
    assert.ok(Math.abs(m.samples[0] - want) < 1e-10,
      `likelihood logp at sigma=${sigma}: ${m.samples[0]} vs closed form ${want}`);
  }
});

test('markovchain: a drift term rides into every increment', async () => {
  // The step kernel's mean is `prev + drift`, so the closed form shifts every
  // residual. Pinned separately: a drift dropped from the inlined body would
  // still agree with a no-drift oracle, and this is the model the acceptance
  // example uses.
  const ctx = makeCtx(DATA_SRC + `
sig = 0.75
drift = 0.4
step = prev -> Normal(prev + drift, sig)
x ~ markovchain(step, 0.0, 5)
j = lawof(record(x = x))
lp = logdensityof(j, record(x = x_data))
`, 1000);
  assert.deepEqual(ctx.errors, []);
  const m = await ctx.getMeasure('lp');
  const want = ar1Logdensity(X_DATA, 0.0, 0.4, 0.75);
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `drifted trajectory logp ${m.samples[0]} vs closed form ${want}`);
});

// =====================================================================
// The posterior — a hand-derived conjugate-free check
// =====================================================================

test('markovchain: the AR-1 scale posterior matches a hand-integrated oracle',
  async () => {
  // Half-Cauchy prior on the scale, AR-1 likelihood. The posterior density is
  //   p(s | x) ∝ (2/(π(1+s²))) · s^-n · exp(-S/(2s²))
  // and the reference moments below are that expression integrated on a fine
  // grid — a deterministic quadrature done HERE, so the engine's importance
  // sampler is checked against maths rather than against itself.
  const S = (() => {
    let s = 0, prev = 0;
    for (const x of X_DATA) { s += (x - prev) * (x - prev); prev = x; }
    return s;
  })();
  const n = X_DATA.length;
  // Trapezoid over a range that covers the mass for this S (MLE = sqrt(S/n)).
  let z = 0, m1 = 0, m2 = 0;
  const lo = 1e-4, hi = 8.0, steps = 400000;
  const h = (hi - lo) / steps;
  for (let i = 0; i <= steps; i++) {
    const s = lo + i * h;
    const logp = -n * Math.log(s) - S / (2 * s * s) - Math.log(1 + s * s);
    const w = Math.exp(logp) * (i === 0 || i === steps ? 0.5 : 1) * h;
    z += w; m1 += w * s; m2 += w * s * s;
  }
  const refMean = m1 / z;
  const refSd = Math.sqrt(m2 / z - refMean * refMean);

  const ctx = makeCtx(DATA_SRC + `
sig ~ normalize(truncate(Cauchy(0, 1), interval(0, inf)))
prior = lawof(record(sig = sig))
step = prev -> Normal(prev, sig)
x ~ markovchain(step, 0.0, 5)
fk = kernelof(record(x = x), sig = sig)
L = likelihoodof(fk, record(x = x_data))
posterior = bayesupdate(L, prior)
`, 200000);
  assert.deepEqual(ctx.errors, []);
  const post = await ctx.getMeasure('posterior');
  assert.ok(post.fields && post.fields.sig, 'posterior is a record measure over sig');
  const xs = post.fields.sig.samples;
  const lw = post.logWeights;
  assert.ok(lw, 'posterior atoms carry logWeights');
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  assert.ok(Number.isFinite(mx), 'the posterior log-weights must be finite');
  let sw = 0, sx = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = Math.exp(lw[i] - mx);
    sw += w; sx += w * xs[i]; sxx += w * xs[i] * xs[i];
  }
  const mean = sx / sw;
  const sd = Math.sqrt(sxx / sw - mean * mean);
  // Importance sampling with the prior as proposal: a half-Cauchy against a
  // posterior this concentrated has a modest ESS, so the budget is ~3%.
  assert.ok(Math.abs(mean - refMean) < 0.03 * refMean,
    `posterior mean ${mean} vs quadrature ${refMean}`);
  assert.ok(Math.abs(sd - refSd) < 0.10 * refSd,
    `posterior sd ${sd} vs quadrature ${refSd}`);
});

// =====================================================================
// The acceptance example, end to end
// =====================================================================

test('markovchain: examples/ar1-noise-estimation reproduces its in-file oracle',
  async () => {
  // The worked example's own frozen numbers (posterior mean 0.838310,
  // sd 0.054553), computed by an external oracle and recorded in the file.
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'ar1-noise-estimation.flatppl'), 'utf8');
  const ctx = makeCtx(src, 200000);
  assert.deepEqual(ctx.errors, [], 'the example must analyse clean');
  const post = await ctx.getMeasure('posterior');
  const xs = post.fields.sigma_step.samples;
  const lw = post.logWeights;
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  let sw = 0, sx = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = Math.exp(lw[i] - mx);
    sw += w; sx += w * xs[i]; sxx += w * xs[i] * xs[i];
  }
  const mean = sx / sw;
  const sd = Math.sqrt(sxx / sw - mean * mean);
  assert.ok(Math.abs(mean - 0.838310) < 0.004,
    `example posterior mean ${mean} vs in-file oracle 0.838310`);
  assert.ok(Math.abs(sd - 0.054553) < 0.004,
    `example posterior sd ${sd} vs in-file oracle 0.054553`);
});

// =====================================================================
// restrict — pinned as a PRE-EXISTING limitation, not a markovchain one
// =====================================================================

test('restrict over a record law refuses with a real message, markovchain or not',
  () => {
  // The brief asked whether `restrict(lawof(record(x = x)), x = x_data)` works
  // or refuses meaninglessly. It refuses, and the refusal is markovchain-
  // INDEPENDENT: an `iid` inner produces the identical message, so this is the
  // restrict/bayesupdate idiom's own limitation and not the trajectory's. The
  // control case is the point of this test — without it the same refusal would
  // read as a markovchain gap.
  const of = (chain: string) => {
    const lifted = processSource(DATA_SRC + chain
      + 'j = lawof(record(x = x))\nr = restrict(j, x = x_data)\n');
    const built = orchestrator.buildDerivations(lifted.bindings);
    return (lifted.diagnostics || []).concat(built.diagnostics || [])
      .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  };
  const wanted = /Posterior 'r' \(bayesupdate\) could not be materialised: its prior is not the law of the likelihood's boundary draws/;
  const mc = of('step = prev -> Normal(prev, 0.75)\nx ~ markovchain(step, 0.0, 5)\n');
  const iid = of('x ~ iid(Normal(mu = 0.0, sigma = 0.75), 5)\n');
  assert.ok(mc.some((m: string) => wanted.test(m)),
    'markovchain restrict must refuse with the idiom message, got ' + JSON.stringify(mc));
  assert.ok(iid.some((m: string) => wanted.test(m)),
    'the SAME refusal must appear without markovchain, got ' + JSON.stringify(iid));
  // And it must never be only the flat engine-gap line — that is what a
  // pruned-by-CLM_SUBSET_VIOLATION derivation used to look like.
  assert.ok(!mc.every((m: string) => /produced no value/.test(m)),
    'the refusal must name the idiom, not only "produced no value"');
});
