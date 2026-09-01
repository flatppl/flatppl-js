'use strict';
// A latent that the enclosing `kernelof` declares as an INPUT must be the
// kernel's ARGUMENT, never integrated out and never fed a prior draw.
//
// Spec §04 "Kernels and `kernelof`": "`kernelof(x, kwargs...)` is equivalent to
// `functionof(lawof(x), kwargs...)` interpreted within the reified subgraph
// delimited by `kwargs` — the boundary substitution applies before the inner
// `lawof` is interpreted, so an enclosing `kernelof` boundary scopes what the
// inner `lawof` marginalizes over." And §04 "Reification with interdependent
// boundary nodes": substitution "replaces every designated node with a fresh
// independent input *before* the ancestor trace runs, so original dependencies
// between boundaries are erased". So a declared input's own prior is GONE from
// the lowered body, and §06 `likelihoodof` scores pdf(K(theta), obs) at the
// theta the caller supplies.
//
// The defect these tests pin: clm's shared-ancestor gate tested only
// `_isMeasureNode(body)`, and `record` is in none of the builtin measure-op
// sets. A `kernelof(record(...), ...)` body keeps the surface `record` op (it
// never goes through derivations.expandMeasureIR's record→joint
// canonicalisation), so the gate skipped it: an unexposed stochastic ancestor
// was then fed as a per-atom PRIOR DRAW and `logdensityof` returned a
// seed-dependent number with no diagnostic. See the seed-stability test at the
// bottom, which is the direct guard on that.
//
// Oracles are INDEPENDENT closed forms, each cross-checked against
// scipy.stats.norm.logpdf:
//   N(x; m, s) = -0.5*log(2*pi) - log s - (x-m)^2/(2 s^2)
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function score(src: string, seed: number): Promise<number> {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'unexpected diagnostics: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1,
    rootKey: seed,
    rootSeed: seed,
    marginalizationCount: 64,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message))
        : Promise.resolve(r);
    },
  };
  return Promise.resolve(ctx.getMeasure('lp'))
    .then((mm: any) => (mm.value ? mm.value.data[0] : mm.samples[0]));
}

// ── two declared inputs, a record-valued kernel variate ───────────────────
//
// The witness shape. Both `mu_a` and `mu_b` are declared inputs, so both
// priors are erased and the observation is scored against Normal(0.3, 1) and
// Normal(1.0, 2):
//   N(0.61; 0.3, 1) + N(1.1; 1.0, 2) = -2.580324246969291
// The marginal (the wrong answer) integrates both priors back in, giving
// Normal(0, sqrt(26)) and Normal(0, sqrt(29)):
//   N(0.61; 0, sqrt26) + N(1.1; 0, sqrt29) = -5.178591088609609
const WITNESS = `
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
mu_b ~ Normal(mu = 0.0, sigma = 5.0)
run = joint(a = Normal(mu = mu_a, sigma = 1.0), b = Normal(mu = mu_b, sigma = 2.0))
r ~ run
L = likelihoodof(kernelof(record(r = r), mu_a = mu_a, mu_b = mu_b), record(r = record(a = 0.61, b = 1.1)))
lp = logdensityof(L, record(mu_a = 0.3, mu_b = 1.0))
`;

const COND_WITNESS = -2.580324246969291;
const MARG_WITNESS = -5.178591088609609;

test('kernelof declared inputs score the conditional pdf, not the prior predictive', async () => {
  const v = await score(WITNESS, 0xBA5E);
  assert.ok(Math.abs(v - COND_WITNESS) < 1e-12, `got ${v}, want ${COND_WITNESS}`);
  // The marginal sigmas sqrt(26) / sqrt(29) must be absent, not merely
  // outscored: an engine that integrated the priors back in lands here.
  assert.ok(Math.abs(v - MARG_WITNESS) > 2.0,
    `got ${v}, which is the prior-predictive marginal ${MARG_WITNESS}`);
});

// ── the same shape under `iid` over the record variate (G1 read across) ───
//
// The two rows SHARE the declared inputs: `run` is a measure expression, not a
// reified law, so §06 iid's per-copy freshening of a reified sub-DAG does not
// apply, and the enclosing kernelof substitutes one `mu_a` / `mu_b` for both.
// The variate is reified directly rather than wrapped in a record — §06 iid
// makes it an N-row table, and §03 keeps a table out of a record field.
//   rows (0.61, 1.1) and (0.2, 0.9)
//   = N(.61;.3,1)+N(1.1;1,2)+N(.2;.3,1)+N(.9;1,2) = -5.117598493938582
// marginal: -10.343899087033543
test('a record-valued iid under kernelof shares the declared inputs across rows', async () => {
  const v = await score(`
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
mu_b ~ Normal(mu = 0.0, sigma = 5.0)
run = joint(a = Normal(mu = mu_a, sigma = 1.0), b = Normal(mu = mu_b, sigma = 2.0))
rs ~ iid(run, 2)
data = table(a = [0.61, 0.2], b = [1.1, 0.9])
L = likelihoodof(kernelof(rs, mu_a = mu_a, mu_b = mu_b), data)
lp = logdensityof(L, record(mu_a = 0.3, mu_b = 1.0))
`, 0xBA5E);
  assert.ok(Math.abs(v - (-5.117598493938582)) < 1e-12, `got ${v}, want -5.117598493938582`);
  assert.ok(Math.abs(v - (-10.343899087033543)) > 2.0,
    `got ${v}, which is the prior-predictive marginal -10.343899087033543`);
});

// ── the equivalent `functionof(lawof(...), ...)` spelling ─────────────────
//
// §04 gives this as kernelof's definition, so it must score identically:
//   N(0.61; 0.3, 1) = -0.9669885332046727   (marginal: -2.555142571446183)
test('functionof(lawof(x), mu = mu) scopes the inner lawof the same way', async () => {
  const v = await score(`
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
x ~ Normal(mu = mu_a, sigma = 1.0)
K = functionof(lawof(record(x = x)), mu_a = mu_a)
L = likelihoodof(K, record(x = 0.61))
lp = logdensityof(L, record(mu_a = 0.3))
`, 0xBA5E);
  assert.ok(Math.abs(v - (-0.9669885332046727)) < 1e-12, `got ${v}, want -0.9669885332046727`);
  assert.ok(Math.abs(v - (-2.555142571446183)) > 1.0,
    `got ${v}, which is the prior-predictive marginal -2.555142571446183`);
});

// ── a mid-chain boundary erases everything upstream of it ─────────────────
//
// `x` is the declared input, so `mu_a`'s prior is erased rather than
// marginalised: N(1.1; 0.3, 2) = -1.6920857137646181.
// Marginalising mu_a instead gives N(1.1; 0, sqrt(30)) = -2.639703890702417.
const MID_CHAIN = -1.6920857137646181;

test('a mid-chain declared input erases the upstream prior', async () => {
  const v = await score(`
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
x ~ Normal(mu = mu_a, sigma = 1.0)
y ~ Normal(mu = x, sigma = 2.0)
L = likelihoodof(kernelof(record(y = y), x = x), record(y = 1.1))
lp = logdensityof(L, record(x = 0.3))
`, 0xBA5E);
  assert.ok(Math.abs(v - MID_CHAIN) < 1e-12, `got ${v}, want ${MID_CHAIN}`);
  assert.ok(Math.abs(v - (-2.639703890702417)) > 0.5,
    `got ${v}, which marginalises mu_a instead of erasing it`);
});

// §04 "Reification with interdependent boundary nodes": "A boundary whose only
// paths to the output went through another boundary then has no occurrence in
// the lowered body — the callable is constant in that input." `mu_a` reaches
// `y` only through the boundary `x`, so the score must ignore it entirely.
test('the kernel is constant in a boundary shadowed by another boundary', async () => {
  const SRC = (muA: string) => `
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
x ~ Normal(mu = mu_a, sigma = 1.0)
y ~ Normal(mu = x, sigma = 2.0)
L = likelihoodof(kernelof(record(y = y), mu_a = mu_a, x = x), record(y = 1.1))
lp = logdensityof(L, record(mu_a = ${muA}, x = 0.3))
`;
  const lo = await score(SRC('0.0'), 0xBA5E);
  const hi = await score(SRC('99.0'), 0xBA5E);
  assert.ok(Math.abs(lo - MID_CHAIN) < 1e-12, `got ${lo}, want ${MID_CHAIN}`);
  assert.equal(hi, lo, 'the score must not depend on the shadowed boundary mu_a');
});

// ── an unexposed latent refuses instead of returning a prior draw ─────────
//
// Here `theta` is neither a declared input nor a field of the observed record,
// so §06's density is the marginal integral over it. The declared input `mu`
// reaches that integral, which the linear-Gaussian recogniser cannot yet
// thread a caller-substituted value into (its KNOWN GAP, tracked in
// flatppl-dev/TODO-flatppl-js.md). It must REFUSE and name the latent. Before
// the record-gate fix this shape returned a seed-dependent number instead.
test('an unexposed stochastic ancestor under kernelof refuses, naming the latent', async () => {
  await assert.rejects(() => score(`
mu ~ Normal(mu = 0.0, sigma = 5.0)
theta ~ Normal(mu = mu, sigma = 1.0)
y ~ Normal(mu = theta, sigma = 2.0)
L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = 0.61))
lp = logdensityof(L, record(mu = 0.3))
`, 0xBA5E), (e: any) => {
    assert.match(e.message, /marginalises the stochastic ancestor\(s\) theta/);
    return true;
  });
});

test('a non-Gaussian unexposed ancestor under kernelof refuses', async () => {
  await assert.rejects(() => score(`
mu ~ Normal(mu = 0.0, sigma = 5.0)
s ~ Exponential(rate = 1.0)
y ~ Normal(mu = mu, sigma = s)
L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = 0.61))
lp = logdensityof(L, record(mu = 0.3))
`, 0xBA5E), (e: any) => {
    assert.match(e.message, /marginalises the stochastic ancestor\(s\) s/);
    assert.match(e.message, /not a Normal/);
    return true;
  });
});

// ── the direct guard on the defect ────────────────────────────────────────
//
// A log-density is a function of the model, the observation and theta — never
// of the RNG. Feeding a latent as a prior draw is invisible in a single-seed
// value but shows up here immediately: the old code returned -6.649143574825803
// at seed 1 and -1.7127060286259714 at seed 2 for the refusing shape above.
test('a kernelof likelihood score does not depend on the seed', async () => {
  for (const src of [WITNESS, `
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
x ~ Normal(mu = mu_a, sigma = 1.0)
y ~ Normal(mu = x, sigma = 2.0)
L = likelihoodof(kernelof(record(y = y), x = x), record(y = 1.1))
lp = logdensityof(L, record(x = 0.3))
`]) {
    const a = await score(src, 1);
    const b = await score(src, 2);
    const c = await score(src, 12345);
    assert.equal(a, b, 'score changed between seeds 1 and 2');
    assert.equal(a, c, 'score changed between seeds 1 and 12345');
  }
});
