'use strict';

// =====================================================================
// ksuperpose-runtime-n.test.ts — spec §06 `ksuperpose` with a runtime N
// =====================================================================
//
// SPEC ANCHOR — docs/06-measure-algebra.md, "Additive superposition",
// `ksuperpose` entry, quoted verbatim:
//
//   "**`ksuperpose(kernel, weights)`** — lifts a kernel to a weighted
//   superposition: the result is itself a kernel, and applying it to a
//   parameter family yields the mixture ν = Σᵢ wᵢ κ(θᵢ), with θᵢ read from
//   row i of the family. The number of components N is the length of
//   `weights`, WHICH NEED NOT BE STATICALLY KNOWN. The family is passed as
//   to broadcast — positional vectors, keyword vectors, or a table (one
//   axis, its rows) — restricted to a single axis: each collection argument
//   has size N or is singular (size one, expanded by repetition), more than
//   one axis is a static error, and non-collection arguments are held
//   constant across the components. `weights` is a distinguished input, not
//   a member of the family, and never expands. It must be non-negative but
//   need not be normalized: the result has total mass Σᵢ wᵢ·totalmass(κ(θᵢ))
//   — Σᵢ wᵢ for a Markov `kernel` — and WHEN EVERY WEIGHT IS ZERO IT IS THE
//   ZERO MEASURE (DENSITY 0, LOG-DENSITY −∞, SAMPLING UNDEFINED). Because
//   the weights do not depend on the variate, the mixture is sampleable
//   whenever `kernel` is."
//
// WHAT THIS COVERS. `ksuperpose-expand.ts` rewrites the applied lift to
// §06's variadic `superpose(weighted(w[i], κ(θᵢ)), …)` spelling, which needs
// N when the pass runs — so a weight vector whose length no type carries was
// a located refusal, against §06's "need not be statically known". The
// runtime arm (`ksuperpose-runtime.ts`) closes that: it resolves N from the
// weight vector's VALUE and synthesises the same component graph, so density,
// mass, `normalize` and the per-output-index component draw are the ones the
// static spelling already uses.
//
// `reverse`, `linspace` and `cumsum` are the runtime-length vehicles: each
// types as `array(rank 1, shape ["%dynamic"])`, so no static length is
// available, and each evaluates to a real vector.
//
// ORACLES — Distributions.jl, computed before any engine output was read.
//
//   using Distributions
//   lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
//   f(x, w, mus, sg) = lse([log(w[i]) + logpdf(Normal(mus[i], sg), x) for i in eachindex(w)])
//
//   TWO components — w = [0.3, 1.2], mus = [-1.0, 2.0], sig = [1.0, 0.5]:
//     | x   | normalized          | unnormalized        |
//     |-----|---------------------|---------------------|
//     | 0.5 | -3.411415107516122  | -3.0059499994079575 |
//     | 2.0 | -0.447547242639128  | -0.0420821345309640 |
//
//   THREE components — mus = [-2.0, 0.0, 2.0], sig = 0.5:
//     | w               | x    | log-density         |
//     |-----------------|------|---------------------|
//     | [0.2, 0.5, 0.3] |  0.3 | -1.096710126724629  |
//     | [0.2, 0.5, 0.3] | -2.0 | -1.834390959984975  |
//     | [0.2, 0.5, 0.3] |  1.0 | -2.448934875825144  |
//     | [0.4, 1.0, 0.6] |  0.3 | -0.403562946164684  |  (unnormalized, Z = 2)
//     | [0.4, 1.0, 0.6] | -2.0 | -1.141243779425030  |  (unnormalized, Z = 2)
//
// TWO cross-checks on the arithmetic itself rather than on the engine. Each
// row was also computed in the NON-LOG domain, log Σᵢ wᵢ·pdf(x; μᵢ, σ), which
// shares no logsumexp step with §06's lowering, and agreed to 1e-15. And
// [0.4, 1.0, 0.6] is exactly 2× [0.2, 0.5, 0.3], so its normalized rows are
// the first set's and its unnormalized rows sit log 2 above them.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error('score: __score__ produced no data');
  return s[0];
}

// Errors from the ANALYZER (processSource) only.
function analyzerErrors(src: string): string[] {
  return require('..').processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

// Errors from the DERIVATION build, where the runtime arm runs. The analyzer
// is silent about a runtime-N mixture, so a refusal that belongs to the
// runtime arm shows up here and nowhere else.
function buildDiagnostics(src: string): any[] {
  const { processSource, orchestrator } = require('..');
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  return built.diagnostics.filter((d: any) => d.severity === 'error');
}

function buildErrors(src: string): string[] {
  return buildDiagnostics(src).map((d: any) => d.message);
}

// `reverse` of the reversed vector — a dynamic length carrying known values.
const RT2 = 'w = reverse([1.2, 0.3])\n'         // [0.3, 1.2]
  + 'means = reverse([2.0, -1.0])\n'            // [-1.0, 2.0]
  + 'sigmas = reverse([0.5, 1.0])\n';           // [1.0, 0.5]

// =====================================================================
// The vehicle really is dynamic — without this the suite proves nothing
// =====================================================================

test('the runtime-length vectors carry NO static length, so these models do '
  + 'exercise the runtime arm and not the static rewrite', () => {
  const { processSource } = require('..');
  for (const expr of ['reverse([1.2, 0.3])', 'linspace(-2.0, 2.0, 3)',
    'cumsum([0.2, 0.3])']) {
    const t = processSource(`v = ${expr}\n`).bindings.get('v').inferredType;
    assert.equal(t.kind, 'array', `${expr}: type kind`);
    assert.deepEqual(t.shape, ['%dynamic'],
      `${expr}: must have no static length, got ${JSON.stringify(t.shape)}`);
  }
});

// =====================================================================
// §06 density with a runtime N
// =====================================================================

test('§06: a runtime-N two-component mixture hits the same oracle rows as the '
  + 'static spelling, normalized and not', async () => {
  const oracle: [number, number, number][] = [
    [0.5, -3.411415107516122, -3.0059499994079575],
    [2.0, -0.447547242639128, -0.0420821345309640],
  ];
  for (const [x, norm, unnorm] of oracle) {
    const gotN = await score(RT2
      + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotN - norm) <= TOL,
      `normalized x = ${x}: got ${gotN}, oracle ${norm}`);
    const gotU = await score(RT2
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotU - unnorm) <= TOL,
      `unnormalized x = ${x}: got ${gotU}, oracle ${unnorm}`);
  }
});

test('§06: N = 3 from a runtime weight vector, with a `linspace` family whose '
  + 'length is dynamic too', async () => {
  // Both vectors have a dynamic length: `reverse` for the weights, and
  // `linspace(-2, 2, 3)` = [-2, 0, 2] for the family.
  const RT3 = 'w = reverse([0.3, 0.5, 0.2])\n'  // [0.2, 0.5, 0.3]
    + 'mus = linspace(-2.0, 2.0, 3)\n';
  const oracle: [number, number][] = [
    [0.3, -1.096710126724629],
    [-2.0, -1.834390959984975],
    [1.0, -2.448934875825144],
  ];
  for (const [x, want] of oracle) {
    const got = await score(RT3
      + 'mix = ksuperpose(Normal, w)(mu = mus, sigma = 0.5)\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(got - want) <= TOL, `x = ${x}: got ${got}, oracle ${want}`);
  }
});

test('§06: an UNNORMALIZED runtime mixture carries mass Σᵢ wᵢ — the same rows '
  + 'sit log 2 above the normalized ones', async () => {
  const RT3 = 'w = reverse([0.6, 1.0, 0.4])\n'   // [0.4, 1.0, 0.6], Σ = 2
    + 'mus = linspace(-2.0, 2.0, 3)\n';
  const rows: [number, number, number][] = [
    [0.3, -0.403562946164684, -1.096710126724629],
    [-2.0, -1.141243779425030, -1.834390959984975],
  ];
  for (const [x, unnorm, norm] of rows) {
    const gotU = await score(RT3
      + 'mix = ksuperpose(Normal, w)(mu = mus, sigma = 0.5)\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotU - unnorm) <= TOL,
      `unnormalized x = ${x}: got ${gotU}, oracle ${unnorm}`);
    const gotN = await score(RT3
      + 'mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 0.5))\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotN - norm) <= TOL,
      `normalized x = ${x}: got ${gotN}, oracle ${norm}`);
    // The identity, on the engine's own two numbers: normalize divides by
    // Z = Σᵢ wᵢ = 2 for a Markov component, so the gap is exactly log 2.
    assert.ok(Math.abs((gotU - gotN) - Math.log(2)) <= TOL,
      `x = ${x}: gap ${gotU - gotN}, log Z = ${Math.log(2)}`);
  }
});

test('§06: a non-collection family argument is held constant, and a singular '
  + 'one is expanded by repetition — with a runtime N', async () => {
  // sigma = 1.0 held constant across both components: the shared-scalar-sigma
  // oracle row from ksuperpose-density.test.ts, reached with a runtime N.
  const held = await score(RT2
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = 1.0))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(held - (-2.0439385332046727)) <= TOL,
    `held-constant sigma: got ${held}, oracle -2.0439385332046727`);
  // A size-one collection reads row 1 for every component, so `[1.0]` in
  // `mu` is the singular-mu oracle row.
  const singular = await score(RT2
    + 'mu1 = [1.0]\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = mu1, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(singular - (-0.7818336515543383)) <= TOL,
    `singular mu: got ${singular}, oracle -0.7818336515543383`);
});

test('§06: the family may be passed POSITIONALLY under a runtime N too',
  async () => {
    const pos = await score(RT2
      + 'mix = normalize(ksuperpose(Normal, w)(means, sigmas))\n'
      + '__score__ = logdensityof(mix, 0.5)\n');
    assert.ok(Math.abs(pos - (-3.411415107516122)) <= TOL,
      `positional: got ${pos}, oracle -3.411415107516122`);
  });

test('§08: a categorical over arbitrary labels is a Dirac superposition with a '
  + 'runtime N — density log pₖ at the label', async () => {
  // p sums to one, so normalize is a no-op and the density at the label is
  // log pₖ, by hand: log 0.8 and log 0.2.
  const SRC = 'p = reverse([0.8, 0.2])\nlabels = reverse([1.5, 0.0])\n'
    + 'cat = normalize(ksuperpose(Dirac, p)(value = labels))\n';
  const hi = await score(SRC + '__score__ = logdensityof(cat, 1.5)\n');
  assert.ok(Math.abs(hi - Math.log(0.8)) <= TOL,
    `at label 1.5: got ${hi}, closed form ${Math.log(0.8)}`);
  const lo = await score(SRC + '__score__ = logdensityof(cat, 0.0)\n');
  assert.ok(Math.abs(lo - Math.log(0.2)) <= TOL,
    `at label 0.0: got ${lo}, closed form ${Math.log(0.2)}`);
});

// =====================================================================
// §06 per-weight rules (#187) hold on the runtime arm
// =====================================================================

test('§06: a ZERO weight drops its component out EXACTLY, with a runtime N',
  async () => {
    const got = await score('w = reverse([1.2, 0.0])\n'    // [0.0, 1.2]
      + 'means = reverse([2.0, -1.0])\nsigmas = reverse([0.5, 1.0])\n'
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
      + '__score__ = logdensityof(mix, 0.5)\n');
    // log(1.2) + logpdf(Normal(2.0, 0.5), 0.5), by hand — no mixture involved.
    const closed = Math.log(1.2)
      + (-Math.log(0.5) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((0.5 - 2.0) / 0.5) ** 2);
    assert.ok(Math.abs(got - closed) <= TOL, `got ${got}, closed form ${closed}`);
    assert.deepEqual(buildErrors('w = reverse([1.2, 0.0])\n'
      + 'means = reverse([2.0, -1.0])\nsigmas = reverse([0.5, 1.0])\n'
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'), [],
    'a zero weight is legal — it must raise nothing');
  });

test('§06: ALL-zero weights are the zero measure — log-density −∞ everywhere, '
  + 'with a runtime N', async () => {
  for (const x of [0.5, -1.0, 2.0]) {
    const got = await score('w = reverse([0.0, 0.0])\n'
      + 'means = reverse([2.0, -1.0])\nsigmas = reverse([0.5, 1.0])\n'
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.equal(got, -Infinity, `x = ${x}: got ${got}`);
  }
});

test('§06: a NEGATIVE weight is refused while scoring a runtime-N mixture, not '
  + 'silently zeroed', async () => {
  const src = 'w = reverse([1.2, 0.0 - 0.3])\n'
    + 'means = reverse([2.0, -1.0])\nsigmas = reverse([0.5, 1.0])\n'
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  await assert.rejects(() => score(src), (e: any) => {
    assert.match(e.message, /non-negative/);
    assert.match(e.message, /-0\.3/);
    assert.match(e.message, /§06/);
    // It must name the value, not report the surviving component's density.
    // Deleting the negative component would have answered this number.
    const deleted = Math.log(1.2)
      + (-Math.log(0.5) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((0.5 - 2.0) / 0.5) ** 2);
    assert.doesNotMatch(e.message, new RegExp(String(deleted)));
    return true;
  });
});

test('§06: a runtime negative weight reaches the MCMC ModelView as a refusal, '
  + 'not as a rejected proposal', async () => {
  // mh/emcee swallow every density throw to −∞, so an untagged refusal turns
  // this model into a constant chain with no diagnostic. `_ctx-factory`'s
  // worker bridge rewraps a thrown Error and drops the tag, so the ModelView
  // route is the only place the tag can be observed.
  const MV = require('../model-view.ts');
  const src = 'w = reverse([1.2, 0.0 - 0.3])\n'
    + 'mu ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'mus = [mu, 2.0]\n'
    + 'mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n'
    + 'y ~ mix\n'
    + 'prior = lawof(record(mu = mu))\n'
    + 'fk = kernelof(record(y = y), mu = mu)\n'
    + 'L = likelihoodof(fk, record(y = 0.5))\n'
    + 'posterior = bayesupdate(L, prior)\n';
  const { ctx } = ctxFor(src, 64);
  const dv = ctx.lookupDerivation
    ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
  await assert.rejects(() => MV.buildModelViewFromCtx(ctx, dv), /non-negative/);
});

test('§06: a NaN weight in a runtime-N mixture gets the NaN message, not a '
  + 'negative-mass one', async () => {
  const src = 'w = reverse([1.2, 0.0 / 0.0])\n'
    + 'means = reverse([2.0, -1.0])\nsigmas = reverse([0.5, 1.0])\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  await assert.rejects(() => score(src), (e: any) => {
    assert.match(e.message, /not a number|NaN/);
    assert.doesNotMatch(e.message, /NEGATIVE/);
    return true;
  });
});

// =====================================================================
// Sampling — proportions follow the weights, per output index
// =====================================================================

// Nearest-component assignment counts over a materialised measure. The
// components are separated by 4 sigma-units of their own width, so the
// assignment is unambiguous and the counts estimate the mixing proportions.
async function proportions(src: string, name: string, mus: number[], N: number) {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  const s = Array.from(m.samples as any).map(Number);
  assert.equal(s.length, N, 'sample count');
  const counts = new Array(mus.length).fill(0);
  for (const x of s) {
    let best = 0;
    for (let i = 1; i < mus.length; i++) {
      if (Math.abs(x - mus[i]) < Math.abs(x - mus[best])) best = i;
    }
    counts[best]++;
  }
  return counts.map((c) => c / N);
}

test('§06: a runtime-N mixture draws each component in proportion to its '
  + 'weight', async () => {
  // w = [0.2, 0.5, 0.3] over means −4 / 0 / 4 at sigma 0.4 — the components
  // are 10 sigma apart, so a draw's nearest mean IS its component.
  const src = 'w = reverse([0.3, 0.5, 0.2])\n'
    + 'mus = linspace(-4.0, 4.0, 3)\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 0.4))\n'
    + 'y ~ M\n';
  const N = 20000;
  const p = await proportions(src, 'y', [-4, 0, 4], N);
  const want = [0.2, 0.5, 0.3];
  // 4 standard errors of a binomial proportion at N = 20000 is under 0.015.
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(p[i] - want[i]) <= 0.02,
      `component ${i + 1}: proportion ${p[i]}, weight ${want[i]}`);
  }
});

test('§06: the runtime-N mixture is sampleable and its moments match '
  + 'Distributions.jl MixtureModel', async () => {
  // MixtureModel([Normal(-3,1), Normal(3,1)], [0.5, 0.5]) — mean 0, var 10.
  const src = 'w = reverse([0.5, 0.5])\nmus = linspace(-3.0, 3.0, 2)\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n'
    + 'y ~ M\n';
  const N = 20000;
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('y');
  const s = Array.from(m.samples as any).map(Number);
  const mean = s.reduce((a: number, b: number) => a + b, 0) / N;
  const varr = s.reduce((a: number, b: number) => a + (b - mean) ** 2, 0) / (N - 1);
  assert.ok(Math.abs(mean - 0) <= 0.15, `mean ${mean}, oracle 0`);
  assert.ok(Math.abs(varr - 10) <= 0.30, `var ${varr}, oracle 10`);
});

test('§06: under iid(runtime-N mixture, k) every coordinate selects its own '
  + 'component', async () => {
  // §06 `iid` is the product measure, so the k coordinates are independent:
  // per-coordinate mean 0 and variance 10, and zero cross-covariance. The
  // branch-pinning defect (engine-concepts §22.4) puts coordinate 0 at −3 and
  // coordinate 1 at +3 with covariance near the full 9.
  const src = 'w = reverse([0.5, 0.5])\nmus = linspace(-3.0, 3.0, 2)\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n';
  const N = 20000; const k = 3;
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('b');
  const flat = Array.from(m.samples as any).map(Number);
  assert.equal(flat.length, N * k, 'sample count');
  const mean = new Array(k).fill(0);
  for (let i = 0; i < flat.length; i++) mean[i % k] += flat[i] / N;
  const varr = new Array(k).fill(0);
  const cov = new Array(k).fill(0);
  for (let a = 0; a < N; a++) {
    const d0 = flat[a * k] - mean[0];
    for (let i = 0; i < k; i++) {
      const di = flat[a * k + i] - mean[i];
      varr[i] += (di * di) / (N - 1);
      cov[i] += (d0 * di) / (N - 1);
    }
  }
  for (let i = 0; i < k; i++) {
    assert.ok(Math.abs(mean[i] - 0) <= 0.15, `coord ${i} mean ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 10) <= 0.30, `coord ${i} var ${varr[i]}`);
    if (i > 0) {
      assert.ok(Math.abs(cov[i]) <= 0.45, `coord 0·${i} covariance ${cov[i]}`);
    }
  }
});

test('§06: all-zero weights make SAMPLING undefined — a located refusal, not a '
  + 'constant repeated for every atom', async () => {
  const src = 'w = reverse([0.0, 0.0])\nmus = linspace(-3.0, 3.0, 2)\n'
    + 'M = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n'
    + 'y ~ M\n';
  const { ctx } = ctxFor(src, 8);
  await assert.rejects(() => ctx.getMeasure('y'), (e: any) => {
    assert.match(e.message, /zero measure|zero mass/);
    assert.match(e.message, /§06/);
    return true;
  });
});

// =====================================================================
// The STATIC spelling is untouched
// =====================================================================

test('the static-N spelling still expands to §06\'s variadic superpose — no '
  + 'runtime-arm binding appears in it, and the numbers are unmoved', async () => {
  const { processSource, orchestrator } = require('..');
  const src = 'w = [0.3, 1.2]\nmeans = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const synthesised = [...built.bindings.keys()]
    .filter((n: any) => String(n).startsWith('__ksrt'));
  assert.deepEqual(synthesised, [],
    'the static spelling must not reach the runtime arm');
  // And the frozen oracle row is unmoved.
  const got = await score(src);
  assert.ok(Math.abs(got - (-3.411415107516122)) <= TOL,
    `got ${got}, oracle -3.411415107516122`);
});

test('the runtime arm and the static rewrite agree on one measure written both '
  + 'ways', async () => {
  const dynamic = await score(RT2
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const stat = await score('w = [0.3, 1.2]\nmeans = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.equal(dynamic, stat,
    'the two spellings denote one measure and must score bit-identically');
});

// =====================================================================
// What the runtime arm still refuses, and where it says so
// =====================================================================

test('the ANALYZER is silent about a runtime-N mixture — the refusal that used '
  + 'to fire there is gone', () => {
  assert.deepEqual(analyzerErrors(RT2
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'), [],
  'no analyzer error — §06 allows a runtime N');
});

test('a family collection whose runtime size is neither N nor one is refused, '
  + 'naming both sizes', () => {
  const errs = buildErrors('w = reverse([1.2, 0.3])\n'
    + 'three = reverse([1.0, 2.0, 3.0])\n'
    + 'mix = ksuperpose(Normal, w)(mu = three, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /size 3/.test(m) && /N = 2/.test(m)),
    `want a size refusal, got ${JSON.stringify(errs)}`);
  assert.ok(buildDiagnostics('w = reverse([1.2, 0.3])\n'
    + 'three = reverse([1.0, 2.0, 3.0])\n'
    + 'mix = ksuperpose(Normal, w)(mu = three, sigma = 1.0)\n')
    .some((d: any) => d.loc), 'the refusal is located');
});

test('a REIFIED kernel with a runtime N is refused, naming the remedy — the '
  + 'runtime arm builds a distribution call, not a kernel application', () => {
  const errs = buildErrors(RT2
    + 'K = m -> Normal(mu = m, sigma = 1.0)\n'
    + 'mix = ksuperpose(K, w)(means)\n');
  assert.ok(errs.some((m) => /reified|user-defined kernel/.test(m)
    && /statically-known/.test(m)),
  `want a kernel refusal, got ${JSON.stringify(errs)}`);
  // The control: the SAME reification with a STATIC N is supported, so the
  // refusal is about the runtime N and not about reification.
  assert.deepEqual(analyzerErrors('w = [0.3, 1.2]\nmeans = [-1.0, 2.0]\n'
    + 'K = m -> Normal(mu = m, sigma = 1.0)\n'
    + 'mix = normalize(ksuperpose(K, w)(means))\n'), []);
});

test('a weight vector that is an INLINE expression is refused, naming the '
  + 'remedy — the runtime arm reads a named binding\'s value', () => {
  const errs = buildErrors('means = reverse([2.0, -1.0])\n'
    + 'mix = ksuperpose(Normal, reverse([1.2, 0.3]))(mu = means, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /Bind it to a name/.test(m)),
    `want a naming remedy, got ${JSON.stringify(errs)}`);
});

test('a refused runtime-N mixture yields ONE diagnostic, not a precise refusal '
  + 'followed by a generic "engine gap"', () => {
  // A refusal drops the mixture, and the cascade-prune then drops every
  // wrapper above it, each of which would otherwise draw the fixed-phase
  // dead-end message — "this is an engine gap" appended to a diagnosis that
  // already named the cause and the remedy.
  for (const src of [
    'w = reverse([1.2, 0.3])\nmeans = reverse([2.0, -1.0])\n'
      + 'K = m -> Normal(mu = m, sigma = 1.0)\n'
      + 'mix = normalize(ksuperpose(K, w)(means))\n'
      + '__score__ = logdensityof(mix, 0.5)\n',
    'w = external(cartpow(nonnegreals, 2))\n'
      + 'mix = normalize(ksuperpose(Normal, w)(mu = 0.0, sigma = 1.0))\n'
      + '__score__ = logdensityof(mix, 0.5)\n',
  ]) {
    const errs = buildErrors(src);
    assert.equal(errs.length, 1, `want one diagnostic, got ${JSON.stringify(errs)}`);
    assert.match(errs[0], /ksuperpose/);
    assert.doesNotMatch(errs[0], /engine gap/);
  }
});

test('a chained user call is not mistaken for an applied mixture', () => {
  // `f(x)(y)` lowers to the same expression-headed shape as an applied lift,
  // so the runtime arm must read the callee's op and not just its kind.
  const errs = buildErrors('f = x -> (y -> x + y)\nz = f(1.0)(2.0)\n');
  assert.ok(!errs.some((m) => /ksuperpose/.test(m)),
    `no ksuperpose diagnostic belongs here, got ${JSON.stringify(errs)}`);
});

test('§06: two family axes over a scalar parameter is refused under a runtime '
  + 'N too', () => {
  const errs = buildErrors('w = reverse([1.2, 0.3])\n'
    + 'grid = [[1.0, 2.0], [3.0, 4.0]]\n'
    + 'mix = ksuperpose(Normal, w)(mu = grid, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /exactly one family axis/.test(m) && /`mu`/.test(m)),
    `want a family-axis refusal, got ${JSON.stringify(errs)}`);
});

test('a TABLE family is refused under a runtime N with the keyword-vector '
  + 'remedy', () => {
  const errs = buildErrors('w = reverse([1.2, 0.3])\n'
    + 'fam = table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])\n'
    + 'mix = ksuperpose(Normal, w)(fam)\n');
  assert.ok(errs.some((m) => /TABLE parameter family/.test(m)
    && /keyword vectors/.test(m)),
  `want a table refusal, got ${JSON.stringify(errs)}`);
});

test('a family argument with neither a length nor a value is refused, naming '
  + 'the argument', () => {
  // A host-supplied `external` column: the type is %deferred and no value
  // exists at build time, so nothing says whether it is a size-N collection.
  const errs = buildErrors('w = reverse([1.2, 0.3])\n'
    + 'ms = external(cartpow(reals, 2))\n'
    + 'mix = ksuperpose(Normal, w)(mu = ms, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /`mu`/.test(m)
    && /no statically-known length and no resolvable value/.test(m)),
  `want a family-argument refusal, got ${JSON.stringify(errs)}`);
});

test('a family argument written as an INLINE expression is refused, naming the '
  + 'argument and the remedy', () => {
  const errs = buildErrors('w = reverse([1.2, 0.3])\n'
    + 'means = reverse([2.0, -1.0])\n'
    + 'mix = ksuperpose(Normal, w)(mu = means + 1.0, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /`mu` is an inline expression/.test(m)
    && /Bind it to a name/.test(m)),
  `want an inline-argument refusal, got ${JSON.stringify(errs)}`);
});

test('a weight vector whose value the engine cannot resolve at all is still '
  + 'refused — the honest remaining limit', () => {
  // `external(...)` is a host-supplied boundary input: no value exists at
  // build time, so N cannot be read from it. §06 allows it; this engine
  // says so rather than scoring a mixture whose component count it guessed.
  const errs = buildErrors('w = external(cartpow(nonnegreals, 2))\n'
    + 'mix = ksuperpose(Normal, w)(mu = 0.0, sigma = 1.0)\n');
  assert.ok(errs.some((m) => /component count N/.test(m)),
    `want an N refusal, got ${JSON.stringify(errs)}`);
});
