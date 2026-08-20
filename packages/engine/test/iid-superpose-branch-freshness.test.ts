'use strict';

// Every coordinate of an `iid` over a `superpose` selects its OWN component.
//
// SPEC ANCHOR — flatppl-design docs/06-measure-algebra.md, "Additive
// superposition", `superpose` entry, quoted verbatim: "**`superpose(M1, M2,
// ...)`** — measure addition: $\nu(A) = M_1(A) + M_2(A) + \ldots$ All
// components must share the same variate space."
//
// SPEC ANCHOR — same file, "Joint composition", `iid` entry, quoted verbatim:
// "**`iid(M, size)`** — the product measure $M^{\otimes N}$ over arrays of
// shape `size`, where `N = prod(size)`. … When `M` is a reified law, each of
// the $N$ copies carries its own copy of the reified sub-DAG, stochastic
// ancestors included; `iid` never shares nodes between copies."
//
// A product of sums is a sum over the component choices of every coordinate
// independently, so each of the N coordinates draws its own component.
//
// THE DEFECT these tests pin. `matSuperpose` selected a component with ONE
// `systematicResample` over the P×K block pool at n = K. Systematic resampling
// spreads its K sample positions evenly over the cumulative weights — (j + u)/K
// for j = 0…K−1 — and the pool was laid out parent-major, so position j always
// landed inside parent-stratum j. Slot 0 took component 0 every time and slot
// K−1 the last component every time. For two equal-weight branches at −3 and
// +3 with k = 3 the per-coordinate means were −2.9989 / 0.0015 / 2.9971 with
// variances 1.0010 / 9.9860 / 1.0029, where §06's product of three mixtures is
// mean 0 and variance 10 in EVERY coordinate. At k = 2 and k = 4 every
// coordinate was pinned (means −3 / +3 and −3 / −3 / +3 / +3). Silent, and a
// live disagreement with the density path, which scored the exact independent
// product throughout.
//
// ORACLES ARE INDEPENDENT of this engine. Every moment target below is
// `Distributions.jl`'s `MixtureModel` (`mean`, `var`) or closed-form
// conditioning by hand, and every density target is a hand-written sum of
// `logpdf` values confirmed in the same Julia session. flatppl-rust evaluates
// no density and is not consulted.
//
// TOLERANCES. The materialiser's stream is deterministic run to run, so these
// are Monte-Carlo slack, not flakiness insurance. At N = 20000 for the ±3
// mixture (Var = 10, E[X⁴] = 138): the standard error is sqrt(10/N) = 0.022 on
// a mean, sqrt((138 − 100)/N) = 0.044 on a variance and sqrt(100/N) = 0.071 on
// a covariance. The tolerances are about 6 sigma of that, and the defect sat
// 9 variance units and 3 mean units away. The determinism does NOT come from
// `_ctx-factory`'s seed — the factory sets a SCALAR `rootKey` where the
// engine's contract is the two-lane PhiloxKey `[k0, k1]`, so `foldIn`
// collapses; that is the defect carded in flatppl-dev/TODO-flatppl-js.md
// (`worker.ts:240`). Quote the draw count with any figure taken from here.

const test = require('node:test');
const assert = require('node:assert');
const { processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 20000;
const MEAN_TOL = 0.15;
const VAR_TOL = 0.30;
const COV_TOL = 0.45;
const F64_TOL = 1e-12;

const infer = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

// Per-coordinate means and variances, plus the covariance of coordinate 0
// against each coordinate, read off a materialised iid measure's atom-major
// samples.
async function momentsOf(src: string, name: string, k: number) {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
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
  return { mean, varr, cov };
}

const scoreOf = async (src: string) => {
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('ld')).samples[0];
};

// `u ~ N(-3, 1)`, `w ~ N(3, 1)`, S the equal-weight superposition of their
// reified laws. Total mass 0.5 + 0.5 = 1, so S is already a probability
// measure and `normalize(S)` is the identity on it.
const MIX = 'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
  + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
  + 'S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n';

// Distributions.jl: MixtureModel([Normal(-3,1), Normal(3,1)], [0.5,0.5]) has
// mean 0.0 and var 10.0.
const MIX_MEAN = 0.0;
const MIX_VAR = 10.0;

// ── Every coordinate selects its own component ───────────────────────────────

// The compound-spelling roster. Each entry routes `iid` to the composite
// fallback around a `superpose`, which is where the pinning lived; every one
// of them was pinned before the fix.
const SPELLINGS: Array<[string, string, number, number, number]> = [
  ['bare superpose', MIX + 'b ~ iid(S, 3)\n', 3, MIX_MEAN, MIX_VAR],
  ['normalize(superpose)', MIX + 'M = normalize(S)\nb ~ iid(M, 3)\n',
    3, MIX_MEAN, MIX_VAR],
  // `weighted(2.0, S)` doubles the mass; the SAMPLE positions are the
  // mixture's, so the moments are unchanged (the weight lives in the density).
  ['weighted(2.0, superpose)', MIX + 'M = weighted(2.0, S)\nb ~ iid(M, 3)\n',
    3, MIX_MEAN, MIX_VAR],
  // A truncation wide enough to clip nothing: the interval holds ±3 out to
  // 7 sigma, so the moments stay the untruncated mixture's to well past the
  // tolerance.
  ['truncate(superpose)',
    MIX + 'M = truncate(S, interval(-10.0, 10.0))\nb ~ iid(M, 3)\n',
    3, MIX_MEAN, MIX_VAR],
  // A unit translation shifts the mean by 1 and leaves the variance.
  ['pushfwd(superpose)',
    MIX + 'M = pushfwd(x -> x + 1.0, S)\nb ~ iid(M, 3)\n', 3, 1.0, MIX_VAR],
  ['inline distribution branches',
    'M = superpose(weighted(0.5, Normal(mu = -3.0, sigma = 1.0)), '
    + 'weighted(0.5, Normal(mu = 3.0, sigma = 1.0)))\nb ~ iid(M, 3)\n',
    3, MIX_MEAN, MIX_VAR],
  // k = 2 and k = 4 pinned EVERY coordinate, not just the outer ones: with P
  // parents and K slots the systematic positions partition the parent strata
  // exactly when K is a multiple of P.
  ['k = 2', MIX + 'b ~ iid(S, 2)\n', 2, MIX_MEAN, MIX_VAR],
  ['k = 4', MIX + 'b ~ iid(S, 4)\n', 4, MIX_MEAN, MIX_VAR],
  ['nested iid(iid(S, 2), 3)', MIX + 'b ~ iid(iid(S, 2), 3)\n',
    6, MIX_MEAN, MIX_VAR],
];

for (const [label, src, k, wantMean, wantVar] of SPELLINGS) {
  test(`iid over ${label}: every coordinate is its own mixture draw`,
    async () => {
      const { mean, varr, cov } = await momentsOf(src, 'b', k);
      for (let i = 0; i < k; i++) {
        assert.ok(Math.abs(mean[i] - wantMean) < MEAN_TOL,
          `mean[${i}] = ${mean[i]}, want ${wantMean} — a per-coordinate mean at `
          + 'a component centre is the branch-pinning signature');
        assert.ok(Math.abs(varr[i] - wantVar) < VAR_TOL,
          `var[${i}] = ${varr[i]}, want ${wantVar} — a variance near a single `
          + "component's is the branch-pinning signature");
      }
      for (let i = 1; i < k; i++) {
        assert.ok(Math.abs(cov[i]) < COV_TOL,
          `cov[0,${i}] = ${cov[i]}, want 0 (§06's product measure)`);
      }
    });
}

// Three branches, and the pinning was total: with P = K = 3 the systematic
// positions land one per parent stratum, so the per-coordinate means were the
// three component centres −3 / 0 / +3 with variance 1 each.
// Distributions.jl: MixtureModel([Normal(-3,1), Normal(0,1), Normal(3,1)],
// [1/3,1/3,1/3]) has mean 0.0 and var 7.0.
test('a THREE-branch superpose freshens every coordinate', async () => {
  const third = '0.3333333333333333';
  const { mean, varr, cov } = await momentsOf(
    'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
    + 'w ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'v ~ Normal(mu = 3.0, sigma = 1.0)\n'
    + `S3 = superpose(weighted(${third}, lawof(u)), weighted(${third}, lawof(w)), `
    + `weighted(${third}, lawof(v)))\nb ~ iid(S3, 3)\n`, 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 7.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 3; i++) {
    assert.ok(Math.abs(cov[i]) < COV_TOL, `cov[0,${i}] = ${cov[i]}`);
  }
});

// UNEQUAL weights are the sharper test of the selection probabilities: the
// pinning gave means −1.5068 / 2.9971 / 2.9971 (coordinate 0 a 0.75/0.25
// mixture, coordinates 1 and 2 pinned to the heavy branch).
// Distributions.jl: MixtureModel([Normal(-3,1), Normal(3,1)], [0.25,0.75]) has
// mean 1.5 and var 7.75.
test('UNEQUAL branch weights select in proportion in every coordinate',
  async () => {
    const { mean, varr, cov } = await momentsOf(
      'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
      + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
      + 'Sq = superpose(weighted(0.25, lawof(u)), weighted(0.75, lawof(w)))\n'
      + 'b ~ iid(Sq, 3)\n', 'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - 1.5) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 7.75) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 3; i++) {
      assert.ok(Math.abs(cov[i]) < COV_TOL, `cov[0,${i}] = ${cov[i]}`);
    }
  });

// ── Genuine shared parameters must STAY shared ───────────────────────────────
//
// The wrong fix freshens the whole subtree. A node read from OUTSIDE the
// replicated measure — a mixing weight, a component parameter — is fixed
// before replication (engine-concepts §22.4: draw psi_i per atom, then k iid
// draws from M at that psi_i), and freshening it swaps a correct joint for a
// correct marginal over a wrong joint. Cross-coordinate covariance is the
// statistic that separates the two: it is 0 when the parameter is freshened
// and Var(E[X | parameter]) when it is shared.

test('CONTROL: a shared mixing WEIGHT stays shared across the coordinates',
  async () => {
    // psi ~ Uniform(0,1), X | psi ~ psi·N(−3,1) + (1−psi)·N(3,1). Closed form
    // by conditioning: E[X|psi] = 3 − 6psi and Var(X|psi) = 1 + 36psi(1−psi),
    // so Cov(X_i, X_j) = Var(3 − 6psi) = 36·(1/12) = 3, Var(X) = E[1 +
    // 36psi(1−psi)] + 3 = 7 + 3 = 10, and E[X] = 0. Freshening psi would send
    // the covariance to 0 while leaving the mean and variance untouched, so
    // the covariance is the whole assertion.
    const { mean, varr, cov } = await momentsOf(
      'psi ~ Uniform(support = interval(0.0, 1.0))\n'
      + 'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
      + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
      + 'Sp = superpose(weighted(psi, lawof(u)), weighted(1.0 - psi, lawof(w)))\n'
      + 'b ~ iid(Sp, 3)\n', 'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 10.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 3; i++) {
      assert.ok(Math.abs(cov[i] - 3.0) < COV_TOL,
        `cov[0,${i}] = ${cov[i]}, want 3 — a covariance near 0 means the `
        + 'shared mixing weight was freshened per coordinate');
    }
  });

test('CONTROL: a shared component PARAMETER stays shared across the coordinates',
  async () => {
    // m ~ N(0,1), X | m ~ 0.5·N(m−3,1) + 0.5·N(m+3,1). E[X|m] = m and
    // Var(X|m) = 1 + 9 = 10, so Cov(X_i, X_j) = Var(m) = 1, Var(X) = 11 and
    // E[X] = 0. `m` is reached only as a distribution parameter, never in
    // measure position, so the freshening carve-out must leave it tiled.
    const { mean, varr, cov } = await momentsOf(
      'm ~ Normal(mu = 0.0, sigma = 1.0)\n'
      + 'Sm = superpose(weighted(0.5, Normal(mu = m - 3.0, sigma = 1.0)), '
      + 'weighted(0.5, Normal(mu = m + 3.0, sigma = 1.0)))\n'
      + 'b ~ iid(Sm, 3)\n', 'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 11.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 3; i++) {
      assert.ok(Math.abs(cov[i] - 1.0) < COV_TOL,
        `cov[0,${i}] = ${cov[i]}, want 1 — a covariance near 0 means the `
        + 'shared component parameter was freshened per coordinate');
    }
  });

test('CONTROL: a bare superpose with no iid is untouched', async () => {
  // K = 1 keeps its exact prng stream: the per-index selection pool is the
  // same P weights and the same single `systematicResample` call. The measured
  // figures are byte-identical either side of the fix (mean −0.0141,
  // var 9.9929 at 200000 draws); the oracle targets are the mixture's own.
  const { mean, varr } = await momentsOf(MIX + 'b ~ S\n', 'b', 1);
  assert.ok(Math.abs(mean[0] - MIX_MEAN) < MEAN_TOL, `mean = ${mean[0]}`);
  assert.ok(Math.abs(varr[0] - MIX_VAR) < VAR_TOL, `var = ${varr[0]}`);
});

// ── The density path is UNCHANGED and still scores the product ───────────────
//
// The density path already scored the exact independent product before the fix
// — it was the sampler that disagreed with it, and with §06. These scores are
// bit-identical either side of the change; nothing in the density walk was
// touched.
//
// §06 "Density of composed measures": for `iid`,
// logdensityof(iid(M, n), x) = sum_i logdensityof(M, x_i); for `superpose` the
// density is the sum of the components' densities.
//
// Julia (Distributions.jl), equal weights, at x = (−3, 0, 3):
//   sum(log(0.5*pdf(Normal(-3,1), x) + 0.5*pdf(Normal(3,1), x))) = -8.64310993027395
// and with weights 0.25 / 0.75:
//   -8.930791982419091

const MIX_LOGDENSITY = -8.64310993027395;
const MIX_LOGDENSITY_UNEQUAL = -8.930791982419091;

test('logdensityof(iid(superpose, 3), x) is the product of the mixtures',
  async () => {
    const got = await scoreOf(MIX + 'M = iid(S, 3)\n'
      + 'ld = logdensityof(M, [-3.0, 0.0, 3.0])\n');
    assert.ok(Math.abs(got - MIX_LOGDENSITY) < F64_TOL, `got ${got}`);
  });

test('normalize(superpose) scores identically — S already has mass 1',
  async () => {
    const got = await scoreOf(MIX + 'M = iid(normalize(S), 3)\n'
      + 'ld = logdensityof(M, [-3.0, 0.0, 3.0])\n');
    assert.ok(Math.abs(got - MIX_LOGDENSITY) < F64_TOL, `got ${got}`);
  });

test('UNEQUAL weights score the weighted product', async () => {
  const got = await scoreOf(
    'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
    + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
    + 'Sq = superpose(weighted(0.25, lawof(u)), weighted(0.75, lawof(w)))\n'
    + 'M = iid(Sq, 3)\nld = logdensityof(M, [-3.0, 0.0, 3.0])\n');
  assert.ok(Math.abs(got - MIX_LOGDENSITY_UNEQUAL) < F64_TOL, `got ${got}`);
});

// ── iid over a kchain of a superpose ────────────────────────────────────────

test('iid(kchain(superpose, K), 3) no longer pins a branch per coordinate',
  async () => {
    // The chain marginalises the mixture draw through N(mu = _, 1), so each
    // coordinate's marginal is the mixture convolved with a unit normal:
    // E[X] = 0 and Var(X) = 10 + 1 = 11. Pinned, the per-coordinate means were
    // −2.9983 / −0.0009 / +2.9944 with variances 2.0075 / 10.9817 / 2.0077.
    //
    // The CROSS-COORDINATE COVARIANCE is deliberately NOT asserted. It is
    // 0.4791 here against an oracle 0, because `_reifiedVariatesUnder` cannot
    // reach `u` and `w` through the clm reroute that lowers `kchain`, so both
    // stay tiled and the coordinates share them: with independent branch
    // selection that leaves P(same branch)·Var(within branch) = (1/4 + 1/4)·1
    // = 0.5 of covariance, which is what is measured. That is the FRESHENING
    // gap recorded as concern 5 of wave IIDFRESH, not the branch pinning this
    // file fixes, and pinning 0.4791 as expected would bake a wrong number in.
    const { mean, varr } = await momentsOf(MIX
      + 'C = kchain(S, fn(Normal(mu = _, sigma = 1.0)))\nb ~ iid(C, 3)\n',
      'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 11.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
  });

// ── The static sharing walk agrees with the fixed sampler ────────────────────
//
// typeinfer's §06 ancestry walk held `superpose` out of the `iid` skip while
// the sampler pinned a branch per coordinate — over-reporting is the safe half
// of the sequencing rule (sampler first, then the report). The sampler now
// freshens, so the report is a false rejection of a legal program and
// `superpose` joins the roster.

const W1_PREAMBLE = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
w ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
`;

const SUPERPOSE_COMPONENTS: Array<[string, string]> = [
  ['weighted branches',
    'M = iid(superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w))), 3)'],
  ['bare branches', 'M = iid(superpose(lawof(u), lawof(w)), 3)'],
  ['mixed law / distribution branches',
    'M = iid(superpose(weighted(0.5, lawof(u)), '
    + 'weighted(0.5, Normal(mu = 0.0, sigma = 1.0))), 3)'],
  ['normalize over superpose',
    'M = iid(normalize(superpose(weighted(0.5, lawof(u)), '
    + 'weighted(0.5, lawof(w)))), 3)'],
  // §04 "Aliasing is just assignment" — the chain resolves.
  ['alias chain',
    'S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n'
    + 'S2 = S\nM = iid(S2, 3)'],
];

for (const [label, decl] of SUPERPOSE_COMPONENTS) {
  test(`a component that is an iid over a superpose (${label}) is not reported`,
    () => {
      assert.deepEqual(
        infer(W1_PREAMBLE + decl + '\nKJ = joint(p = K1, q = M)\n'), []);
    });
}

test('a shared mixing WEIGHT inside a replicated superpose still reports', () => {
  // The branch's other arguments are still walked. `p` is the mixing weight,
  // read once before replication, and the sampler still shares it — so the
  // report and the sampler agree, which is the property the skip exists to
  // keep.
  const errors = infer(`
z = elementof(reals)
p ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = 0.0, sigma = 1.0)
w ~ Normal(mu = 5.0, sigma = 1.0)
a1 ~ Normal(mu = p, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(superpose(weighted(p, lawof(u)), weighted(1.0 - p, lawof(w))), 3)
KJ = joint(p = K1, q = M)
`);
  assert.ok(errors.some((m: string) => /share the stochastic node 'p'/.test(m)),
    'got: ' + errors.join(' | '));
});

test('a DISTRIBUTION branch of a replicated superpose is still walked', () => {
  // A branch that replicates nothing keeps its report: `Normal(mu = p, …)`
  // reads one `p` fixed before replication.
  const errors = infer(`
z = elementof(reals)
p ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = p, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(superpose(weighted(0.5, lawof(u)), weighted(0.5, Normal(mu = p, sigma = 1.0))), 3)
KJ = joint(p = K1, q = M)
`);
  assert.ok(errors.some((m: string) => /share the stochastic node 'p'/.test(m)),
    'got: ' + errors.join(' | '));
});

test('a superpose NOT under an iid still reports its shared node', () => {
  // The skip is scoped to an `iid` measure argument. A bare superpose
  // component shares `u` with the kernel and must still report.
  const errors = infer(W1_PREAMBLE
    + 'M = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n'
    + 'KJ = joint(p = K1, q = M)\n');
  assert.ok(errors.length > 0, 'expected the sharing report to stay');
});
