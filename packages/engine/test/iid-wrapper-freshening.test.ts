'use strict';

// `iid` over a WRAPPED reified law freshens per coordinate.
//
// SPEC ANCHOR — flatppl-design docs/06-measure-algebra.md, `iid` entry, quoted
// verbatim: "**`iid(M, size)`** — the product measure $M^{\otimes N}$ over
// arrays of shape `size` … When `M` is a reified law, each of the $N$ copies
// carries its own copy of the reified sub-DAG, stochastic ancestors included;
// `iid` never shares nodes between copies."
//
// A product measure has independent coordinates by construction, so the
// operative property is that the replicated measure CARRIES a sub-DAG to copy,
// not that its top node is literally `lawof`. `weighted(2.0, lawof(u))` carries
// `lawof(u)`'s trace; the weight rescales each copy and does not identify the
// copies.
//
// THE DEFECT these tests pin. `iid(weighted(2.0, lawof(u)), 3)` sampled
// var = cov — correlation exactly 1, every coordinate the SAME draw, a singular
// diagonal instead of a product measure. Silent: no diagnostic, plausible
// marginals. The materialiser's composite fallback classified the reified
// variate `u` as an atom-level VALUE draw (it is value-TYPED) and TILED it
// across the k inner positions, which is right for a distribution PARAMETER and
// wrong for the variate of the law being replicated.
//
// ORACLES ARE INDEPENDENT of this engine. The moment targets are closed form for
// `u ~ Normal(0, 1)`: Var = 1 and Cov = 0 per §06's product definition, against
// the singular alternative Cov = Var = 1. The `truncate` row's Var uses the
// closed form for a standard normal truncated to [-1, 1],
// 1 - 2aφ(a)/(2Φ(a)-1) = 0.291125 at a = 1. The density targets are
// log 2 + logpdf sums written out below. flatppl-rust evaluates no density and
// is not consulted.
//
// TOLERANCES. Every moment here is deterministic run to run, so the tolerances
// below are Monte-Carlo slack rather than flakiness insurance. At N = 20000 the
// standard error is sqrt(2/N) = 0.010 for Var and 1/sqrt(N) = 0.0071 for Cov,
// so 0.06 is about 6 sigma — wide enough to survive an RNG-neutral refactor,
// and 16 sigma away from the defect's Cov = 1. Note that the determinism does
// NOT come from `_ctx-factory`'s seed: the factory sets a SCALAR `rootKey` where
// the engine's contract is the two-lane PhiloxKey `[k0, k1]`, so `foldIn`
// collapses and every seed draws one stream. That is the scalar-as-rootKey
// defect carded in flatppl-dev/TODO-flatppl-js.md (`worker.ts:240`), of which
// the factory is a fourth site — not an engine-wide seed fault. It means these
// numbers must not be read as one sample from a seeded stream, so quote the
// draw count with any figure taken from here.

const test = require('node:test');
const assert = require('node:assert');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 20000;
const MOMENT_TOL = 0.06;
const F64_TOL = 1e-12;

// Per-coordinate variances and the covariances of coordinate 0 against the
// rest, read off a materialised iid measure's flat atom-major samples.
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

const STD_NORMAL = 'u ~ Normal(mu = 0.0, sigma = 1.0)\n';

// ── Sampling: each coordinate is its own copy of the sub-DAG ─────────────────

// The wrapper roster. `normalize` and `truncate` already freshened before the
// fix (they reach the materialiser's iid LEAF fast path, which draws N*k at the
// worker); `weighted`, `logweighted` and `pushfwd` reach the composite fallback
// and produced the singular diagonal. All five are pinned so a change to which
// path a wrapper takes cannot silently reintroduce the defect.
const FRESHENING_ROSTER: Array<[string, string, number, number]> = [
  ['weighted',    'W = weighted(2.0, lawof(u))\nb ~ iid(W, 3)\n',                0.0, 1.0],
  ['logweighted', 'W = logweighted(0.5, lawof(u))\nb ~ iid(W, 3)\n',             0.0, 1.0],
  ['normalize',   'W = normalize(lawof(u))\nb ~ iid(W, 3)\n',                    0.0, 1.0],
  ['pushfwd',     'W = pushfwd(x -> x + 1.0, lawof(u))\nb ~ iid(W, 3)\n',        1.0, 1.0],
  // Standard normal truncated to [-1, 1]: mean 0 by symmetry, variance
  // 1 - 2φ(1)/(2Φ(1)-1) = 0.29112509477279314, which
  // `scipy.stats.truncnorm(-1, 1).var()` gives as 0.291125094772793.
  ['truncate',    'W = truncate(lawof(u), interval(-1.0, 1.0))\nb ~ iid(W, 3)\n', 0.0, 0.29112509477279314],
];

for (const [label, body, wantMean, wantVar] of FRESHENING_ROSTER) {
  test(`iid over ${label}(… lawof(u) …) freshens: Cov = 0, not Cov = Var`, async () => {
    const { mean, varr, cov } = await momentsOf(STD_NORMAL + body, 'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - wantMean) < MOMENT_TOL,
        `mean[${i}] = ${mean[i]}, want ${wantMean}`);
      assert.ok(Math.abs(varr[i] - wantVar) < MOMENT_TOL,
        `var[${i}] = ${varr[i]}, want ${wantVar}`);
    }
    // The defect's signature: cov equal to var. Zero is the product measure.
    for (let i = 1; i < 3; i++) {
      assert.ok(Math.abs(cov[i]) < MOMENT_TOL,
        `cov[0,${i}] = ${cov[i]}, want 0 (var is ${varr[i]}; cov == var is the `
        + 'singular diagonal this test exists to catch)');
    }
  });
}

test('a MULTI-AXIS iid over a wrapped reified law freshens every cell', async () => {
  // `size` as a vector is the same product measure over a rank-2 array
  // (§06: "$N = \\mathrm{prod}(\\text{size})$"), and it took the same fallback.
  const { varr, cov } = await momentsOf(
    STD_NORMAL + 'W = weighted(2.0, lawof(u))\nb ~ iid(W, [2, 2])\n', 'b', 4);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(varr[i] - 1.0) < MOMENT_TOL, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 4; i++) {
    assert.ok(Math.abs(cov[i]) < MOMENT_TOL, `cov[0,${i}] = ${cov[i]}`);
  }
});

test('a NESTED iid over a wrapped reified law freshens all six coordinates',
  async () => {
    const { varr, cov } = await momentsOf(
      STD_NORMAL + 'W = weighted(2.0, lawof(u))\nI = iid(W, 2)\nb ~ iid(I, 3)\n',
      'b', 6);
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(varr[i] - 1.0) < MOMENT_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 6; i++) {
      assert.ok(Math.abs(cov[i]) < MOMENT_TOL, `cov[0,${i}] = ${cov[i]}`);
    }
  });

// `iid` over a record law materialises as a table (G1). Pre-fix all k rows were
// byte-identical — the same defect, visible without any moment. BOTH spellings
// were broken at da7391c, the BARE one included: the locus is the composite
// fallback, which every one of these compositions routes to, not the wrapper.
const RECORD_SPELLINGS: Array<[string, string]> = [
  ['bare', 'J = joint(a = lawof(u), b = lawof(w))\n'],
  ['wrapped', 'J = joint(a = weighted(2.0, lawof(u)), b = weighted(2.0, lawof(w)))\n'],
];

for (const [label, decl] of RECORD_SPELLINGS) {
  test(`a G1 record iid over ${label} reified laws produces DISTINCT rows`,
    async () => {
      const { ctx } = ctxFor(STD_NORMAL
        + 'w ~ Normal(mu = 10.0, sigma = 1.0)\n' + decl
        + 'b ~ iid(J, 3)\n', 1);
      const m = await ctx.getMeasure('b');
      assert.ok(m.__table__, 'expected a table value');
      assert.equal(m.nrows, 3);
      for (const col of ['a', 'b']) {
        const vals = Array.from(m.columns[col].data as any).map(Number);
        assert.equal(vals.length, 3);
        assert.equal(new Set(vals).size, 3,
          `column ${col} rows are not distinct: ${vals.join(', ')}`);
      }
    });
}

// ── The two shapes that must NOT freshen ────────────────────────────────────

test('iid over a DISTRIBUTION still SHARES its stochastic parameter', async () => {
  // §06's own example `iid(Normal(mu = a, sigma = b), 100)` reads ONE `a` and
  // ONE `b`: the parameter is fixed before replication and sits outside the
  // product, so the marginals correlate. Closed form for u ~ N(0,1) and
  // b_i ~ N(u, 1): Var = 2, Cov = Var(u) = 1. The freshening fix must not
  // reach this — it is the direction that turns a correct number into noise.
  const { varr, cov } = await momentsOf(
    STD_NORMAL + 'b ~ iid(Normal(mu = u, sigma = 1.0), 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(varr[i] - 2.0) < MOMENT_TOL, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 3; i++) {
    assert.ok(Math.abs(cov[i] - 1.0) < MOMENT_TOL, `cov[0,${i}] = ${cov[i]}`);
  }
});

test('the REPEAT AXIS still shares an atom-level parameter across the k draws',
  async () => {
    // engine-concepts §22.4: with a stochastic upstream parameter in M, §06 is
    // "draw psi_i per atom, then k iid draws from M at that psi_i". `psi` is
    // reached as the weight ARGUMENT, never in measure position, so the
    // freshening carve-out must leave it tiled. Without the tiling the k inner
    // draws would each see their own psi — the right marginal, the wrong joint.
    const src = `
psi ~ Beta(alpha = 2.0, beta = 2.0)
zib = superpose(weighted(psi, Normal(mu = 0.0, sigma = 0.001)), weighted(1.0 - psi, Normal(mu = 5.0, sigma = 1.0)))
y ~ iid(zib, 4)
`;
    const { mean, varr, cov } = await momentsOf(src, 'y', 4);
    // CROSS-COORDINATE COVARIANCE is the statistic that separates a shared psi
    // from a freshened one: it is Var(E[Y | psi]) when psi is shared and 0 when
    // it is not, while the mean and variance are the same either way.
    //
    // Closed form by conditioning, psi ~ Beta(2, 2) (mean 1/2, var 1/20):
    // E[Y|psi] = 5(1-psi) so Cov(Y_i, Y_j) = 25·Var(psi) = 1.25;
    // Var(Y|psi) = psi·0.001² + (1-psi) + 25·psi(1-psi) so
    // E[Var(Y|psi)] = 0.5 + 25·(1/2 - 3/10) = 5.5 and Var(Y) = 6.75;
    // E[Y] = 2.5.
    //
    // This replaces an earlier witness that asserted the per-position means
    // stayed ORDERED and spread over 3 units (0.2579 / 1.5730 / 3.4082 /
    // 4.7392). That gradient was the branch PINNING, not shared-psi evidence:
    // matSuperpose stratified its component selection across the k slots, so
    // slot 0 favoured the psi-weighted spike and slot 3 the wide component.
    // With the selection independent per coordinate all four positions share
    // one marginal, which is what §06's product measure requires.
    //
    // Wider than this file's MOMENT_TOL: a covariance on a variance-6.75
    // variate at N = 20000 has a standard error around 0.05, and the measured
    // spread here is 1.154 to 1.207 against 1.239 to 1.251 at 200000 draws.
    const ZIB_TOL = 0.3;
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(mean[i] - 2.5) < ZIB_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 6.75) < ZIB_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 4; i++) {
      assert.ok(Math.abs(cov[i] - 1.25) < ZIB_TOL,
        `cov[0,${i}] = ${cov[i]}, want 1.25 — a covariance near 0 means psi was `
        + 'freshened per coordinate instead of tiled across the repeat block');
    }
  });

// ── The density path agrees with the fixed sampler ───────────────────────────
//
// Sampler/density disagreement is the catastrophic failure mode, so the density
// side is pinned at the same shapes. The density path already scored these as a
// product of independent coordinates BEFORE the fix — it was the sampler that
// disagreed with it, and with §06.
//
//   logpdf(N(0,1), x) = -0.5*log(2*pi) - x*x/2 = -0.9189385332046727 - x*x/2
//   sum over x = 0.1, 0.2, 0.3  = -2.826815599614018
//   plus 3*log(2)               = +2.0794415416798357  →  -0.7473740579341823

const SUM_LOGPDF = -2.826815599614018;
const PLUS_3LOG2 = -0.7473740579341823;

test('logdensityof(iid(weighted(2.0, lawof(u)), 3), x) is the product density',
  async () => {
    // §06 "Density of composed measures": "for `iid` always,
    // $\\log\\mathrm{densityof}(\\mathrm{iid}(M, n), x) = \\sum_i
    // \\log\\mathrm{densityof}(M, x_i)$", and weighted's density carries the
    // weight (§06 `weighted`: $\\mathrm{d}\\nu = \\text{weight} \\cdot
    // \\mathrm{d}M$), so each of the three coordinates contributes log 2.
    const got = await scoreOf(STD_NORMAL
      + 'W = weighted(2.0, lawof(u))\nM = iid(W, 3)\n'
      + 'ld = logdensityof(M, [0.1, 0.2, 0.3])\n');
    assert.ok(Math.abs(got - PLUS_3LOG2) < F64_TOL, `got ${got}`);
  });

test('logweighted scores identically at log-weight log 2', async () => {
  const got = await scoreOf(STD_NORMAL
    + 'W = logweighted(0.6931471805599453, lawof(u))\nM = iid(W, 3)\n'
    + 'ld = logdensityof(M, [0.1, 0.2, 0.3])\n');
  assert.ok(Math.abs(got - PLUS_3LOG2) < F64_TOL, `got ${got}`);
});

test('normalize and pushfwd wrappers score the unweighted product', async () => {
  const norm = await scoreOf(STD_NORMAL
    + 'W = normalize(lawof(u))\nM = iid(W, 3)\n'
    + 'ld = logdensityof(M, [0.1, 0.2, 0.3])\n');
  assert.ok(Math.abs(norm - SUM_LOGPDF) < F64_TOL, `normalize got ${norm}`);
  // A unit translation has |det J| = 1, so the pushforward density is the base
  // density at the preimage — the same three logpdf terms, shifted by 1.
  const push = await scoreOf(STD_NORMAL
    + 'W = pushfwd(x -> x + 1.0, lawof(u))\nM = iid(W, 3)\n'
    + 'ld = logdensityof(M, [1.1, 1.2, 1.3])\n');
  assert.ok(Math.abs(push - SUM_LOGPDF) < F64_TOL, `pushfwd got ${push}`);
});
