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
const { processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 20000;
const MOMENT_TOL = 0.06;
const F64_TOL = 1e-12;

const infer = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

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
    // draws would each see their own psi — the right marginal, the wrong joint —
    // and the mixture's per-coordinate means would collapse toward each other.
    const src = `
psi ~ Beta(alpha = 2.0, beta = 2.0)
zib = superpose(weighted(psi, Normal(mu = 0.0, sigma = 0.001)), weighted(1.0 - psi, Normal(mu = 5.0, sigma = 1.0)))
y ~ iid(zib, 4)
`;
    const { mean } = await momentsOf(src, 'y', 4);
    // The per-position means stay ORDERED and far apart, the fingerprint of a
    // shared psi driving a within-atom resample. (Pinned as a regression
    // witness, not as a spec claim about this shape's own correctness.)
    assert.ok(mean[0] < mean[1] && mean[1] < mean[2] && mean[2] < mean[3],
      'repeat-axis means are no longer ordered: ' + mean.join(', '));
    assert.ok(mean[3] - mean[0] > 3.0,
      'repeat-axis mean spread collapsed: ' + mean.join(', '));
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

// ── The static sharing walk agrees with the fixed sampler ────────────────────
//
// typeinfer's §06 ancestry walk skipped an `iid` only when its measure argument
// was a BARE reified law, so the wrapped spelling kept descending and reported
// sharing. That static error was the right posture while the sampler returned
// the diagonal — it was the only thing standing between a user and that number.
// With the sampler freshening, the report is a FALSE rejection of a legal
// program, so the skip widens to the same wrapper roster the sampler freshens.

const W1_PREAMBLE = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
w ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
`;

const WRAPPED_COMPONENTS: Array<[string, string]> = [
  ['weighted',    'M = iid(weighted(2.0, lawof(u)), 3)'],
  ['logweighted', 'M = iid(logweighted(0.5, lawof(u)), 3)'],
  ['normalize',   'M = iid(normalize(lawof(u)), 3)'],
  ['truncate',    'M = iid(truncate(lawof(u), interval(-1.0, 1.0)), 3)'],
  ['pushfwd',     'M = iid(pushfwd(x -> x + 1.0, lawof(u)), 3)'],
  ['nested iid',  'M = iid(iid(weighted(2.0, lawof(u)), 2), 3)'],
  // §04 "Aliasing is just assignment" — the chain resolves.
  ['alias chain', 'L = lawof(u)\nL2 = L\nW = weighted(2.0, L2)\nM = iid(W, 3)'],
  // `joint` / `record` replicate every component, so each one prunes its own
  // law. The BARE record spelling was singular at da7391c for the same reason
  // the wrapped one was — it takes the same composite fallback — and the
  // sampler freshens it now, so the walk must stop reporting it.
  ['bare record',    'M = iid(joint(a = lawof(u), b = lawof(w)), 3)'],
  ['wrapped record', 'M = iid(joint(a = weighted(2.0, lawof(u)), b = weighted(2.0, lawof(w))), 3)'],
  ['positional joint', 'M = iid(joint(lawof(u), lawof(w)), 3)'],
];

for (const [label, decl] of WRAPPED_COMPONENTS) {
  test(`a component that is an iid over ${label}(… lawof(u) …) is not reported`,
    () => {
      assert.deepEqual(
        infer(W1_PREAMBLE + decl + '\nKJ = joint(p = K1, q = M)\n'), []);
    });
}

test('the wrapper ARGUMENTS are still walked — a shared weight node reports',
  () => {
    // `p` is the weight, not the replicated law: it is drawn once and every copy
    // reads it, exactly like a distribution parameter. Widening the skip over
    // the whole `iid` subtree would have dropped this diagnostic.
    const errors = infer(`
z = elementof(reals)
p ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = p, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(weighted(p, lawof(u)), 3)
KJ = joint(p = K1, q = M)
`);
    assert.ok(errors.some((m: string) => /share the stochastic node 'p'/.test(m)),
      'got: ' + errors.join(' | '));
  });

test('a DISTRIBUTION component of a replicated joint is still walked', () => {
  // Pruning a `joint` component by component must not prune a component that
  // replicates nothing: `Normal(mu = p, …)` reads one `p` fixed before
  // replication, so the sharing report stands for that component alone.
  const errors = infer(`
z = elementof(reals)
p ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = p, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(joint(a = lawof(u), b = Normal(mu = p, sigma = 1.0)), 3)
KJ = joint(p = K1, q = M)
`);
  assert.ok(errors.some((m: string) => /share the stochastic node 'p'/.test(m)),
    'got: ' + errors.join(' | '));
});

test('a shared node in a TRUNCATION region is still walked', () => {
  const errors = infer(`
z = elementof(reals)
p ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = p, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(truncate(lawof(u), interval(p, p + 1.0)), 3)
KJ = joint(p = K1, q = M)
`);
  assert.ok(errors.some((m: string) => /share the stochastic node 'p'/.test(m)),
    'got: ' + errors.join(' | '));
});

test('superpose is NOT in the widened roster — its sampler does not freshen',
  () => {
    // Measured at this commit: `iid(superpose(weighted(0.5, lawof(u)),
    // weighted(0.5, lawof(w))), 3)` pins coordinate 0 to one branch and
    // coordinate 2 to the other (per-coordinate means -3.0 / 0.0 / +3.0 for
    // laws centred at -3 and +3, where a product of mixtures has mean 0 in
    // every coordinate). The sequencing rule is that the sampler is fixed
    // before the static report is removed, so the report stays.
    const errors = infer(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = iid(superpose(weighted(0.5, lawof(u)), weighted(0.5, Normal(mu = 0.0, sigma = 1.0))), 3)
KJ = joint(p = K1, q = M)
`);
    assert.ok(errors.length > 0, 'expected the sharing report to stay');
  });
