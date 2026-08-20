'use strict';

// `iid` over a `kchain`/`jointchain` copies the chain's reified BASE law per
// coordinate, and `iid` over a TUPLE-variate measure refuses instead of
// crashing.
//
// SPEC ANCHOR — flatppl-design docs/06-measure-algebra.md, "Joint
// composition", `iid` entry, quoted verbatim: "**`iid(M, size)`** — the
// product measure $M^{\otimes N}$ over arrays of shape `size`, where `N =
// prod(size)`. … When `M` is a reified law, each of the $N$ copies carries its
// own copy of the reified sub-DAG, stochastic ancestors included; `iid` never
// shares nodes between copies."
//
// SPEC ANCHOR — same file, "Dependent composition", `kchain` entry, quoted
// verbatim: "**`kchain(M, K1, K2, ...)`** — left-associative Kleisli
// composition (monadic bind). Keeps only the last kernel's variates,
// marginalizing out all intermediate variates."
//
// SPEC ANCHOR — same entry, the stochastic-node equivalence: `model =
// kchain(M1, K2, K3)` is equivalent to `a ~ M1; b ~ K2(a); c ~ K3([a, b]);
// model = lawof(c)`. So the chain's base measure is part of M's own sub-DAG,
// which `iid` copies — it is not an atom-level parameter the copies share.
//
// DEFECT 1, which the freshening rows pin. `_reifiedVariatesUnder` walked only
// the `from` / `fromNames` / `elems` / `fields` derivation edges. A
// `kchain`/`jointchain` records its components as an explicit STEP LIST
// instead, so the walk stopped at the chain and never reached the base law's
// variate. That variate stayed classified as an atom-level value draw, so the
// repeat axis TILED it: one draw shared by all k coordinates. With `z ~
// Normal(0, sqrt(10))` and `C = kchain(lawof(z), fn(Normal(mu = _, sigma =
// 1)))`, `iid(C, 3)` measured cross-coordinate covariance 10.0130 at 200 000
// draws where §06's product measure requires 0 — the whole base draw shared,
// only the kernel noise fresh. Per-coordinate mean and variance were already
// right (0 and 11), so the marginals hid it entirely and only the JOINT
// showed it.
//
// DEFECT 2, pinned by the refusal rows. `iid` over a measure whose variate is
// a TUPLE — a positional `joint(M1, M2)`, or a `jointchain` retaining all its
// variates — threw the internal `iid: inner measure for C produced no
// samples`. The engine has no value for `array(k) of tuple(…)`: a record
// variate becomes a table because its components are NAMED, and a tuple's are
// not. It now refuses with a message naming the shape and the spec-supported
// spelling that does materialise.
//
// ORACLES ARE INDEPENDENT of this engine. Every target below is a closed form
// derived by hand and cross-checked by Monte-Carlo simulation in Julia at
// 4 000 000 draws (a hand-written mixture/chain simulation, not a FlatPPL
// path). No engine output is pinned as an expected value anywhere in this
// file, and flatppl-rust evaluates no density and was not consulted.
//
// TOLERANCES. The materialiser's stream is deterministic run to run, so these
// are Monte-Carlo slack, not flakiness insurance. At N = 20000 on a variate of
// variance 11 the standard error is about sqrt(11/N) = 0.023 on a mean and
// sqrt(2*121/N) = 0.11 on a variance or covariance, so the tolerances below
// are roughly 3 to 6 sigma. Each covariance target is separated from the
// defect's value by at least 0.5 (the superpose base) and at most 10 (the pure
// witness). The determinism does NOT come from `_ctx-factory`'s seed — the
// factory sets a SCALAR `rootKey` where the engine's contract is the two-lane
// PhiloxKey `[k0, k1]`, so `foldIn` collapses; that is the defect carded in
// flatppl-dev/TODO-flatppl-js.md (`worker.ts:240`). Quote the draw count with
// any figure taken from here.

const test = require('node:test');
const assert = require('node:assert');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 20000;
const MEAN_TOL = 0.15;
const VAR_TOL = 0.35;
const COV_TOL = 0.35;

// sqrt(10), so a base of variance 10 marginalised through `Normal(mu = _, 1)`
// has variance 11 — far enough from the base's own 10 that a missing kernel
// step would show.
const SQRT10 = '3.1622776601683795';

// Per-coordinate means and variances plus the covariance of coordinate 0
// against each coordinate, read off a materialised iid measure's atom-major
// samples.
async function momentsOf(src: string, name: string, k: number) {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  const flat = Array.from((m.samples || (m.value && m.value.data)) as any)
    .map(Number);
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

// Assert every coordinate's marginal and every cross-coordinate covariance.
function assertMoments(
  label: string, got: any, wantMean: number, wantVar: number, wantCov: number,
) {
  for (let i = 0; i < got.mean.length; i++) {
    assert.ok(Math.abs(got.mean[i] - wantMean) < MEAN_TOL,
      `${label}: mean[${i}] = ${got.mean[i]}, want ${wantMean}`);
    assert.ok(Math.abs(got.varr[i] - wantVar) < VAR_TOL,
      `${label}: var[${i}] = ${got.varr[i]}, want ${wantVar}`);
  }
  for (let i = 1; i < got.cov.length; i++) {
    assert.ok(Math.abs(got.cov[i] - wantCov) < COV_TOL,
      `${label}: cov[0,${i}] = ${got.cov[i]}, want ${wantCov}`);
  }
}

const refusalOf = async (src: string, name: string) => {
  try {
    const { ctx } = ctxFor(src, 1);
    await ctx.getMeasure(name);
  } catch (e: any) { return e.message; }
  return null;
};

// ── The reified base law freshens per coordinate ─────────────────────────────

// `z ~ Normal(0, sqrt(10))`, `C = kchain(lawof(z), fn(Normal(mu = _, 1)))`.
// Each coordinate of `iid(C, 3)` carries its own copy of the reified sub-DAG,
// so its own `z`. Marginal: Var = 10 + 1 = 11, E = 0. Joint: the coordinates
// are independent, so Cov = 0. Shared, `Cov = Var(z) = 10` — the defect's
// measured 10.0130. The covariance carries the whole discrimination; mean and
// variance are 0 and 11 either way.
const CHAIN_MEAN = 0.0;
const CHAIN_VAR = 11.0;

const Z_BASE = `z ~ Normal(mu = 0.0, sigma = ${SQRT10})\n`;
const Z_CHAIN = Z_BASE
  + 'C = kchain(lawof(z), fn(Normal(mu = _, sigma = 1.0)))\n';

// `u ~ N(-3, 1)`, `w ~ N(3, 1)`, S their equal-weight superposition. Marginal
// of the mixture is mean 0 / var 10 (Distributions.jl `MixtureModel`), so
// through the chain's unit kernel it is mean 0 / var 11 as well. Shared `u`
// and `w` leave `P(same branch) * Var(within branch) = (1/4 + 1/4) * 1 = 0.5`
// of covariance — the defect's measured 0.4791.
const MIX_CHAIN = 'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
  + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
  + 'S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n'
  + 'C = kchain(S, fn(Normal(mu = _, sigma = 1.0)))\n';

// Wrapper spellings all route `iid` to the composite fallback around the same
// chain, and every one measured cov 10 before the fix.
const FRESH_SPELLINGS: Array<[string, string, number, number]> = [
  ['kchain(lawof(z), K)', Z_CHAIN + 'b ~ iid(C, 3)\n', 3, CHAIN_VAR],
  ['kchain(superpose, K)', MIX_CHAIN + 'b ~ iid(C, 3)\n', 3, CHAIN_VAR],
  ['normalize(kchain(lawof(z), K))',
    Z_CHAIN + 'M = normalize(C)\nb ~ iid(M, 3)\n', 3, CHAIN_VAR],
  ['multi-axis iid(kchain(lawof(z), K), [2, 3])',
    Z_CHAIN + 'b ~ iid(C, [2, 3])\n', 6, CHAIN_VAR],
  ['nested iid(iid(kchain(lawof(z), K), 2), 3)',
    Z_CHAIN + 'b ~ iid(iid(C, 2), 3)\n', 6, CHAIN_VAR],
];

for (const [label, src, k, wantVar] of FRESH_SPELLINGS) {
  test(`iid over ${label}: each coordinate draws its own base`, async () => {
    assertMoments(label, await momentsOf(src, 'b', k),
      CHAIN_MEAN, wantVar, 0.0);
  });
}

// A unit translation shifts the mean and leaves the variance and covariance,
// so it checks that the freshening survives an order-preserving wrapper
// between the `iid` and the chain.
test('iid over pushfwd(kchain(lawof(z), K)): base freshens through the map',
  async () => {
    assertMoments('pushfwd',
      await momentsOf(Z_CHAIN + 'M = pushfwd(x -> x + 1.0, C)\nb ~ iid(M, 3)\n',
        'b', 3),
      1.0, CHAIN_VAR, 0.0);
  });

// ── Must-stay-shared controls ────────────────────────────────────────────────
//
// The wrong fix freshens the whole subtree. A node reached only as a
// distribution PARAMETER or a weight is not the replicated sub-DAG — §06's own
// `iid` example is `obs ~ iid(Normal(mu = a, sigma = b), 100)`, where `a` and
// `b` are read once. Cross-coordinate covariance separates the two: it is
// `Var(E[X | parameter])` when the parameter is shared and 0 when it is
// freshened, so each row below asserts a NONZERO covariance.

// `a ~ N(0, 1)` is the base distribution's `mu`, reached through `distIR`.
// X_i = a + e_i + f_i with e_i the base draw and f_i the kernel noise, so
// Var = 3 and Cov = Var(a) = 1. Freshened it would be 0.
test('a stochastic base PARAMETER of the chain stays shared', async () => {
  assertMoments('base mu', await momentsOf(
    'a ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'C = kchain(Normal(mu = a, sigma = 1.0), fn(Normal(mu = _, sigma = 1.0)))\n'
    + 'b ~ iid(C, 3)\n', 'b', 3), 0.0, 3.0, 1.0);
});

// `m ~ N(0, 1)` shifts BOTH branches, so E[Y|m] = m and Var(Y|m) = 1 + 9 + 1 =
// 11: Var = 12, Cov = Var(m) = 1. Before the fix this row measured 1.5632,
// because the shared `m` contributed its 1 and the shared reified `u`, `w`
// added `Var((u + w) / 2 - m) = 0.5` on top — so this row discriminates in
// BOTH directions at once: it fails at 1.5 if the base does not freshen, and
// at 0 if the parameter wrongly does.
test('a shared component PARAMETER survives while the branches freshen',
  async () => {
    assertMoments('component mean', await momentsOf(
      'm ~ Normal(mu = 0.0, sigma = 1.0)\n'
      + 'u ~ Normal(mu = m - 3.0, sigma = 1.0)\n'
      + 'w ~ Normal(mu = m + 3.0, sigma = 1.0)\n'
      + 'S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n'
      + 'C = kchain(S, fn(Normal(mu = _, sigma = 1.0)))\n'
      + 'b ~ iid(C, 3)\n', 'b', 3), 0.0, 12.0, 1.0);
  });

// `psi ~ Uniform(interval(0, 1))` is the mixing WEIGHT, reached through
// `weightIR`. E[Y|psi] = 3 - 6psi, so Cov = Var(3 - 6psi) = 36/12 = 3, and
// Var(Y|psi) = 1 + 36 psi (1 - psi) gives Var = 7 + 3 + 1 = 11. This is the
// sharpest control: before the fix it measured 3.6667 (3 from the shared psi
// plus `E[psi^2] + E[(1-psi)^2] = 2/3` from the shared `u`, `w`), so it too
// discriminates in both directions. `normalize` is the identity on S — its
// component weights sum to 1 — and only states the mass `~` requires.
test('a shared mixing WEIGHT under the chain stays shared', async () => {
  assertMoments('mixing weight', await momentsOf(
    'psi ~ Uniform(support = interval(0.0, 1.0))\n'
    + 'u ~ Normal(mu = -3.0, sigma = 1.0)\n'
    + 'w ~ Normal(mu = 3.0, sigma = 1.0)\n'
    + 'S = normalize(superpose(weighted(psi, lawof(u)), '
    + 'weighted(1.0 - psi, lawof(w))))\n'
    + 'C = kchain(S, fn(Normal(mu = _, sigma = 1.0)))\n'
    + 'b ~ iid(C, 3)\n', 'b', 3), 0.0, 11.0, 3.0);
});

// ── Must-not-change controls ─────────────────────────────────────────────────

// A plain distribution base reifies nothing, so the walk has nothing to reach
// and this shape was already correct. It pins that the widened walk did not
// disturb it.
test('a plain DISTRIBUTION base under the chain is unaffected', async () => {
  assertMoments('dist base', await momentsOf(
    `C = kchain(Normal(mu = 0.0, sigma = ${SQRT10}), `
    + 'fn(Normal(mu = _, sigma = 1.0)))\nb ~ iid(C, 3)\n', 'b', 3),
  CHAIN_MEAN, CHAIN_VAR, 0.0);
});

// Without an `iid` there is no repeat axis and no tiling, so the chain's own
// marginal must be untouched.
test('a bare chain with no iid keeps its marginal', async () => {
  assertMoments('bare chain', await momentsOf(Z_CHAIN + 'b ~ C\n', 'b', 1),
    CHAIN_MEAN, CHAIN_VAR, CHAIN_VAR);
});

// ── iid over a TUPLE variate refuses instead of crashing ─────────────────────

const TUPLE_SHAPES: Array<[string, string]> = [
  ['jointchain retaining both variates',
    'C = jointchain(Normal(mu = 0.0, sigma = 1.0), '
    + 'fn(Normal(mu = _, sigma = 1.0)))\nb ~ iid(C, 3)\n'],
  ['jointchain over a superpose base',
    'u ~ Normal(mu = -3.0, sigma = 1.0)\nw ~ Normal(mu = 3.0, sigma = 1.0)\n'
    + 'S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n'
    + 'C = jointchain(S, fn(Normal(mu = _, sigma = 1.0)))\nb ~ iid(C, 3)\n'],
  ['positional joint',
    'C = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 5.0, sigma = 1.0))\n'
    + 'b ~ iid(C, 3)\n'],
];

for (const [label, src] of TUPLE_SHAPES) {
  test(`iid over a ${label} refuses, naming the shape`, async () => {
    const msg = await refusalOf(src, 'b');
    assert.ok(msg, `${label}: expected a refusal`);
    assert.match(msg!, /TUPLE variate/,
      `${label}: refusal must name the shape, got: ${msg}`);
    assert.match(msg!, /jointchain\(name1 = M, name2 = K1/,
      `${label}: refusal must name the keyword form, got: ${msg}`);
    assert.doesNotMatch(msg!, /produced no samples/,
      `${label}: the internal message must not surface, got: ${msg}`);
  });
}

// The spelling the refusal recommends. §06: "`jointchain(name1 = M, name2 =
// K1, ...)` names the component variates, producing a measure over a space of
// records." An `iid` of a record variate is a TABLE (§03), so this
// materialises. Oracle, closed form: `a ~ N(0, 1)` and `b | a ~ N(a, 1)` give
// E[a] = E[b] = 0, Var(a) = 1, Var(b) = 2, Cov(a, b) = 1. The lag-1
// autocovariance down the `a` column is 0 because the k rows are iid — that is
// what makes this a real check on the recommended route and not just a
// smoke test.
test('the keyword form the refusal recommends materialises correctly',
  async () => {
    const K = 20000;
    const { ctx } = ctxFor(
      'C = jointchain(a = Normal(mu = 0.0, sigma = 1.0), '
      + 'b = fn(Normal(mu = _, sigma = 1.0)))\nr ~ iid(C, ' + K + ')\n', 1);
    const m = await ctx.getMeasure('r');
    assert.equal(m.nrows, K, 'row count');
    const A = Array.from(m.columns.a.data as any).map(Number);
    const B = Array.from(m.columns.b.data as any).map(Number);
    const mu = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;
    const ma = mu(A);
    const mb = mu(B);
    const va = mu(A.map((v) => (v - ma) ** 2));
    const vb = mu(B.map((v) => (v - mb) ** 2));
    const cab = mu(A.map((v, i) => (v - ma) * (B[i] - mb)));
    const lag1 = mu(A.slice(0, -1).map((v, i) => (v - ma) * (A[i + 1] - ma)));
    assert.ok(Math.abs(ma) < MEAN_TOL, `E[a] = ${ma}`);
    assert.ok(Math.abs(mb) < MEAN_TOL, `E[b] = ${mb}`);
    assert.ok(Math.abs(va - 1.0) < VAR_TOL, `Var(a) = ${va}`);
    assert.ok(Math.abs(vb - 2.0) < VAR_TOL, `Var(b) = ${vb}`);
    assert.ok(Math.abs(cab - 1.0) < COV_TOL, `Cov(a, b) = ${cab}`);
    assert.ok(Math.abs(lag1) < COV_TOL, `lag-1 autocov(a) = ${lag1}`);
  });
