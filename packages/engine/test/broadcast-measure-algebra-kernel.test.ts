'use strict';

// =====================================================================
// broadcast-measure-algebra-kernel.test.ts
// =====================================================================
//
// A broadcast whose KERNEL BODY is a §06 measure-ALGEBRA tree — the
// mixture spelling §06 recommends under `superpose`, "To build a
// normalized mixture distribution, use `normalize(superpose(weighted(w1,
// M1), weighted(w2, M2)))`" — must both SCORE and SAMPLE. Per §04
// sec:functionof-measure a `functionof` over a measure node is a
// transition kernel, and §04 sec:broadcasting makes `broadcast(K, col)`
// "an array-valued measure: the independent product measure of the kernel
// applications at each array position".
//
// Density already worked for a literal weight column; SAMPLING did not —
// the binding fell through every composite-body recogniser to a value
// `evaluate`, and the sampler met the measure op in value position
// ("call op 'normalize' not evaluable in sampler context"). The
// `measure_algebra` recogniser + executor close that.
//
// ORACLES. Every density number is a closed-form scipy value, computed
// independently of the engine:
//   mixture  log p(x) = log(w·pdf_Exponential(x; scale=lam)
//                          + (1-w)·pdf_Normal(x; mu, sigma))
//   truncate log p(x) = log pdf_Normal(x; mu, 1)          (§06: truncate
//            restricts, it does not renormalize)
//   weighted log p(x) = log pdf_Normal(x; 0, 1) + log w
// Sampling is checked against the mixture's closed-form mean and
// variance with explicit Monte-Carlo margins, never against the engine's
// own output.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const LAM = 40.0;
const S_MU = 100.0;
const S_SIGMA = 2.0;

const HEAD = `
lam = ${LAM}
bkg = Exponential(1.0 / lam)
sig = Normal(${S_MU}, ${S_SIGMA})
`;

const MIX_LAMBDA =
  'mix = p -> normalize(superpose(weighted(p, bkg), weighted(1.0 - p, sig)))\n';
// §04 sec:functionof "Lambda notation": `p -> expr` is shorthand for
// `functionof(expr', p = _p_)`. Both spellings must behave identically.
const MIX_FUNCTIONOF =
  'mix = functionof(normalize(superpose(weighted(_p_, bkg), '
  + 'weighted(1.0 - _p_, sig))), p = _p_)\n';

const W_EV = [0.8, 0.5, 0.2];
const DATA = [32.0, 99.0, 55.0];

// scipy: log(w·expon.pdf(x, scale=40) + (1-w)·norm.pdf(x, 100, 2)) summed.
const ORACLE_A = -13.803691384344681;

async function scoreOf(src: string, target = '__score__'): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(target);
  const s = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error('no data for ' + target);
  return s[0];
}

function relErr(got: number, want: number): number {
  return Math.abs((got - want) / want);
}

// ---------------------------------------------------------------------
// Density — both licensed kernel spellings, plus the hand-unrolled twin
// ---------------------------------------------------------------------

for (const [label, mix] of [
  ['arrow lambda', MIX_LAMBDA],
  ['functionof + placeholder', MIX_FUNCTIONOF],
] as Array<[string, string]>) {
  test(`density: broadcast of a measure-algebra kernel (${label}) `
    + 'matches the closed-form mixture', async () => {
    const src = HEAD + `w_ev = [${W_EV.join(', ')}]\n` + mix
      + `E ~ mix.(w_ev)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`;
    const got = await scoreOf(src);
    assert.ok(relErr(got, ORACLE_A) < 1e-12,
      `score ${got} vs scipy ${ORACLE_A} (rel ${relErr(got, ORACLE_A)})`);
  });
}

test('density: `broadcast(mix, col)` and the `mix.(col)` sugar agree', async () => {
  const base = HEAD + `w_ev = [${W_EV.join(', ')}]\n` + MIX_LAMBDA;
  const dotted = await scoreOf(base
    + `E ~ mix.(w_ev)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`);
  const explicit = await scoreOf(base
    + `E ~ broadcast(mix, w_ev)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`);
  assert.equal(dotted, explicit);
});

test('regression: the hand-unrolled per-weight `iid` form still scores '
  + 'the same closed-form value', async () => {
  // One `iid(normalize(superpose(…)), 1)` block per event — the spelling a
  // model had to use before the broadcast form scored and sampled. It must
  // keep hitting the same oracle, so the fix cannot have moved it.
  let src = HEAD;
  const parts: string[] = [];
  for (let j = 0; j < W_EV.length; j++) {
    src += `m_${j} = normalize(superpose(weighted(${W_EV[j]}, bkg), `
      + `weighted(${1.0 - W_EV[j]}, sig)))\n`;
    parts.push(`logdensityof(m_${j}, ${DATA[j]})`);
  }
  src += '__score__ = ' + parts.join(' + ') + '\n';
  const got = await scoreOf(src);
  assert.ok(relErr(got, ORACLE_A) < 1e-12,
    `unrolled score ${got} vs scipy ${ORACLE_A} (rel ${relErr(got, ORACLE_A)})`);
});

// ---------------------------------------------------------------------
// Density — a kernel BOUNDARY inside the kernel body
// ---------------------------------------------------------------------

// scipy, same formula at lam = 40 with weights [0.8, 0.8, 0.6].
const ORACLE_B = -13.586548476027453;

const BOUNDARY_HEAD = `
lam ~ Uniform(interval(1.0, 100.0))
bkg = Exponential(1.0 / lam)
sig = Normal(${S_MU}, ${S_SIGMA})
`;
const THETA = 'record(lam = 40.0)';

test('density: a component parameter that is a kernelof BOUNDARY resolves '
  + 'per atom inside the cell sub-walk', async () => {
  // `lam` reaches the density walker as a per-atom refArray, not a
  // fixed-phase value. The cell sub-walk runs at N=1, so it needs atom i's
  // slice of every such ref — without it the walk reported "unbound self
  // reference 'lam'".
  const src = BOUNDARY_HEAD + 'w_ev = [0.8, 0.8, 0.6]\n' + MIX_LAMBDA + `
E ~ mix.(w_ev)
K = kernelof(record(E = E), lam = lam)
L = likelihoodof(K, record(E = [${DATA.join(', ')}]))
__score__ = logdensityof(L, ${THETA})
`;
  const got = await scoreOf(src);
  assert.ok(relErr(got, ORACLE_B) < 1e-12,
    `score ${got} vs scipy ${ORACLE_B} (rel ${relErr(got, ORACLE_B)})`);
});

test('density: the boundary form agrees with its hand-unrolled `iid` twin',
  async () => {
    const unrolled = await scoreOf(BOUNDARY_HEAD + `
w = [0.8, 0.6]
E1 ~ iid(normalize(superpose(weighted(w[1], bkg), weighted(1.0 - w[1], sig))), 2)
E2 ~ iid(normalize(superpose(weighted(w[2], bkg), weighted(1.0 - w[2], sig))), 1)
K = kernelof(record(E1 = E1, E2 = E2), lam = lam)
L = likelihoodof(K, record(E1 = [${DATA[0]}, ${DATA[1]}], E2 = [${DATA[2]}]))
__score__ = logdensityof(L, ${THETA})
`);
    assert.ok(relErr(unrolled, ORACLE_B) < 1e-12,
      `unrolled ${unrolled} vs scipy ${ORACLE_B}`);
  });

test('density: a per-event weight column INDEXED out of a latent-dependent '
  + 'per-dataset vector scores', async () => {
  // The shape a real per-dataset model needs: `w` is one entry per dataset
  // and derived from a latent, `w_ev` repeats it per event via §07 `get`
  // with an index array.
  const src = `
lam ~ Uniform(interval(1.0, 100.0))
bkg = Exponential(1.0 / lam)
sig = Normal(${S_MU}, ${S_SIGMA})
nu_B = [4.0, 3.0]
nu_S = [1.0, 2.0]
w = nu_B ./ (nu_B .+ nu_S)
dataset_of_event = cat(fill(1, 2), fill(2, 1))
w_ev = w[dataset_of_event]
` + MIX_LAMBDA + `
E ~ mix.(w_ev)
K = kernelof(record(E = E), lam = lam)
L = likelihoodof(K, record(E = [${DATA.join(', ')}]))
__score__ = logdensityof(L, ${THETA})
`;
  const got = await scoreOf(src);
  assert.ok(relErr(got, ORACLE_B) < 1e-12,
    `indexed-weight score ${got} vs scipy ${ORACLE_B} (rel ${relErr(got, ORACLE_B)})`);
});

// ---------------------------------------------------------------------
// Density — the other measure-algebra body shapes
// ---------------------------------------------------------------------

const TRUNC_BODY = `
mu_v = [0.0, 1.0]
K = m -> truncate(Normal(m, 1.0), interval(-1.0, 2.0))
X ~ K.(mu_v)
__score__ = logdensityof(X, [0.5, 1.5])
`;

test('density: a `truncate` kernel body restricts without renormalizing '
  + '(spec §06)', async () => {
  const oracle = -2.0878770664093453;   // Σ log pdf_Normal(x_j; mu_j, 1)
  const got = await scoreOf(TRUNC_BODY);
  assert.ok(relErr(got, oracle) < 1e-12, `truncate score ${got} vs ${oracle}`);
});

test('density: a `normalize(truncate(…))` kernel body divides by each '
  + 'cell\'s own region mass', async () => {
  const oracle = -1.68754447776042;     // above minus Σ log Z_j
  const got = await scoreOf(`
mu_v = [0.0, 1.0]
K = m -> normalize(truncate(Normal(m, 1.0), interval(-1.0, 2.0)))
X ~ K.(mu_v)
__score__ = logdensityof(X, [0.5, 1.5])
`);
  assert.ok(relErr(got, oracle) < 1e-12, `normalize-truncate score ${got} vs ${oracle}`);
});

test('density: a bare `weighted` kernel body scales each cell\'s density',
  async () => {
    const oracle = -1.2961175971812904;   // Σ [log pdf_Normal(x_j;0,1) + log w_j]
    const got = await scoreOf(`
w_v = [2.0, 3.0]
K = a -> weighted(a, Normal(0.0, 1.0))
X ~ K.(w_v)
__score__ = logdensityof(X, [0.5, 1.5])
`);
    assert.ok(relErr(got, oracle) < 1e-12, `weighted score ${got} vs ${oracle}`);
  });

test('density: a bare `logweighted` kernel body shifts each cell\'s log '
  + 'density', async () => {
  const oracle = -1.2961175971812904;   // same weights, given in log space
  const got = await scoreOf(`
lw_v = [log(2.0), log(3.0)]
K = a -> logweighted(a, Normal(0.0, 1.0))
X ~ K.(lw_v)
__score__ = logdensityof(X, [0.5, 1.5])
`);
  assert.ok(relErr(got, oracle) < 1e-12, `logweighted score ${got} vs ${oracle}`);
});

// ---------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------

// Closed-form moments of w·Exponential(scale=lam) + (1-w)·Normal(mu, sigma).
function mixMoments(w: number) {
  const mean = w * LAM + (1 - w) * S_MU;
  const m2 = w * 2 * LAM * LAM + (1 - w) * (S_MU * S_MU + S_SIGMA * S_SIGMA);
  const m3 = w * 6 * LAM ** 3 + (1 - w) * (S_MU ** 3 + 3 * S_MU * S_SIGMA ** 2);
  const m4 = w * 24 * LAM ** 4 + (1 - w)
    * (S_MU ** 4 + 6 * S_MU ** 2 * S_SIGMA ** 2 + 3 * S_SIGMA ** 4);
  const varr = m2 - mean * mean;
  const mu4c = m4 - 4 * mean * m3 + 6 * mean * mean * m2 - 3 * mean ** 4;
  return { mean, varr, sd: Math.sqrt(varr), mu4c };
}

test('sampling: each broadcast cell reproduces its own mixture\'s '
  + 'closed-form mean and variance, and the cells are independent',
async () => {
  const N = 200000;
  const src = HEAD + `w_ev = [${W_EV.join(', ')}]\n` + MIX_LAMBDA + 'E ~ mix.(w_ev)\n';
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('E');
  assert.deepEqual(m.value.shape, [N, W_EV.length],
    'shape is [atoms, broadcast cells]');
  // A normalized cell is a probability measure, so the product measure
  // carries no importance event — the same `logWeights: null` a
  // builtin-distribution head returns.
  assert.equal(m.logWeights, null, 'a normalized mixture leaves no weights');
  assert.equal(m.logTotalmass, 0, 'a product of probability measures has mass 1');

  const K = W_EV.length;
  const cols: Float64Array[] = [];
  for (let j = 0; j < K; j++) {
    const c = new Float64Array(N);
    for (let i = 0; i < N; i++) c[i] = m.value.data[i * K + j];
    cols.push(c);
  }
  for (let j = 0; j < K; j++) {
    const cf = mixMoments(W_EV[j]);
    let s = 0;
    for (let i = 0; i < N; i++) s += cols[j][i];
    const mean = s / N;
    let s2 = 0;
    for (let i = 0; i < N; i++) s2 += (cols[j][i] - mean) ** 2;
    const varr = s2 / (N - 1);
    // 5 standard errors: se(mean) = sd/sqrt(N),
    // se(var) = sqrt((mu4c - var^2)/N).
    const seMean = cf.sd / Math.sqrt(N);
    const seVar = Math.sqrt((cf.mu4c - cf.varr * cf.varr) / N);
    assert.ok(Math.abs(mean - cf.mean) < 5 * seMean,
      `cell ${j} mean ${mean} vs ${cf.mean} (5 se = ${5 * seMean})`);
    assert.ok(Math.abs(varr - cf.varr) < 5 * seVar,
      `cell ${j} var ${varr} vs ${cf.varr} (5 se = ${5 * seVar})`);
  }
  // §04: kernel broadcast is the INDEPENDENT product measure. Under
  // independence Pearson r has se ≈ 1/sqrt(N); 5 se is the margin.
  const margin = 5 / Math.sqrt(N);
  for (let a = 0; a < K; a++) {
    for (let b = a + 1; b < K; b++) {
      let ma = 0; let mb = 0;
      for (let i = 0; i < N; i++) { ma += cols[a][i]; mb += cols[b][i]; }
      ma /= N; mb /= N;
      let cab = 0; let va = 0; let vb = 0;
      for (let i = 0; i < N; i++) {
        const da = cols[a][i] - ma; const db = cols[b][i] - mb;
        cab += da * db; va += da * da; vb += db * db;
      }
      const r = cab / Math.sqrt(va * vb);
      assert.ok(Math.abs(r) < margin,
        `corr(cell ${a}, cell ${b}) = ${r}, expected |r| < ${margin}`);
    }
  }
});

test('sampling: `rand(state, iid(lawof(E), n))` draws the broadcast\'s '
  + 'product measure', async () => {
  const src = HEAD + `w_ev = [${W_EV.join(', ')}]\n` + MIX_LAMBDA + `
E ~ mix.(w_ev)
rstate = rnginit([1, 2, 3, 4])
Es, _ = rand(rstate, iid(lawof(E), 4000))
`;
  const { ctx } = ctxFor(src, 8);
  const m = await ctx.getMeasure('Es');
  const data: Float64Array = m.value.data;
  assert.equal(data.length % W_EV.length, 0, 'a whole number of cells per draw');
  const K = W_EV.length;
  const rows = data.length / K;
  for (let j = 0; j < K; j++) {
    const cf = mixMoments(W_EV[j]);
    let s = 0;
    for (let i = 0; i < rows; i++) s += data[i * K + j];
    const mean = s / rows;
    const seMean = cf.sd / Math.sqrt(rows);
    assert.ok(Math.abs(mean - cf.mean) < 5 * seMean,
      `rand cell ${j} mean ${mean} vs ${cf.mean} (5 se = ${5 * seMean})`);
  }
});

test('sampling: a latent-dependent mixing weight stays correlated with its '
  + 'latent', async () => {
  // The mixing proportion must condition on the SAME atom's latent draw:
  // Cov(p, X) = (lam - mu)·Var(p) exactly, since E[X | p] = p·lam +
  // (1-p)·mu is affine in p. A weight that decoupled from the latent
  // (a pooled E[p]) would measure ≈ 0.
  // A scalar (rank-0) broadcast argument is held constant across the cells
  // per §04 "Non-collection inputs", so there is one cell and `E` is
  // [N, 1].
  const N = 200000;
  const src = HEAD + `
p ~ Uniform(interval(0.2, 0.8))
` + MIX_LAMBDA + 'E ~ mix.(p)\n';
  const { ctx } = ctxFor(src, N);
  const pm = await ctx.getMeasure('p');
  const em = await ctx.getMeasure('E');
  assert.deepEqual(em.value.shape, [N, 1], 'a held-constant argument gives one cell');
  const ps: Float64Array = pm.samples;
  const es: Float64Array = em.value.data;
  let mp = 0; let me = 0;
  for (let i = 0; i < N; i++) { mp += ps[i]; me += es[i]; }
  mp /= N; me /= N;
  let cov = 0;
  for (let i = 0; i < N; i++) cov += (ps[i] - mp) * (es[i] - me);
  cov /= N - 1;
  const varP = (0.8 - 0.2) ** 2 / 12;
  const expected = (LAM - S_MU) * varP;
  // Var(X) bounds the covariance's own error: se(cov) <= sd(p)·sd(X)/sqrt(N).
  const sdX = mixMoments(0.5).sd;
  const se = Math.sqrt(varP) * sdX / Math.sqrt(N);
  assert.ok(Math.abs(cov - expected) < 6 * se,
    `Cov(p, E) = ${cov}, closed form ${expected} (6 se = ${6 * se})`);
});

// ---------------------------------------------------------------------
// Refusals — loud, never a silently wrong number
// ---------------------------------------------------------------------

test('refusal: a measure-algebra kernel body over TWO broadcast axes is '
  + 'refused, not flattened', async () => {
  const src = HEAD + `
w_mat = [[0.8, 0.5], [0.2, 0.9]]
` + MIX_LAMBDA + 'E ~ mix.(w_mat)\n';
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(ctx.getMeasure('E'),
    /broadcast axes is not supported yet|unsupported value shape/,
    'a 2-D weight grid refuses rather than losing the grid shape');
});

test('refusal: too few broadcast arguments for the kernel\'s formals',
  async () => {
    const src = HEAD + `
a_v = [0.3, 0.4]
K = (a, b) -> normalize(superpose(weighted(a, bkg), weighted(b, sig)))
E ~ K.(a_v)
`;
    const { ctx } = ctxFor(src, 4);
    await assert.rejects(ctx.getMeasure('E'),
      /expected 2 argument\(s\) for kernel parameter\(s\) a, b, got 1/,
      'an arity mismatch is named, not guessed');
  });

test('refusal: a keyword argument that names no kernel formal', async () => {
  const src = HEAD + 'w_ev = [0.8, 0.5]\n' + MIX_LAMBDA + 'E ~ mix.(q = w_ev)\n';
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(ctx.getMeasure('E'),
    /kernel parameter 'p' has no broadcast argument/,
    'an unbound formal is named');
});

// ---------------------------------------------------------------------
// Two-formal kernels — the unnormalised-yield mixture spelling
// ---------------------------------------------------------------------
//
// `normalize(superpose(weighted(nu_B, bkg), weighted(nu_S, sig)))` is the
// SAME measure as the `w` / `1 - w` spelling, since
// (a·p + b·q)/(a + b) = w·p + (1 - w)·q with w = a/(a + b). It never forms
// `1 - w`, so it needs two broadcast arguments — one collection per
// formal — rather than one.

const YIELD_KERNEL =
  'mix2 = (a, b) -> normalize(superpose(weighted(a, bkg), weighted(b, sig)))\n';

test('the unnormalised-yield spelling scores the same value as the '
  + 'normalized-weight spelling', async () => {
  // nu_B / (nu_B + nu_S) reproduces W_EV exactly at these yields.
  const nuB = W_EV.map((w) => w);
  const nuS = W_EV.map((w) => 1 - w);
  const got = await scoreOf(HEAD
    + `nu_B = [${nuB.join(', ')}]\nnu_S = [${nuS.join(', ')}]\n`
    + YIELD_KERNEL
    + `E ~ mix2.(nu_B, nu_S)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`);
  assert.ok(relErr(got, ORACLE_A) < 1e-12,
    `yield-form score ${got} vs scipy ${ORACLE_A} (rel ${relErr(got, ORACLE_A)})`);
});

test('the yield spelling is scale-invariant, as `normalize` requires',
  async () => {
  // Scaling both yields by the same factor leaves the normalized mixture
  // unchanged, so the density must not move.
  const nuB = W_EV.map((w) => 7.5 * w);
  const nuS = W_EV.map((w) => 7.5 * (1 - w));
  const got = await scoreOf(HEAD
    + `nu_B = [${nuB.join(', ')}]\nnu_S = [${nuS.join(', ')}]\n`
    + YIELD_KERNEL
    + `E ~ mix2.(nu_B, nu_S)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`);
  assert.ok(relErr(got, ORACLE_A) < 1e-12,
    `scaled-yield score ${got} vs scipy ${ORACLE_A}`);
});

test('sampling: the yield spelling reproduces the mixture\'s closed-form '
  + 'mean per cell', async () => {
  const N = 200000;
  const nuB = W_EV.map((w) => 4.0 * w);
  const nuS = W_EV.map((w) => 4.0 * (1 - w));
  const src = HEAD
    + `nu_B = [${nuB.join(', ')}]\nnu_S = [${nuS.join(', ')}]\n`
    + YIELD_KERNEL + 'E ~ mix2.(nu_B, nu_S)\n';
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('E');
  assert.equal(m.logTotalmass, 0, 'each cell normalizes to mass 1');
  const K = W_EV.length;
  for (let j = 0; j < K; j++) {
    const cf = mixMoments(W_EV[j]);
    let s = 0;
    for (let i = 0; i < N; i++) s += m.value.data[i * K + j];
    const mean = s / N;
    const seMean = cf.sd / Math.sqrt(N);
    assert.ok(Math.abs(mean - cf.mean) < 5 * seMean,
      `yield cell ${j} mean ${mean} vs ${cf.mean} (5 se = ${5 * seMean})`);
  }
});

test('a size-one argument axis expands against a longer one (§04 singleton '
  + 'expansion)', async () => {
  const src = HEAD + 'nu_B = [1.0]\nnu_S = [1.0, 2.0, 3.0]\n' + YIELD_KERNEL
    + 'E ~ mix2.(nu_B, nu_S)\n';
  const { ctx } = ctxFor(src, 16);
  const m = await ctx.getMeasure('E');
  assert.deepEqual(m.value.shape, [16, 3],
    'the size-one axis repeats to the longer argument\'s length');
});

test('refusal: arguments whose axis sizes conflict (§04 requires equal or 1)',
  async () => {
    const src = HEAD + 'nu_B = [1.0, 2.0]\nnu_S = [1.0, 2.0, 3.0]\n' + YIELD_KERNEL
      + 'E ~ mix2.(nu_B, nu_S)\n';
    const { ctx } = ctxFor(src, 4);
    await assert.rejects(ctx.getMeasure('E'),
      /incompatible collection sizes on axis 0 \(2 vs 3\)/,
      'the conflicting sizes are named');
  });

test('the per-cell Z divisor does not disturb the non-broadcast spellings',
  async () => {
    // A direct `normalize(superpose(…))` and an `iid` of one resolve Z on
    // the main thread. Both must still land on the single-event value, so
    // the worker-side divisor cannot be applied twice.
    const oracle = -4.7120230054279;   // log(0.8·f_bkg(32) + 0.2·f_sig(32))
    const direct = await scoreOf(HEAD + '__score__ = logdensityof(normalize('
      + 'superpose(weighted(4.0, bkg), weighted(1.0, sig))), 32.0)\n');
    const iid = await scoreOf(HEAD + 'E ~ iid(normalize(superpose('
      + 'weighted(4.0, bkg), weighted(1.0, sig))), 1)\n'
      + '__score__ = logdensityof(E, [32.0])\n');
    for (const [what, got] of [['direct', direct], ['iid', iid]] as Array<[string, number]>) {
      assert.ok(relErr(got, oracle) < 1e-12, `${what} ${got} vs scipy ${oracle}`);
    }
  });

test('refusal: a superposition of zero total mass (spec §06 leaves '
  + 'normalize undefined at Z = 0)', async () => {
  const src = HEAD + `
z_v = [0.0]
K = a -> normalize(superpose(weighted(a, bkg), weighted(a, sig)))
E ~ K.(z_v)
__score__ = logdensityof(E, [32.0])
`;
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(ctx.getMeasure('__score__'),
    /total mass is 0|undefined at Z = 0/,
    'a zero-mass superposition refuses rather than scoring -inf silently');
});

test('refusal: a RECORD broadcast argument (§04 disallows records as '
  + 'broadcast inputs)', async () => {
  const src = HEAD + MIX_LAMBDA + 'r = record(a = 0.5)\nE ~ mix.(r)\n';
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(ctx.getMeasure('E'),
    /argument 'p' resolved to an unsupported value shape/,
    'a record argument is named as an unsupported shape');
});

test('refusal: a record-variate measure-algebra body', async () => {
  // The executor stitches one SCALAR column per cell. A record variate has
  // no such column, so it refuses instead of producing a wrong shape.
  const src = HEAD + `
w_v = [1.0, 2.0]
K = a -> weighted(a, record(x = bkg, y = sig))
E ~ K.(w_v)
`;
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(ctx.getMeasure('E'),
    /produced no scalar samples/,
    'a record-variate body refuses rather than mis-stitching');
});

// ---------------------------------------------------------------------
// Argument and component spellings
// ---------------------------------------------------------------------

test('the keyword argument spelling `mix.(p = col)` binds the same formal '
  + 'as the positional one', async () => {
  const base = HEAD + `w_ev = [${W_EV.join(', ')}]\n` + MIX_LAMBDA;
  const kw = await scoreOf(base
    + `E ~ mix.(p = w_ev)\n__score__ = logdensityof(E, [${DATA.join(', ')}])\n`);
  assert.ok(relErr(kw, ORACLE_A) < 1e-12,
    `kwarg-bound score ${kw} vs scipy ${ORACLE_A}`);
});

test('a NAMED superpose binding as the body\'s component resolves through '
  + 'its `select` derivation', async () => {
  // A named `superpose` lowers to a `select` derivation, so the executor's
  // measure-subtree closure has to follow `branches[].ref` as well as
  // `from` / `fromNames` — otherwise the component would be shared across
  // cells instead of redrawn.
  const NAMED = HEAD + `
mixture = superpose(weighted(0.5, bkg), weighted(0.5, sig))
a_v = [1.0, 1.0]
K = a -> normalize(weighted(a, mixture))
E ~ K.(a_v)
`;
  const oracle = -7.6003776470423814;   // Σ log(0.5·f_bkg + 0.5·f_sig)
  const got = await scoreOf(NAMED + '__score__ = logdensityof(E, [32.0, 99.0])\n');
  assert.ok(relErr(got, oracle) < 1e-12,
    `named-superpose component score ${got} vs scipy ${oracle}`);
  // Sampling exercises the closure walk that has to follow `branches[].ref`.
  const { ctx } = ctxFor(NAMED, 400);
  const m = await ctx.getMeasure('E');
  assert.deepEqual(m.value.shape, [400, 2]);
});

test('a NON-CONSTANT `weighted` weight per cell scores through the '
  + 'by-name weight evaluator', async () => {
  // The weight is latent-dependent, so the inline-IR bridge cannot fold it
  // to a constant log-shift; it must reach matWeighted as a `weightIR`.
  const oracle = -3.7810242469692907;   // Σ [log pdf_Normal(x_j;0,1) + log w_j]
  const got = await scoreOf(`
s ~ Uniform(interval(0.1, 0.9))
w_v = s .* [1.0, 2.0]
K = a -> weighted(a, Normal(0.0, 1.0))
X ~ K.(w_v)
KK = kernelof(record(X = X), s = s)
L = likelihoodof(KK, record(X = [0.5, 1.5]))
__score__ = logdensityof(L, record(s = 0.5))
`);
  assert.ok(relErr(got, oracle) < 1e-12,
    `non-constant weighted score ${got} vs scipy ${oracle}`);
});

test('a NON-CONSTANT `logweighted` weight per cell scores through the '
  + 'by-name weight evaluator', async () => {
  const oracle = -4.5878770664093453;   // Σ [log pdf_Normal(x_j;0,1) + lw_j]
  const got = await scoreOf(`
s ~ Uniform(interval(0.1, 0.9))
lw_v = s .* [-1.0, -2.0]
K = a -> logweighted(a, Normal(0.0, 1.0))
X ~ K.(lw_v)
KK = kernelof(record(X = X), s = s)
L = likelihoodof(KK, record(X = [0.5, 1.5]))
__score__ = logdensityof(L, record(s = 0.5))
`);
  assert.ok(relErr(got, oracle) < 1e-12,
    `non-constant logweighted score ${got} vs scipy ${oracle}`);
});

test('sampling: a non-constant `weighted` body carries the cells\' masses '
  + 'into the product measure', async () => {
  // Two cells at weights s and 2s: the product measure's mass is 2·s².
  // A per-cell weight stream that was folded as a raw log-weight instead
  // of an excess over its own mass would double-count the scale.
  const src = `
s = 0.5
w_v = s .* [1.0, 2.0]
K = a -> weighted(a, Normal(0.0, 1.0))
X ~ K.(w_v)
`;
  const { ctx } = ctxFor(src, 4000);
  const m = await ctx.getMeasure('X');
  const want = Math.log(2 * 0.25);
  assert.ok(Math.abs(m.logTotalmass - want) < 1e-9,
    `logTotalmass ${m.logTotalmass} vs log(2 s^2) = ${want}`);
});

test('sampling: a non-constant `logweighted` body carries the cells\' log '
  + 'masses', async () => {
  // lw = [-s, -2s] at s = 0.5, so the product mass is exp(-0.5)·exp(-1.0).
  const src = `
s = 0.5
lw_v = s .* [-1.0, -2.0]
K = a -> logweighted(a, Normal(0.0, 1.0))
X ~ K.(lw_v)
`;
  const { ctx } = ctxFor(src, 4000);
  const m = await ctx.getMeasure('X');
  assert.ok(Math.abs(m.logTotalmass - (-1.5)) < 1e-9,
    `logTotalmass ${m.logTotalmass} vs -1.5`);
});

test('guard: a generative value-expression kernel body is still NOT claimed '
  + 'by the measure-algebra recogniser', async () => {
  // §06 case 3 — the pushforward marginalises an internal draw, so the
  // density has no closed form. The measure-algebra recogniser must not
  // reclassify it: its body op is `lawof`, not an algebra op.
  const shape = require('../kernel-broadcast-shape.ts');
  const { processSource, orchestrator } = require('..');
  const proc = processSource(`
x = elementof(reals)
delta = draw(Uniform(interval(0.0, 1.0)))
y = (x + delta)^3
transport = kernelof(y, x = x)
xs = [1.0, 2.0]
ys = transport.(xs)
`);
  const built = orchestrator.buildDerivations(proc.bindings);
  assert.equal(
    shape.isMeasureAlgebraCompositeKernelBinding('transport', built.bindings),
    false, 'a generative body is not a measure-algebra body');
  assert.equal(
    shape.isGenerativeCompositeKernelBinding('transport', built.bindings),
    true, 'it is still the generative recogniser\'s shape');
});

test('guard: an iid-bodied kernel stays with the iid recogniser', async () => {
  const shape = require('../kernel-broadcast-shape.ts');
  const { processSource, orchestrator } = require('..');
  const proc = processSource(`
a = [1.0, 2.0]
b = [3.0, 4.0]
K = (aa, bb) -> iid(Beta(aa, bb), 3)
P ~ K.(a, b)
`);
  const built = orchestrator.buildDerivations(proc.bindings);
  assert.equal(shape.isMeasureAlgebraCompositeKernelBinding('K', built.bindings),
    false, '`iid` is a measure CONSTRUCTION, not an algebra wrapper');
  assert.equal(shape.isIidCompositeKernelBinding('K', built.bindings, undefined),
    true, 'the iid recogniser keeps it');
});

test('the per-cell Z divisor recognises the KEYWORD leaf spelling too '
  + '(§04 calling convention)', async () => {
  // §08 fixes the parameter order, so `Normal(mu = …, sigma = …)` and
  // `Normal(…, …)` are one call. Both must reach the same Z.
  const positional = await scoreOf(HEAD + `
a_v = [4.0]
K = a -> normalize(superpose(weighted(a, Exponential(1.0 / lam)),
                             weighted(1.0, Normal(${S_MU}, ${S_SIGMA}))))
E ~ K.(a_v)
__score__ = logdensityof(E, [32.0])
`);
  const keyword = await scoreOf(HEAD + `
a_v = [4.0]
K = a -> normalize(superpose(weighted(a, Exponential(rate = 1.0 / lam)),
                             weighted(1.0, Normal(mu = ${S_MU}, sigma = ${S_SIGMA}))))
E ~ K.(a_v)
__score__ = logdensityof(E, [32.0])
`);
  assert.equal(positional, keyword,
    'the keyword spelling gets the same Z as the positional one');
  const oracle = -4.7120230054279;
  assert.ok(relErr(keyword, oracle) < 1e-12, `${keyword} vs scipy ${oracle}`);
});

test('a component reached through a `select` derivation redraws per cell',
  async () => {
    // `ifelse` over two measures classifies as a `select` derivation, whose
    // measure parents live in `branches[].ref` rather than `from` /
    // `fromNames`. The executor's measure-subtree closure has to follow
    // that field too, or the component would be shared across the cells.
    const src = HEAD + `
flag = true
comp = ifelse(flag, bkg, sig)
a_v = [1.0, 1.0]
K = a -> normalize(weighted(a, comp))
E ~ K.(a_v)
`;
    const { ctx } = ctxFor(src, 4000);
    const m = await ctx.getMeasure('E');
    assert.deepEqual(m.value.shape, [4000, 2]);
    // Both cells are Exponential(1/40) draws; independent cells decorrelate.
    let ma = 0; let mb = 0;
    for (let i = 0; i < 4000; i++) { ma += m.value.data[i * 2]; mb += m.value.data[i * 2 + 1]; }
    ma /= 4000; mb /= 4000;
    let cab = 0; let va = 0; let vb = 0;
    for (let i = 0; i < 4000; i++) {
      const da = m.value.data[i * 2] - ma; const db = m.value.data[i * 2 + 1] - mb;
      cab += da * db; va += da * da; vb += db * db;
    }
    const r = cab / Math.sqrt(va * vb);
    assert.ok(Math.abs(r) < 5 / Math.sqrt(4000),
      `cells correlated at r = ${r} — the component was shared, not redrawn`);
  });

test('a component weight the worker env cannot evaluate leaves the Z '
  + 'resolver declining, not guessing', async () => {
  // `totalmass(M)` is not evaluable in the sampler env, so the resolver
  // returns null (fall through) rather than inventing a Z. The walk then
  // reports the real gap from the weight evaluation itself.
  const src = HEAD + `
a_v = [1.0]
K = a -> normalize(superpose(weighted(a, bkg), weighted(totalmass(sig), sig)))
E ~ K.(a_v)
__score__ = logdensityof(E, [32.0])
`;
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(ctx.getMeasure('__score__'),
    /'totalmass' not evaluable in sampler context/,
    'the unevaluable weight is reported, not silently normalized');
});
