'use strict';

// =====================================================================
// broadcast-measure-kernel-density.test.ts — Buffy #268
// =====================================================================
//
// Per spec §04, a `functionof`/lambda of a measure node IS a (Markov)
// kernel, and `broadcast(kernel, …)` is the array-valued independent
// product measure of the kernel applications — density = Σ of per-
// application log-densities. `walkBroadcast` (density.ts ~1645) only
// scored a BUILTIN-distribution head; a user-defined kernel whose body
// is a composite MEASURE (`normalize(superpose(weighted(...)))` of
// sampleable leaves) was rejected outright, even though its per-atom
// density IS tractable (the engine already scores that composite
// measure when written directly, non-broadcast — see
// mixture-normalize-scorer.test.ts).
//
// Motivating model: a per-event signal/background mixture where the
// per-event signal fraction `w_ev[j]` varies — the natural vectorized
// form is one `broadcast(energy, w_ev)` node rather than a hand-unrolled
// `iid`-per-distinct-weight-group.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const LAM = 50.0;
const S_MU = 5.0;
const S_SIGMA = 0.5;
const W_EV = [0.04, 0.03, 0.02, 0.03, 0.015];
const E_DATA = [4.9, 5.1, 12.0, 5.3, 30.0];

const MODEL = `
lam = ${LAM}
S_mu = ${S_MU}
S_sigma = ${S_SIGMA}
bkg = Exponential(1.0 / lam)
sig = Normal(S_mu, S_sigma)
w_ev = [${W_EV.join(', ')}]
energy = p -> normalize(superpose(weighted(p, bkg), weighted(1.0 - p, sig)))
E ~ energy.(w_ev)
`;

async function scoreE(): Promise<number> {
  const src = MODEL + `\n__score__ = logdensityof(E, [${E_DATA.join(', ')}])\n`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) {
    throw new Error('scoreE: __score__ produced no data');
  }
  return s[0];
}

test('#268 repro: broadcast of a composite-measure-valued kernel head scores '
  + '(vectorized == hand-unrolled per-event mixture)', async () => {
  // Hand-unrolled equivalent: one normalize(superpose(...)) per event, each
  // scored at its own datum, summed. Independently exercises the ALREADY-
  // SUPPORTED composite-measure scorer (mixture-normalize-scorer.test.ts) —
  // this is the oracle for the vectorized broadcast form, not the other way
  // around.
  let unrolledSrc = MODEL;
  let expectExpr = '';
  const parts: string[] = [];
  for (let j = 0; j < W_EV.length; j++) {
    const name = `mix_${j}`;
    unrolledSrc += `${name} = normalize(superpose(weighted(${W_EV[j]}, bkg), `
      + `weighted(${1.0 - W_EV[j]}, sig)))\n`;
    parts.push(`logdensityof(${name}, ${E_DATA[j]})`);
  }
  expectExpr = parts.join(' + ');
  unrolledSrc += `__unrolled__ = ${expectExpr}\n`;
  const { ctx } = ctxFor(unrolledSrc, 1);
  const um = await ctx.getMeasure('__unrolled__');
  const unrolled: number = (um.samples ?? (um.value && um.value.data))[0];
  assert.ok(Number.isFinite(unrolled), 'hand-unrolled score is finite');

  const vectorized = await scoreE();
  assert.ok(Math.abs(vectorized - unrolled) < 1e-9,
    `vectorized broadcast score ${vectorized} == hand-unrolled ${unrolled} `
    + `(Δ ${Math.abs(vectorized - unrolled)})`);
});

test('guard: a generative-value-body kernel broadcast still refuses loudly '
  + '(§06 case 3, no silent Monte-Carlo fallback)', async () => {
  // `y = (x + delta)^3` closes over the internal `delta` draw — a
  // generative VALUE expression, not a measure. Its pushforward
  // marginalises `delta`, which has no closed-form density (spec §06 case
  // 3). The fix must NOT reclassify this as scorable — it isn't a measure-
  // bodied kernel, so it must keep throwing exactly as before this change.
  const src = `
x = elementof(reals)
delta = draw(Uniform(interval(0.0, 1.0)))
y = (x + delta)^3
transport = kernelof(y, x = x)
xs = [1.0, 2.0]
ys = transport.(xs)
__score__ = logdensityof(ys, [1.0, 2.0])
`;
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(
    ctx.getMeasure('__score__'),
    /not a built-in distribution kernel|user-defined kernel|generative value-expression|§06|bijection\(/,
    'generative value-expression kernel broadcast still refuses loudly');
});

test('guard: builtin-distribution-head broadcast (Normal.(mu_vec, sigma)) '
  + 'density is unchanged', async () => {
  const src = `
mu = [0.0, 10.0, 100.0]
sigma = 1.0
x ~ Normal.(mu, sigma)
__score__ = logdensityof(x, [0.5, 9.5, 101.0])
`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const got: number = m.samples[0];
  // Independent closed-form check: Σ logpdf(Normal(mu_j, 1), y_j).
  const logpdfNormal = (x: number, muv: number, sd: number) =>
    -Math.log(sd) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((x - muv) / sd) ** 2;
  const expected = logpdfNormal(0.5, 0.0, 1.0) + logpdfNormal(9.5, 10.0, 1.0)
    + logpdfNormal(101.0, 100.0, 1.0);
  assert.ok(Math.abs(got - expected) < 1e-9,
    `builtin-head broadcast density ${got} == closed-form ${expected} `
    + `(Δ ${Math.abs(got - expected)})`);
});
