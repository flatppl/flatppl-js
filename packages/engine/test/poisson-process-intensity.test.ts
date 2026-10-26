'use strict';

// §08 PoissonProcess(intensity): with Λ = totalmass(intensity) and
// λ(t) = densityof(intensity, t), the density is e^{-Λ} ∏ λ(t_i). §06 makes
// `truncate(M, S)` a restriction that does NOT renormalise, and writes the
// sideband model as `PoissonProcess(intensity = truncate(intensity, R))`
// (§02, §06, §10). These intensities did not score or draw:
//   - a truncated intensity, `truncate(weighted(w, s), S)` or `truncate` of a
//     superpose, refused with "a shape must be a scalar distribution";
//   - the §06 homogeneous form `truncate(weighted(rate, Lebesgue(support =
//     reals)), R)` refused the same way;
//   - the positional spelling `PoissonProcess(weighted(...))` lost its shape;
//   - a Uniform shape failed to draw ("cannot coerce object to Value").
//
// ORACLE: the extended likelihood written out in scipy 1.x through the python
// MCP — Λ = Σ_c w_c·P_c(S) (w_c for a normalized component) and
// λ(t) = Σ_c w_c·pdf_c(t) on S — with no engine value in it.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function score(src: string, name = 'L'): Promise<number> {
  const { ctx } = makeMatCtx(src);
  const m = await ctx.getMeasure(name);
  return m.value.data[0];
}

const close = (got: number, want: number, what: string) =>
  assert.ok(Math.abs(got - want) <= 1e-12 * Math.max(1, Math.abs(want)), `${what}: ${got} vs ${want}`);

const TRUNC = 'truncate(weighted(2.0, Normal(0.0, 1.0)), interval(0.0, 1.0))';

test('a truncated intensity scores e^{-Λ} ∏ λ(t_i) with Λ = w·P(S)', async () => {
  close(await score(`M = PoissonProcess(intensity = ${TRUNC})\nL = logdensityof(M, [0.2, 0.7])`),
    -1.3992721974265407, 'two points');
  close(await score(`M = PoissonProcess(intensity = ${TRUNC})\nL = logdensityof(M, [0.2])`),
    -0.6826894921370859 + Math.log(2) - 0.5 * Math.log(2 * Math.PI) - 0.02, 'one point');
  assert.equal(await score(`M = PoissonProcess(intensity = ${TRUNC})\nL = logdensityof(M, [0.2, 1.5])`),
    -Infinity, 'a point outside S');
  // The commuted spelling and a named intensity are the same component.
  close(await score('M = PoissonProcess(intensity = weighted(2.0, truncate(Normal(0.0, 1.0), '
    + 'interval(0.0, 1.0))))\nL = logdensityof(M, [0.2, 0.7])'), -1.3992721974265407, 'commuted');
  close(await score('intensity = weighted(2.0, Normal(0.0, 1.0))\nR = interval(0.0, 1.0)\n'
    + 'M = PoissonProcess(intensity = truncate(intensity, R))\nL = logdensityof(M, [0.2, 0.7])'),
  -1.3992721974265407, 'named');
});

test('a normalized truncated shape keeps the expected count', async () => {
  close(await score('M = PoissonProcess(weighted(3.0, normalize(truncate(Normal(0.0, 1.0), '
    + 'interval(0.0, 1.0)))))\nL = logdensityof(M, [0.2, 0.7])'), -0.7559278353489836, 'normalized');
});

test('truncate distributes over a superposed intensity (the §10 sideband form)', async () => {
  const src = 'rate = superpose(weighted(1.5, Normal(2.0, 0.5)), weighted(5.0, Exponential(0.5)))\n'
    + 'sideband = interval(0.0, 3.0)\n'
    + 'M = PoissonProcess(intensity = truncate(rate, sideband))\nL = logdensityof(M, [0.5, 2.4, 2.9])';
  close(await score(src), -4.3878467222831885, 'sideband');
});

// §06 "Posterior construction": the homogeneous intensity on R = [-3, 3] has
// Λ = 6·lambda_bar and λ = lambda_bar on R; 3 of the 5 events lie in R.
test('the §06 Lebesgue intensity scores 3·log(lambda_bar) − 6·lambda_bar', async () => {
  const model = 'intensity = weighted(lambda_bar, Lebesgue(support = reals))\n'
    + 'obs_events = [1.2, 3.4, 5.1, -1.5, 2.8]\nR = interval(-3.0, 3.0)\n'
    + 'obs_R = filter(fn(_ in R), obs_events)\n'
    + 'model_R = PoissonProcess(intensity = truncate(intensity, R))\n';
  close(await score(`lambda_bar = 2.0\n${model}L = logdensityof(model_R, obs_R)`),
    3 * Math.log(2) - 12, 'fixed rate');
  close(await score(`lambda_bar = elementof(posreals)\n${model}`
    + 'L_R = likelihoodof(functionof(model_R), obs_R)\n'
    + 'L = logdensityof(L_R, record(lambda_bar = 2.0))'), 3 * Math.log(2) - 12, 'likelihood');
});

test('an untruncated Lebesgue intensity is refused as infinite mass', async () => {
  const { ctx } = makeMatCtx('M = PoissonProcess(weighted(2.0, Lebesgue(support = reals)))\n'
    + 'L = logdensityof(M, [0.5])');
  await assert.rejects(ctx.getMeasure('L'), /must be truncated to a bounded set to have finite mass/);
});

test('positional and keyword intensities score identically', async () => {
  for (const I of ['weighted(2.0, Uniform(interval(0.0, 1.0)))', TRUNC,
    'superpose(weighted(1.5, Normal(2.0, 0.5)), weighted(5.0, Exponential(0.5)))']) {
    const pos = await score(`M = PoissonProcess(${I})\nL = logdensityof(M, [0.2, 0.7])`);
    const kw = await score(`M = PoissonProcess(intensity = ${I})\nL = logdensityof(M, [0.2, 0.7])`);
    assert.equal(pos, kw, I);
  }
  close(await score('M = PoissonProcess(weighted(2.0, Uniform(interval(0.0, 1.0))))\n'
    + 'L = logdensityof(M, [0.2, 0.7])'), -0.6137056388801094, 'uniform');
});

// Draw moments over 20000 seeded atoms, each within 4 standard errors: the count
// mean is Λ, and the points follow the restricted shape.
async function draws(src: string): Promise<{ countMean: number, pts: Float64Array }> {
  const { ctx } = makeMatCtx(src, { sampleCount: 20000 });
  const m = await ctx.getMeasure('s');
  const off = m.ragged.offsets;
  return { countMean: off[off.length - 1] / (off.length - 1), pts: m.ragged.data };
}

function checkDraws(d: { countMean: number, pts: Float64Array }, lambda: number,
  lo: number, hi: number, mean: number, sd: number, what: string) {
  assert.ok(Math.abs(d.countMean - lambda) < 4 * Math.sqrt(lambda / 20000), `${what} count ${d.countMean}`);
  let s = 0;
  for (const x of d.pts) {
    assert.ok(x >= lo && x <= hi, `${what} point ${x} outside [${lo}, ${hi}]`);
    s += x;
  }
  const ptMean = s / d.pts.length;
  assert.ok(Math.abs(ptMean - mean) < 4 * sd / Math.sqrt(d.pts.length), `${what} point mean ${ptMean}`);
}

test('draws: counts have mean Λ and points follow the restricted shape', async () => {
  // Truncated N(0, 1) on [0, 1]: mean 0.4598622292864266, sd 0.2822265488016873.
  checkDraws(await draws(`M = PoissonProcess(intensity = ${TRUNC})\ns ~ M`),
    0.6826894921370859, 0, 1, 0.4598622292864266, 0.2822265488016873, 'truncated');
  checkDraws(await draws('M = PoissonProcess(weighted(3.0, normalize(truncate(Normal(0.0, 1.0), '
    + 'interval(0.0, 1.0)))))\ns ~ M'), 3, 0, 1, 0.4598622292864266, 0.2822265488016873, 'normalized');
  checkDraws(await draws('M = PoissonProcess(weighted(2.0, Uniform(interval(0.0, 1.0))))\ns ~ M'),
    2, 0, 1, 0.5, Math.sqrt(1 / 12), 'uniform');
  checkDraws(await draws('M = PoissonProcess(truncate(weighted(2.0, Lebesgue(support = reals)), '
    + 'interval(-1.0, 2.0)))\ns ~ M'), 6, -1, 2, 0.5, Math.sqrt(9 / 12), 'lebesgue');
  checkDraws(await draws('rate = superpose(weighted(1.5, Normal(2.0, 0.5)), weighted(5.0, Exponential(0.5)))\n'
    + 'M = PoissonProcess(intensity = truncate(rate, interval(0.0, 3.0)))\ns ~ M'),
  5.350176494472833, 0, 3, 1.3668722409547762, 0.8290923775913214, 'sideband');
});

// Each atom scores with its own drawn weight: 2·log(lam) + log φ(0.2) + log φ(0.7)
// − lam·(Φ(1) − Φ(0)), where Φ(1) − Φ(0) = 0.3413447460685429.
test('a draw-dependent weight scores per atom', async () => {
  const { ctx } = makeMatCtx('lam ~ Uniform(interval(1.0, 3.0))\n'
    + `M = PoissonProcess(truncate(weighted(lam, Normal(0.0, 1.0)), interval(0.0, 1.0)))\n`
    + 'L = logdensityof(M, [0.2, 0.7])', { sampleCount: 16 });
  const lam = await ctx.getMeasure('lam');
  const L = await ctx.getMeasure('L');
  const lams = lam.samples, Ls = L.samples;
  const logPhi = (t: number) => -0.5 * Math.log(2 * Math.PI) - t * t / 2;
  for (let i = 0; i < 16; i++) {
    close(Ls[i], 2 * Math.log(lams[i]) + logPhi(0.2) + logPhi(0.7) - lams[i] * 0.3413447460685429,
      `atom ${i}`);
  }
});

// Shapes and sets this engine cannot score are refused, each with its reason.
test('an intensity outside the supported forms is refused with its reason', async () => {
  const density = async (I: string, re: RegExp) => {
    const { ctx } = makeMatCtx(`M = PoissonProcess(intensity = ${I})\nL = logdensityof(M, [0.5])`);
    await assert.rejects(ctx.getMeasure('L'), re, I);
  };
  const draw = async (I: string, re: RegExp) => {
    const { ctx } = makeMatCtx(`k = 1.0 - 2.0\nM = PoissonProcess(intensity = ${I})\ns ~ M`);
    await assert.rejects(ctx.getMeasure('s'), re, I);
  };
  await density('truncate(weighted(2.0, Normal(0.0, 1.0)), integers)', /needs an interval set/);
  await density('weighted(2.0, Lebesgue(support = integers))', /needs an interval support/);
  await density('truncate(weighted(2.0, Poisson(3.0)), interval(0.0, 5.0))',
    /needs a continuous shape with a CDF, got 'Poisson'/);
  // A literal negative weight is refused statically; a computed one here.
  await draw('weighted(k, Normal(0.0, 1.0))', /must be ≥ 0/);
  await draw('weighted(2.0, Lebesgue(support = reals))', /must be truncated to a bounded set/);
  await draw('truncate(weighted(2.0, Normal(0.0, 1.0)), integers)', /needs an interval set/);
  await draw('weighted(2.0, superpose(Normal(0.0, 1.0), Normal(1.0, 1.0)))',
    /a shape must be a scalar distribution, got 'superpose'/);
});

test('a Lebesgue intensity on a bounded or half-line support takes its bounds', async () => {
  close(await score('M = PoissonProcess(weighted(2.0, Lebesgue(support = interval(0.0, 2.0))))\n'
    + 'L = logdensityof(M, [0.5])'), Math.log(2) - 4, 'interval');
  close(await score('M = PoissonProcess(truncate(weighted(2.0, Lebesgue(support = posreals)), '
    + 'interval(-1.0, 3.0)))\nL = logdensityof(M, [0.5, 2.5])'), 2 * Math.log(2) - 6, 'posreals');
  assert.equal(await score('M = PoissonProcess(truncate(weighted(2.0, Lebesgue(support = posreals)), '
    + 'interval(-1.0, 3.0)))\nL = logdensityof(M, [-0.5])'), -Infinity, 'outside the support');
});

// flatppl-testsuite corpora/coverage/beam_bunch: a latent-weighted superposition
// of a normalized truncated Gaussian and a windowed flat background. The frozen
// oracle is that row's scipy closed form (extended unbinned likelihood times the
// Gamma priors); the row's det-js and stablehlo engines score the same values.
test('the beam_bunch posterior scores the testsuite scipy oracle', async () => {
  const model = 'window = interval(0.0, 10.0)\n'
    + 'events_obs = [1.4558, 2.7229, 3.44, 3.5532, 3.815, 4.1881, 4.5518, 4.6498, 4.7677, 4.8254, '
    + '4.8584, 4.9804, 5.0494, 5.2282, 5.4442, 5.5463, 5.8047, 5.8668, 5.9805, 6.1686, 6.2184, '
    + '6.6791, 7.2659, 7.4487, 9.375]\n'
    + 's ~ Gamma(shape = 5.0, rate = 0.5)\nb ~ Gamma(shape = 8.0, rate = 0.5)\n'
    + 'sig_shape = normalize(truncate(Normal(mu = 5.0, sigma = 0.8), window))\n'
    + 'bkg_shape = weighted(0.1, Lebesgue(support = window))\n'
    + 'intensity = superpose(weighted(s, sig_shape), weighted(b, bkg_shape))\n'
    + 'events ~ PoissonProcess(intensity = intensity)\n'
    + 'prior = lawof(record(s = s, b = b))\n'
    + 'forward_kernel = kernelof(record(events = events), s = s, b = b)\n'
    + 'L = likelihoodof(forward_kernel, record(events = events_obs))\n'
    + 'posterior = bayesupdate(L, prior)\n';
  for (const [sv, bv, want] of [[15, 10, 2.3043568289094196], [10, 15, 1.923644959428831],
    [20, 5, -1.6491108182009562], [12.5, 12.5, 2.531750050755763]]) {
    const got = await score(`${model}lp = logdensityof(posterior, record(s = ${sv.toFixed(1)}, `
      + `b = ${bv.toFixed(1)}))`, 'lp');
    assert.ok(Math.abs(got - want) < 1e-12, `(${sv}, ${bv}): ${got} vs ${want}`);
  }
});

// `normalize` divides by the shape's mass on its support, also for a bounded
// Lebesgue shape: rate 3/2 on [0, 2], expected count 3, so 2·log 1.5 − 3.
test('a normalized bounded Lebesgue shape keeps the expected count', async () => {
  const I = 'weighted(3.0, normalize(Lebesgue(support = interval(0.0, 2.0))))';
  close(await score(`M = PoissonProcess(${I})\nL = logdensityof(M, [0.5, 1.5])`),
    2 * Math.log(1.5) - 3, 'density');
  checkDraws(await draws(`M = PoissonProcess(${I})\ns ~ M`), 3, 0, 2, 1, Math.sqrt(4 / 12), 'draws');
  close(await score('M = PoissonProcess(weighted(3.0, normalize(Normal(0.0, 1.0))))\n'
    + 'L = logdensityof(M, [0.5])'), Math.log(3) - 0.5 * Math.log(2 * Math.PI) - 0.125 - 3, 'untruncated');
});

// §03 predefined interval sets truncate like `interval(a, b)`: posreals keeps half
// of N(0, 1), so Λ = 1 and log λ(0.5) = log 2 + log φ(0.5).
test('a predefined interval set truncates an intensity', async () => {
  close(await score('M = PoissonProcess(truncate(weighted(2.0, Normal(0.0, 1.0)), posreals))\n'
    + 'L = logdensityof(M, [0.5])'), Math.log(2) - 0.5 * Math.log(2 * Math.PI) - 0.125 - 1, 'posreals');
  assert.equal(await score('M = PoissonProcess(truncate(weighted(2.0, Normal(0.0, 1.0)), unitinterval))\n'
    + 'L = logdensityof(M, [1.5])'), -Infinity, 'unitinterval');
});
