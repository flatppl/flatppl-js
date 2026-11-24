'use strict';

// §06 `locscale(m, shift, scale)` is the map x ↦ scale·x + shift. Over a
// vector law with a scalar scale, §07 `mul` scales every coordinate and `add`
// needs a shift of the variate's length. The map takes the pushforward's
// diagonal route, so a simplex law keeps its (n − 1)·log|scale| volume, and a
// drawn scale is a live per-atom coefficient. It used to throw "could not
// statically determine the scale shape".

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function errors(src: string): string[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

async function score(measure: string, point: string): Promise<number> {
  const src = `M = ${measure}\nlp = logdensityof(M, ${point})\n`;
  assert.deepEqual(errors(src), [], measure);
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('lp')).samples[0];
}

async function draws(src: string, name: string, N: number): Promise<number[][]> {
  assert.deepEqual(errors(src), [], src);
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  const width = m.value.shape[1];
  const data = Array.from(m.samples ?? m.value.data) as number[];
  return Array.from({ length: N }, (_, i) => data.slice(i * width, (i + 1) * width));
}

const close = (got: number, want: number, what: string) =>
  assert.ok(Math.abs(got - want) < 1e-12, `${what}: ${got} vs ${want}`);
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

test('a scalar scale and a vector shift over a simplex law take the free-coordinate volume', async () => {
  // Dirichlet(1, 1) has density 1 on its simplex; the volume is (2 − 1)·log 2.
  close(await score('locscale(Dirichlet([1.0, 1.0]), [3.0, 3.0], 2.0)', '[3.5, 4.5]'),
    -Math.log(2), 'Dirichlet(1, 1)');
  // Dirichlet(2, 3, 4) has density 3360·x1·x2²·x3³ (Γ(9) / (Γ(2)Γ(3)Γ(4))) at x = [0.2, 0.3, 0.5].
  // scipy.stats.dirichlet.logpdf([0.2, 0.3, 0.5], [2, 3, 4]) - 2 * np.log(2) = 0.6365768290715528
  const want = Math.log(3360 * 0.2 * 0.3 ** 2 * 0.5 ** 3) - 2 * Math.log(2);
  close(want, 0.6365768290715528, 'closed form against scipy');
  close(await score('locscale(Dirichlet([2.0, 3.0, 4.0]), [1.0, 1.0, 1.0], 2.0)', '[1.4, 1.6, 2.0]'),
    want, 'Dirichlet(2, 3, 4)');
});

test('the pushfwd spellings of a vector shift agree with locscale', async () => {
  for (const [map, point] of [['fn(2.0 .* _ .+ [3.0, 4.0])', '[3.5, 5.5]'],
    ['fn(2.0 * _ + [3.0, 4.0])', '[3.5, 5.5]'], ['fn(2.0 .* _ .- [3.0, 4.0])', '[-2.5, -2.5]']]) {
    close(await score(`pushfwd(${map}, Dirichlet([1.0, 1.0]))`, point), -Math.log(2), map);
  }
});

test('a scalar scale over a full-dimensional vector law scores the change of variables', async () => {
  // scipy.stats.norm.logpdf((np.array([1.5, 4.0]) - [1.0, 2.0]) / 3).sum() - 2 * np.log(3) = -4.271212754856676
  close(await score('locscale(iid(Normal(0.0, 1.0), 2), [1.0, 2.0], 3.0)', '[1.5, 4.0]'),
    -4.271212754856676, 'iid base');
  // scipy.stats.norm.logpdf(np.array([1.0, -1.0]) / 2).sum() - 2 * np.log(2) = -3.474171427529236
  close(await score('locscale(iid(Normal(0.0, 1.0), 2), [0.0, 0.0], 2.0)', '[1.0, -1.0]'),
    -3.474171427529236, 'zero shift');
  // scipy.stats.multivariate_normal.logpdf((np.array([0.5, 0.5]) - [1.0, 2.0]) / 3, [0, 0], np.eye(2))
  //   - 2 * np.log(3) = -4.173990532634454
  close(await score('locscale(MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]])), '
    + '[1.0, 2.0], 3.0)', '[0.5, 0.5]'), -4.173990532634454, 'MvNormal base');
});

test('a fixed vector shift is per coordinate when the atom count equals the length', async () => {
  // N = D = 3: read as one shift per atom, the rows would sum to 5, 8 and 11.
  for (const row of await draws('theta ~ locscale(Dirichlet([2.0, 3.0, 4.0]), [1.0, 2.0, 3.0], 2.0)\n', 'theta', 3)) {
    assert.ok(Math.abs(sum(row) - 8) < 1e-12, `${row}`);
    row.forEach((y, j) => assert.ok(y - (j + 1) > 0 && y - (j + 1) < 2, `${row}`));
  }
});

test('a drawn vector shift moves each atom by its own draw', async () => {
  const src = 'b ~ iid(Normal(0.0, 1.0), 3)\ntheta ~ locscale(Dirichlet([2.0, 3.0, 4.0]), b, 2.0)\n';
  const { ctx } = ctxFor(src, 4);
  const theta = await ctx.getMeasure('theta'), b = await ctx.getMeasure('b');
  const t = Array.from(theta.samples ?? theta.value.data) as number[];
  const s = Array.from(b.samples ?? b.value.data) as number[];
  for (let n = 0; n < 4; n++) {
    const row = t.slice(3 * n, 3 * n + 3), shift = s.slice(3 * n, 3 * n + 3);
    assert.ok(Math.abs(sum(row) - (2 + sum(shift))) < 1e-12, `${row} with ${shift}`);
  }
});

test('a latent scale scores the non-centred likelihood', async () => {
  const src = (shift: string) => 'mu ~ Normal(0, 10)\ntau ~ Exponential(1)\n'
    + `obs ~ locscale(iid(Normal(0.0, 1.0), 3), ${shift}, tau)\n`
    + 'fk = kernelof(record(obs = obs), mu = mu, tau = tau)\n'
    + 'L = likelihoodof(fk, record(obs = [1.5, -0.5, 2.0]))\n'
    + 'ld = logdensityof(L, record(mu = 0.5, tau = 2.0))\n';
  // scipy.stats.norm.logpdf([1.5, -0.5, 2.0], loc=0.5, scale=2.0).sum() = -5.367507141293855
  // scipy.stats.norm.logpdf([1.5, -0.5, 2.0], loc=[0.5, 1.5, 2.5], scale=2.0).sum() = -5.492507141293855
  for (const [shift, want] of [['fill(mu, 3)', -5.367507141293855],
    ['mu .+ [0.0, 1.0, 2.0]', -5.492507141293855]] as [string, number][]) {
    assert.deepEqual(errors(src(shift)), [], shift);
    const { ctx } = ctxFor(src(shift), 64);
    close((await ctx.getMeasure('ld')).samples[0], want, shift);
  }
});

test('draws under a latent scale are standard normal once the scale is divided out', async () => {
  const N = 4000;
  for (const [scale, factor] of [['tau', 1], ['2.0 * tau', 2]] as [string, number][]) {
    const src = `tau ~ Exponential(1.0)\ntheta ~ locscale(iid(Normal(0.0, 1.0), 3), [1.0, 2.0, 3.0], ${scale})\n`;
    assert.deepEqual(errors(src), [], scale);
    const { ctx } = ctxFor(src, N);
    const theta = await ctx.getMeasure('theta'), tau = await ctx.getMeasure('tau');
    const t = Array.from(theta.samples ?? theta.value.data) as number[];
    const s = Array.from(tau.samples ?? tau.value.data) as number[];
    const z = t.map((y, i) => (y - (i % 3 + 1)) / (factor * s[Math.floor(i / 3)]));
    const mean = sum(z) / z.length, variance = sum(z.map((v) => (v - mean) ** 2)) / z.length;
    // Five standard errors at 12000 values: 0.046 for the mean, 0.065 for the variance.
    assert.ok(Math.abs(mean) < 0.046 && Math.abs(variance - 1) < 0.065, `${scale}: ${mean}, ${variance}`);
  }
});

test('shapes the map does not take are refused with their reason', () => {
  assert.deepEqual(errors('M = locscale(Dirichlet([2.0, 3.0, 4.0]), 1.0, 2.0)\n'),
    ['locscale: a scalar shift is incompatible with a non-scalar variate (spec §06); use '
      + 'iid(locscale(M, shift, scale), size) for independent scalar location-scale transforms']);
  assert.deepEqual(errors('M = locscale(Dirichlet([2.0, 3.0, 4.0]), [1.0, 1.0], 2.0)\n'),
    ['locscale: a shift of length 2 does not match the variate of length 3; shift and scale must be '
      + 'value-compatible with the variate (spec §06)']);
  for (const m of ['iid(Normal(0.0, 1.0), 3), [1.0, 2.0, 3.0]', 'Normal(0.0, 1.0), 1.0']) {
    assert.deepEqual(errors(`M = locscale(${m}, 0.0)\n`),
      ['locscale() scale must be a nonzero finite value; a zero or non-finite scale is a degenerate, '
        + 'non-invertible affine map (spec §06 "Engine contract for `pushfwd` density evaluation": '
        + 'affine maps with nonzero scaling)'], m);
  }
});

test('a zero scale at a point names locscale', async () => {
  const src = 'mu ~ Normal(0, 10)\ntau ~ Exponential(1)\n'
    + 'obs ~ locscale(iid(Normal(0.0, 1.0), 3), fill(mu, 3), tau)\n'
    + 'fk = kernelof(record(obs = obs), mu = mu, tau = tau)\n'
    + 'L = likelihoodof(fk, record(obs = [1.5, -0.5, 2.0]))\n'
    + 'ld = logdensityof(L, record(mu = 0.5, tau = 0.0))\n';
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(ctx.getMeasure('ld'),
    /pushfwd bijection 'locscale' registry call to 'diagonal_affine'.*inverse requires finite nonzero scale/);
});
