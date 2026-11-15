'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { builtinLogdensityof } = require('../density-prims.ts');
const { processSource } = require('../index.ts');
const { PARAMS } = require('../regression.ts');
const L = Math.log(2), X = [[1, 0], [0, 1], [1, 1]];

// Rational PMFs and integer Gamma/Beta normalizers, with t(1)=Cauchy.
// Independently checked with Julia Distributions and ForwardDiff.
const cases = [
  ['BernoulliLogitGLM', {}, [1, 0, 1], -1.5040773967762742,
    { alpha: .5, beta: [5 / 6, 1 / 6] }],
  ['BinomialLogitGLM', { n: 3 }, [2, 1, 0], -3.7013019741124937,
    { alpha: -1.5, beta: [-1.5, -1.5] }],
  ['NormalGLM', { sigma: 2 }, [1, 2, .5], -5.785907189913391,
    { alpha: .875, beta: [.20171320486001368, .7982867951399863], sigma: -.5503499513804633 }],
  ['PoissonLogGLM', {}, [0, 2, 3], -7.371201010907891,
    { alpha: 1.5, beta: [0, 3.5] }],
  ['NegativeBinomial2LogGLM', { psi: 2 }, [0, 2, 3], -6.673317721049169,
    { alpha: 1.5333333333333332, beta: [1 / 3, 2.533333333333333], psi: -.1717558399823199 }],
  ['StudentTGLM', { nu: 1, sigma: 2 }, [1, 2, .5], -6.631866806461046,
    { alpha: .8638442867539018, beta: [.3851919821229869, .7139464222779738],
      sigma: -.77363762707436, nu: .7466861109889706 }],
  ['GammaLogGLM', { shape: 2 }, [1, 2, .5], -5.8411169166403285,
    { alpha: 4, beta: [-2, 5], shape: -1.1889114636155655 }],
  ['OrderedLogitGLM', { alpha: undefined, cutpoints: [-L, L] }, [1, 2, 3], -3.912023005428146,
    { beta: [-2 / 15, 29 / 30], cutpoints: [-1 / 30, -2 / 15] }],
  ['BetaLogitGLM', { precision: 6 }, [.25, .75, .5], -2.2730571050925494,
    { alpha: 0, beta: [-2.575927496001924, 2.575927496001924], precision: -.4944388214654315 }],
  ['CategoricalLogitGLM', { alpha: [0, 0, 0], beta: [[L, 0, -L], [0, L, 0]] }, [1, 2, 3],
    -3.4499875458315876, { alpha: [-.2658730158730158, -.23015873015873012, .496031746031746],
      beta: [[-.015873015873015817, -.7301587301587301, .746031746031746],
        [-.6944444444444444, .05555555555555558, .6388888888888888]] }],
] as any[];

function value(v: any): any {
  return Array.isArray(v) ? { shape: Array.isArray(v[0]) ? [v.length, v[0].length] : [v.length],
    data: Float64Array.from(v.flat()) } : v;
}
function literal(v: any): string { return JSON.stringify(v); }
function params(extra: any): any { return { x: X, alpha: 0, beta: [L, -L], ...extra }; }
function near(got: number, want: number, tol = 2e-12): void {
  assert.ok(Math.abs(got - want) <= tol, `${got} != ${want}`);
}
function score(name: string, kw: any, y: any): number {
  return builtinLogdensityof(name, Object.fromEntries(Object.entries(kw)
    .filter(([, v]) => v !== undefined).map(([k, v]) => [k, value(v)])), value(y));
}

test('all regression laws score and draw vector responses through qualified calls', async () => {
  for (const [name, extra, y, expected] of cases) {
    const p = params(extra);
    const args = PARAMS[name].map((k: string) => `${k} = ${Array.isArray(p[k]?.[0])
      ? `rowstack(${literal(p[k])})` : literal(p[k])}`).join(', ');
    const src = `reg = standard_module("regression", "0.1")\nM = reg.${name}(${args})\n`
      + `lp = logdensityof(M, ${literal(y)})\ny ~ M\nr, s = rand(M, rnginit(123))\nnr, ns = rand(normalize(M), rnginit(123))\nmass = totalmass(M)\n`
      + `pos = logdensityof(reg.${name}(${PARAMS[name].map((k: string) =>
        Array.isArray(p[k]?.[0]) ? `rowstack(${literal(p[k])})` : literal(p[k])).join(', ')}), ${literal(y)})\n`;
    const { ctx, lifted } = makeMatCtx(src, { sampleCount: 8 });
    assert.deepEqual(lifted.bindings.get('M').inferredType.domain.shape, [3]);
    for (const v of (await ctx.getMeasure('lp')).samples) near(v, expected);
    for (const v of (await ctx.getMeasure('pos')).samples) near(v, expected);
    assert.deepEqual((await ctx.getMeasure('y')).value.shape, [8, 3]);
    assert.deepEqual((await ctx.getMeasure('r')).value.shape, [3]);
    assert.deepEqual((await ctx.getMeasure('nr')).value.data, (await ctx.getMeasure('r')).value.data);
    assert.equal((await ctx.getMeasure('mass')).value.data[0], 1);
    assert.ok(processSource(`${name}(${args})`).diagnostics.some((d: any) => d.severity === 'error'));
  }
});

test('regression draws match independent first-row CDF probabilities', async () => {
  // Exact Bernoulli/Binomial/NB/ordinal/categorical masses, Gamma(2,1)
  // survival exp(-x)(1+x), Beta(4,2) CDF 5x^4-4x^5, and symmetric medians.
  const cdfs: Record<string, [number, number]> = {
    BernoulliLogitGLM: [0, 1 / 3], BinomialLogitGLM: [1, 7 / 27],
    NormalGLM: [L, .5], PoissonLogGLM: [1, 3 * Math.exp(-2)],
    NegativeBinomial2LogGLM: [1, .5], StudentTGLM: [L, .5],
    GammaLogGLM: [1, 1 - 2 / Math.E], OrderedLogitGLM: [1, .2],
    BetaLogitGLM: [.5, 3 / 16], CategoricalLogitGLM: [1, 4 / 7],
  };
  const { walk } = require('../sampler.ts'), rng = require('../rng.ts');
  const draws = 4096;
  for (const [name, extra] of cases) {
    const p = params(extra), kwargs = Object.fromEntries(PARAMS[name].map((key: string) =>
      [key, { kind: 'ref', ns: 'self', name: key }]));
    const env = Object.fromEntries(Object.entries(p).filter(([, v]) => v !== undefined)
      .map(([key, v]) => [key, value(v)]));
    let state = rng.stateFromKey(1234, 5678), count = 0;
    const [threshold, probability] = cdfs[name];
    for (let i = 0; i < draws; i++) {
      const r = walk(state, { kind: 'call', op: name, kwargs }, env);
      state = r.state;
      count += +(r.value.data[0] <= threshold);
    }
    near(count / draws, probability, 6 * Math.sqrt(probability * (1 - probability) / draws));
  }
});

test('regression scores and all parameter derivatives match independent analytic values', () => {
  for (const [name, extra, y, expected, gradient] of cases) {
    const p = params(extra);
    near(score(name, p, y), expected);
    for (const [key, want] of Object.entries(gradient)) {
      const flat = Array.isArray(want) ? want.flat() : [want];
      for (let i = 0; i < flat.length; i++) {
        const a = structuredClone(p), b = structuredClone(p), h = 1e-5;
        if (Array.isArray(want)) {
          if (Array.isArray(want[0])) {
            const cols = want[0].length, row = Math.floor(i / cols), col = i % cols;
            a[key][row][col] += h; b[key][row][col] -= h;
          } else { a[key][i] += h; b[key][i] -= h; }
        } else { a[key] += h; b[key] -= h; }
        near((score(name, a, y) - score(name, b, y)) / (2 * h), flat[i] as number, 2e-8);
      }
    }
  }
});

test('link-scale scoring retains finite tail probabilities and the StudentT normal limit', () => {
  const x = [[1]], beta = [0];
  near(score('BernoulliLogitGLM', { x, beta, alpha: 1000 }, [0]), -1000);
  near(score('BinomialLogitGLM', { x, beta, alpha: 1000, n: 3 }, [1]), Math.log(3) - 2000);
  near(score('NegativeBinomial2LogGLM', { x, beta, alpha: 1000, psi: 2 }, [0]),
    -2000 + 2 * Math.log(2));
  near(score('NegativeBinomial2LogGLM', { x, beta, alpha: -40, psi: 2 }, [0]),
    -4.248354255291589e-18, 1e-31);
  near(score('OrderedLogitGLM', { x, beta: [1000], cutpoints: [-1, 1] }, [2]),
    -999 + Math.log1p(-Math.exp(-2)));
  near(score('GammaLogGLM', { x, beta, alpha: 800, shape: 1 }, [1]), -800);
  near(score('BetaLogitGLM', { x, beta, alpha: 1000, precision: 2 }, [.5]),
    -1000 + Math.log(2));
  near(score('StudentTGLM', { x, beta, alpha: 0, sigma: 2, nu: Infinity }, [1]),
    -.5 * Math.log(2 * Math.PI) - Math.log(2) - .125);
  near(score('StudentTGLM', { x, beta, alpha: 0, sigma: 1, nu: 1 }, [1e200]),
    -922.1787670834677);
});

test('regression density rejects points outside support and preserves captured parameters', async () => {
  for (const [name, extra] of cases) {
    if (name === 'NormalGLM' || name === 'StudentTGLM') continue;
    assert.equal(score(name, params(extra), [-1, -1, -1]), -Infinity);
  }
  const { ctx } = makeMatCtx(`
reg = standard_module("regression", "0.1")
a ~ Normal(0, 1)
beta ~ iid(Normal(0, 1), 2)
y ~ reg.NormalGLM(rowstack([[1, 0], [0, 1]]), a, beta, 1)
lp = logdensityof(lawof(record(a = a, beta = beta, y = y)), record(a = 0, beta = [1, 2], y = [1, 2]))
`, { sampleCount: 2 });
  for (const lp of (await ctx.getMeasure('lp')).samples) near(lp, -2.5 * Math.log(2 * Math.PI) - 2.5);
  assert.deepEqual((await ctx.getMeasure('y')).value.shape, [2, 2]);
});

test('constructor aliases and outer broadcasts preserve whole matrix and vector cells', async () => {
  const { ctx } = makeMatCtx(`
reg = standard_module("regression", "0.1")
G = reg.NormalGLM
x = rowstack([[1.0], [2.0]])
xs = [x, rowstack([[3.0], [4.0]])]
M = G.(x = xs, alpha = 0.0, beta = [[1.0]], sigma = 1.0)
q ~ M
y ~ G(x, 0.0, [1.0], 1.0)
lp = logdensityof(G(x, 0.0, [1.0], 1.0), [1.0, 2.0])
blp = logdensityof(M, [[1.0, 2.0], [3.0, 4.0]])
`, { sampleCount: 4 });
  for (const lp of (await ctx.getMeasure('lp')).samples) near(lp, -Math.log(2 * Math.PI));
  for (const lp of (await ctx.getMeasure('blp')).samples) near(lp, -2 * Math.log(2 * Math.PI));
  assert.deepEqual((await ctx.getMeasure('y')).value.shape, [4, 2]);
  const q = (await ctx.getMeasure('q')).value;
  assert.deepEqual(q.shape, [4, 2, 2]);
  assert.equal(q.outerRank, 2);
  const { ctx: bad } = makeMatCtx(`
reg = standard_module("regression", "0.1")
y ~ reg.NormalGLM.(rowstack([[1.0], [2.0]]), 0.0, [[1.0]], 1.0)
`, { sampleCount: 1 });
  await assert.rejects(bad.getMeasure('y'));
});
