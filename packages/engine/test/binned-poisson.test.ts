'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const density = require('../density.ts');

function logPoisson(x: number, rate: number): number {
  let result = x * Math.log(rate) - rate;
  for (let k = 2; k <= x; k++) result -= Math.log(k);
  return result;
}

test('measure intensity gives Poisson bin masses in sampling and density', async () => {
  const N = 4096;
  const { ctx } = ctxFor(`
I = weighted(100, Uniform(interval(0, 1)))
M = BinnedPoissonProcess([-1, 0, 0.25, 1], I)
lp = logdensityof(M, [0, 25, 75])
`, N);
  const m = await ctx.getMeasure('M');
  const means = [0, 0, 0];
  let covariance = 0;
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 3; k++) means[k] += m.samples[i * 3 + k] / N;
    covariance += (m.samples[i * 3 + 1] - 25) * (m.samples[i * 3 + 2] - 75) / N;
  }
  assert.equal(means[0], 0);
  assert.ok(Math.abs(means[1] - 25) < 0.5);
  assert.ok(Math.abs(means[2] - 75) < 0.9);
  assert.ok(Math.abs(covariance) < 4);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - logPoisson(25, 25) - logPoisson(75, 75)) < 1e-10);
});

test('superposed intensities add bin masses without renormalizing to the bins', async () => {
  const { ctx } = ctxFor(`
I = superpose(logweighted(log(8), Uniform(interval(0, 2))), weighted(6, Uniform(interval(1, 3))))
M = BinnedPoissonProcess(bins=[0, 1, 2], intensity=I)
lp = logdensityof(M, [4, 7])
`, 4096);
  const m = await ctx.getMeasure('M');
  let first = 0, second = 0;
  for (let i = 0; i < 4096; i++) {
    first += m.samples[2 * i] / 4096;
    second += m.samples[2 * i + 1] / 4096;
  }
  assert.ok(Math.abs(first - 4) < 0.2);
  assert.ok(Math.abs(second - 7) < 0.3);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - logPoisson(4, 4) - logPoisson(7, 7)) < 1e-12);
});

test('conditional bin density resolves intensity weights and shape parameters per atom', () => {
  const lit = (value: number) => ({ kind: 'lit', value });
  const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
  const ir = { kind: 'call', op: 'BinnedPoissonProcess', args: [
    { kind: 'call', op: 'vector', args: [lit(0), lit(1)] },
    { kind: 'call', op: 'weighted', args: [ref('mass'),
      { kind: 'call', op: 'Uniform', args: [
        { kind: 'call', op: 'interval', args: [lit(0), ref('hi')] },
      ] },
    ] },
  ] };
  const result = density.logDensityN(ir, [2], {
    mass: Float64Array.of(4, 12), hi: Float64Array.of(2, 3),
  }, 2);
  assert.ok(Math.abs(result[0] - logPoisson(2, 2)) < 1e-12);
  assert.ok(Math.abs(result[1] - logPoisson(2, 4)) < 1e-12);
});

test('direct rates remain sampleable and score stably at large counts', async () => {
  const { ctx } = ctxFor(`
M = BinnedPoissonProcess(rates=[0, 2])
lp = logdensityof(M, [0, 2])
big = logdensityof(BinnedPoissonProcess(rates=[1000000000000]), [1000000000000])
`, 1024);
  const m = await ctx.getMeasure('M');
  for (let i = 0; i < 1024; i++) assert.equal(m.samples[2 * i], 0);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - logPoisson(2, 2)) < 1e-12);
  const big = await ctx.getMeasure('big');
  // Stirling's next omitted term is 1/(360*n^3), below 3e-39.
  const expected = -0.5 * Math.log(2 * Math.PI * 1e12) - 1 / (12e12);
  assert.ok(Math.abs(big.samples[0] - expected) < 1e-12);
});

test('point intensities honor half-open bins and the closed terminal edge', async () => {
  const { ctx } = ctxFor(`
I = truncate(superpose(weighted(2, Dirac(0)), weighted(3, Dirac(1)), weighted(5, Dirac(2))), interval(1, 2))
M = BinnedPoissonProcess([0, 1, 2], I)
lp = logdensityof(M, [0, 8])
`, 128);
  const m = await ctx.getMeasure('M');
  for (let i = 0; i < 128; i++) assert.equal(m.samples[2 * i], 0);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - logPoisson(8, 8)) < 1e-12);
});

test('restricted continuous intensity keeps its unnormalized bin mass', async () => {
  const { ctx } = ctxFor(`
I = truncate(weighted(4, Uniform(interval(0, 2))), interval(0, 1))
M = BinnedPoissonProcess([0, 1, 2], I)
lp = logdensityof(M, [0, 0])
`, 128);
  const lp = await ctx.getMeasure('lp');
  assert.equal(lp.samples[0], -2);
});

test('independent record intensity preserves Cartesian bin shape and law', async () => {
  const { proc, ctx } = ctxFor(`
I = weighted(12, joint(x=Uniform(interval(0, 2)), y=Uniform(interval(0, 3))))
M = BinnedPoissonProcess(record(x=[0, 0.5, 2], y=[0, 1, 2, 3]), I)
lp = logdensityof(M, rowstack([[1,1,1],[3,3,3]]))
counts ~ M
selected = counts[1,2]
`, 4096);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.deepEqual(proc.bindings.get('M').inferredType.domain.shape, [2, 3]);
  const m = await ctx.getMeasure('M');
  assert.deepEqual(m.value.shape, [4096, 2, 3]);
  const rates = [1, 1, 1, 3, 3, 3];
  for (let k = 0; k < 6; k++) {
    let mean = 0;
    for (let i = 0; i < 4096; i++) mean += m.samples[i * 6 + k] / 4096;
    assert.ok(Math.abs(mean - rates[k]) < 0.15);
  }
  const counts = await ctx.getMeasure('counts');
  const selected = await ctx.getMeasure('selected');
  for (let i = 0; i < 4096; i++) assert.equal(selected.samples[i], counts.samples[i * 6 + 1]);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - rates.reduce((s, r) => s + logPoisson(r, r), 0)) < 1e-12);
});

test('conditional bin generation retains weighted parameter ancestry', async () => {
  const { ctx } = ctxFor(`
theta ~ normalize(weighted(fn(_), Uniform(interval(0, 4))))
I = weighted(theta, Uniform(interval(0, 1)))
M = BinnedPoissonProcess([0, 1], I)
`, 4096);
  const m = await ctx.getMeasure('M');
  assert.ok(m.logWeights);
  let mass = 0, mean = 0;
  for (let i = 0; i < 4096; i++) {
    const weight = Math.exp(m.logWeights[i]);
    mass += weight;
    mean += weight * m.samples[i];
  }
  // Density of theta is theta/8 on [0,4], so E[count]=E[theta]=8/3.
  assert.ok(Math.abs(mean / mass - 8 / 3) < 0.15);
});

test('zero conditional intensity yields zero counts only for its own atoms', async () => {
  const { ctx } = ctxFor(`
theta ~ Bernoulli(0.5)
I = weighted(4 * theta, Uniform(interval(0, 1)))
M = BinnedPoissonProcess([0, 1], I)
`, 2048);
  const m = await ctx.getMeasure('M');
  const theta = await ctx.getMeasure('theta');
  let count = 0, sum = 0;
  for (let i = 0; i < 2048; i++) {
    if (theta.samples[i] === 0) assert.equal(m.samples[i], 0);
    else { count++; sum += m.samples[i]; }
  }
  assert.ok(Math.abs(sum / count - 4) < 0.25);
});

test('normalized restricted intensity divides by its own full mass', async () => {
  const { ctx } = ctxFor(`
I = weighted(4, normalize(truncate(Normal(0,1), interval(0,1))))
M = BinnedPoissonProcess([0,1], I)
lp = logdensityof(M, [0])
`, 1024);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] + 4) < 1e-12);
  const m = await ctx.getMeasure('M');
  const mean = m.samples.reduce((s: number, x: number) => s + x, 0) / 1024;
  assert.ok(Math.abs(mean - 4) < 0.3);
});

test('normalized intensity preserves its law across extreme common log scales', async () => {
  for (const shift of [-1000, 1000]) {
    const { ctx } = ctxFor(`
I = normalize(logweighted(${shift}, Uniform(interval(0,1))))
M = BinnedPoissonProcess([0,1], I)
lp = logdensityof(M, [0])
`, 1024);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] + 1) < 1e-12);
    const m = await ctx.getMeasure('M');
    const mean = m.samples.reduce((s: number, x: number) => s + x, 0) / 1024;
    assert.ok(Math.abs(mean - 1) < 0.15);
  }
});

test('discrete intensities respect integer atoms and half-open bin boundaries', async () => {
  const { ctx } = ctxFor(`
P = BinnedPoissonProcess([-0.2, 0.2, 1, 2], weighted(exp(1), Poisson(1)))
p = logdensityof(P, [1,0,1])
B = BinnedPoissonProcess([-1,0,1,2], weighted(8, Binomial(2,0.5)))
b = logdensityof(B, [0,2,6])
`, 4096);
  const p = await ctx.getMeasure('p'), b = await ctx.getMeasure('b');
  assert.ok(Math.abs(p.samples[0] - logPoisson(1, 1) - logPoisson(1, 1.5)) < 1e-12);
  assert.ok(Math.abs(b.samples[0] - logPoisson(2, 2) - logPoisson(6, 6)) < 1e-12);
  const m = await ctx.getMeasure('P');
  let first = 0, last = 0;
  for (let i = 0; i < 4096; i++) {
    first += m.samples[3 * i] / 4096;
    assert.equal(m.samples[3 * i + 1], 0);
    last += m.samples[3 * i + 2] / 4096;
  }
  assert.ok(Math.abs(first - 1) < 0.1);
  assert.ok(Math.abs(last - 1.5) < 0.1);
});

test('scaled discrete tail intensities retain finite bin masses', async () => {
  let logFactorial30 = 0;
  for (let k = 2; k <= 30; k++) logFactorial30 += Math.log(k);
  const { ctx } = ctxFor(`
p = logdensityof(BinnedPoissonProcess([30,31], logweighted(${1 + logFactorial30}, Poisson(1))), [0])
b = logdensityof(BinnedPoissonProcess([99.5,100], logweighted(100*log(2), Binomial(100,0.5))), [0])
r = logdensityof(BinnedPoissonProcess([0.5,1], logweighted(700, Bernoulli(exp(-700)))), [0])
`, 1);
  const p = await ctx.getMeasure('p'), b = await ctx.getMeasure('b'), r = await ctx.getMeasure('r');
  // Poisson's bin includes k=30 and31, giving scaled mass1+1/31;
  // Binomial's terminal atom is2^-100; Bernoulli's atom is exp(-700).
  assert.ok(Math.abs(p.samples[0] + 1 + 1 / 31) < 1e-12);
  assert.ok(Math.abs(b.samples[0] + 1) < 1e-12);
  assert.ok(Math.abs(r.samples[0] + 1) < 1e-12);
});

test('categorical intensity bins preserve category offsets, zero atoms and tiny masses', async () => {
  for (const [dist, offset] of [['Categorical', 1], ['Categorical0', 0]] as const) {
    const bins = [offset - 0.5, offset, offset + 0.5, offset + 1.5, offset + 2];
    const { ctx } = ctxFor(`
M = BinnedPoissonProcess([${bins}], weighted(5, ${dist}([0.2,0,0.8])))
lp = logdensityof(M, [0,1,0,4])
T = BinnedPoissonProcess([${offset + 1.5},${offset + 2}], logweighted(700, ${dist}([1-exp(-700),0,exp(-700)])))
tail = logdensityof(T, [0])
`, 1024);
    const lp = await ctx.getMeasure('lp'), tail = await ctx.getMeasure('tail');
    assert.ok(Math.abs(lp.samples[0] - logPoisson(1, 1) - logPoisson(4, 4)) < 1e-12);
    assert.ok(Math.abs(tail.samples[0] + 1) < 1e-12);
    const m = await ctx.getMeasure('T');
    assert.ok(Math.abs(m.samples.reduce((s: number, x: number) => s + x, 0) / 1024 - 1) < 0.15);
  }
});

test('geometric intensity bins use finite series and infinite tails without underflow', async () => {
  const { ctx } = ctxFor(`
M = BinnedPoissonProcess([-inf,0,1.5,inf], weighted(8, Geometric(0.25)))
lp = logdensityof(M, [0,4,4])
point = logdensityof(BinnedPoissonProcess([-1,0,1], Geometric(1)), [0,0])
tail = logdensityof(BinnedPoissonProcess([1999.5,2000], logweighted(2001*log(2), Geometric(0.5))), [0])
`, 2048);
  const lp = await ctx.getMeasure('lp'), point = await ctx.getMeasure('point'), tail = await ctx.getMeasure('tail');
  // P(K<2)=1-.75^2; P(K>=2)=.75^2.
  assert.ok(Math.abs(lp.samples[0] - logPoisson(4, 3.5) - logPoisson(4, 4.5)) < 1e-12);
  assert.equal(point.samples[0], -1);
  assert.ok(Math.abs(tail.samples[0] + 1) < 1e-10);
  const m = await ctx.getMeasure('M');
  let first = 0, last = 0;
  for (let i = 0; i < 2048; i++) {
    assert.equal(m.samples[3 * i], 0);
    first += m.samples[3 * i + 1] / 2048;
    last += m.samples[3 * i + 2] / 2048;
  }
  assert.ok(Math.abs(first - 3.5) < 0.25);
  assert.ok(Math.abs(last - 4.5) < 0.25);
});

test('noninteger negative-binomial shapes match bounded independent PMF sums', async () => {
  // Sum P(k+1)/P(k)=(k+r)q/(k+1); no registry or CDF participates.
  const mass = (r: number, p: number, q: number, a: number, b: number) => {
    let term = p ** r, total = 0;
    for (let k = 0; k <= b; k++) {
      if (k >= a) total += term;
      term *= (k + r) * q / (k + 1);
    }
    return total;
  };
  const expected = [
    [8 * mass(0.5, 0.5, 0.5, 0, 0), 0, 8 * mass(0.5, 0.5, 0.5, 1, 3)],
    [4 * mass(0.5, 1 / 7, 6 / 7, 0, 0), 4 * mass(0.5, 1 / 7, 6 / 7, 1, 1), 4 * mass(0.5, 1 / 7, 6 / 7, 2, 4)],
  ];
  const { ctx } = ctxFor(`
a = logdensityof(BinnedPoissonProcess([-0.2,0.2,1,3], weighted(8, NegativeBinomial(0.5,1))), [6,0,2])
b = logdensityof(BinnedPoissonProcess([0,1,2,4], weighted(4, NegativeBinomial2(3,0.5))), [2,1,1])
`, 1);
  for (const [i, name, counts] of [[0, 'a', [6, 0, 2]], [1, 'b', [2, 1, 1]]] as const) {
    const lp = await ctx.getMeasure(name);
    const oracle = counts.reduce<number>((s, x, j) => s + (expected[i][j] === 0 ? 0 : logPoisson(x, expected[i][j])), 0);
    assert.ok(Math.abs(lp.samples[0] - oracle) < 1e-12);
  }
});

test('negative-binomial tails and infinite support retain analytic bin masses', async () => {
  const scale = 102 * Math.LN2 - Math.log(101);
  const { ctx } = ctxFor(`
a = logdensityof(BinnedPoissonProcess([100,102], logweighted(${scale}, NegativeBinomial(2,1))), [0])
b = logdensityof(BinnedPoissonProcess([99.5,100], logweighted(${scale}, NegativeBinomial2(2,2))), [0])
whole = logdensityof(BinnedPoissonProcess([-inf,0,inf], NegativeBinomial(0.5,2)), [0,0])
end = logdensityof(BinnedPoissonProcess([-inf,1,inf], NegativeBinomial2(3,0.5)), [0,1])
`, 1);
  const a = await ctx.getMeasure('a'), b = await ctx.getMeasure('b');
  const whole = await ctx.getMeasure('whole'), end = await ctx.getMeasure('end');
  // For r=2,p=q=1/2, P(k)=(k+1)/2^(k+2), independent of the CDF identity.
  assert.ok(Math.abs(a.samples[0] + 1 + 102 / (2 * 101) + 103 / (4 * 101)) < 1e-12);
  assert.ok(Math.abs(b.samples[0] + 1) < 1e-12);
  assert.equal(whole.samples[0], -1);
  assert.ok(Math.abs(end.samples[0] + Math.sqrt(1 / 7) - logPoisson(1, 1 - Math.sqrt(1 / 7))) < 1e-12);
});


test('extreme finite NB2 bins use exact recurrence when Beta CDF loses its parameters', async () => {
  const { ctx } = ctxFor(`
a = logdensityof(BinnedPoissonProcess([0,1], NegativeBinomial2(1,1e300)), [0])
b = logdensityof(BinnedPoissonProcess([1,2], NegativeBinomial2(1,1e300)), [0])
`, 1);
  const a = await ctx.getMeasure('a'), b = await ctx.getMeasure('b');
  // The NB2 Poisson-limit corrections here are O(1e-300), far below tolerance.
  assert.ok(Math.abs(a.samples[0] + 2 / Math.E) < 1e-12);
  assert.ok(Math.abs(b.samples[0] + 1.5 / Math.E) < 1e-12);
});

test('open truncation endpoints exclude atoms through nested restrictions', async () => {
  const { ctx } = ctxFor(`
G = BinnedPoissonProcess([-1,1], truncate(Geometric(1), posreals))
g = logdensityof(G, [0])
d = logdensityof(BinnedPoissonProcess([-1,1], truncate(Dirac(0), posreals)), [0])
n = logdensityof(BinnedPoissonProcess([-1,1], truncate(truncate(Geometric(1), posreals), interval(0,1))), [0])
c = logdensityof(BinnedPoissonProcess([-1,1], truncate(Geometric(1), nonnegreals)), [0])
pi = logdensityof(BinnedPoissonProcess([-inf,inf], normalize(truncate(Dirac(inf), posreals))), [0])
ni = logdensityof(BinnedPoissonProcess([-inf,inf], normalize(truncate(Dirac(-inf), reals))), [0])
`, 32);
  for (const name of ['g', 'd', 'n']) assert.equal((await ctx.getMeasure(name)).samples[0], 0);
  assert.equal((await ctx.getMeasure('c')).samples[0], -1);
  assert.equal((await ctx.getMeasure('pi')).samples[0], -1);
  assert.equal((await ctx.getMeasure('ni')).samples[0], -1);
  for (const count of (await ctx.getMeasure('G')).samples) assert.equal(count, 0);
});

test('normalized discrete truncation uses the same exact rates for density and sampling', async () => {
  const { ctx } = ctxFor(`
D = Geometric(0.5)
E = D
I = weighted(3, normalize(truncate(E, interval(1,2))))
M = BinnedPoissonProcess([1,2,3], I)
lp = logdensityof(M, [2,1])
raw = logdensityof(BinnedPoissonProcess([1,2,3], truncate(E, interval(1,2))), [0,0])
inline = logdensityof(BinnedPoissonProcess([1,2,3], normalize(truncate(Geometric(0.5), interval(1,2)))), [0,0])
`, 2048);
  assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] - logPoisson(2, 2) - logPoisson(1, 1)) < 1e-12);
  assert.equal((await ctx.getMeasure('raw')).samples[0], -0.375);
  assert.ok(Math.abs((await ctx.getMeasure('inline')).samples[0] + 1) < 1e-12);
  const m = await ctx.getMeasure('M');
  let first = 0, second = 0;
  for (let i = 0; i < 2048; i++) {
    first += m.samples[2 * i] / 2048;
    second += m.samples[2 * i + 1] / 2048;
  }
  assert.ok(Math.abs(first - 2) < 0.15);
  assert.ok(Math.abs(second - 1) < 0.1);
});

test('negative-binomial bins retain relative precision at large counts', async () => {
  for (const [first, beta, count] of [[1e12, 1e-12, 2], [1e15, 1e-20, 257], [1e15, 1e-20, 1024]]) {
    const logQ = -Math.log1p(beta);
    const logMass = first * logQ + Math.log(-Math.expm1(count * logQ));
    for (const dist of [`NegativeBinomial(1,${beta})`, `NegativeBinomial2(${1 / beta},1)`]) {
      const { ctx } = ctxFor(`
lp = logdensityof(BinnedPoissonProcess([${first},${first + count - 1}], logweighted(${-logMass}, ${dist})), [0])
`, 1);
      // At shape1 all included atoms sum to the exact geometric series.
      assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] + 1) < 1e-10);
    }
  }
});

test('wide negative-binomial bins require well-conditioned CDF subtraction', async () => {
  const beta = 0.01, p = beta / (1 + beta), logQ = -Math.log1p(beta);
  // For shape2, P(K>=a)=q^a*(1+a*p), independently of Beta CDF wiring.
  const mass = Math.exp(100 * logQ) * (1 + 100 * p) - Math.exp(501 * logQ) * (1 + 501 * p);
  const { ctx } = ctxFor(`
good = logdensityof(BinnedPoissonProcess([100,500], NegativeBinomial(2,0.01)), [0])
bad = logdensityof(BinnedPoissonProcess([1e15,1000000000000256], NegativeBinomial(2,1e-20)), [0])
`, 1);
  assert.ok(Math.abs((await ctx.getMeasure('good')).samples[0] + mass) < 1e-12);
  await assert.rejects(ctx.getMeasure('bad'), /trustworthy analytic mass or at most 256 finite atoms/);
});
