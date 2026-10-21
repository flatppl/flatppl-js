'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('nested imported collection callbacks keep one whole parameter object', async () => {
  const { ctx } = makeMatCtx(`
lib = load_module("helpers.flatppl")
calc(p) = scan(lib.step, 0, [1, 2, 3], p)
a = calc((2, 1))
b = calc((3, -1))
c = calc((2, 1))
fold = reduce(lib.plus, [1, 2, 3], 4)
kept = filter(lib.pred, table(value = [1, 2, 3], other = [10, 20, 30]), (cut = 2))
result = kept.other
`, { sampleCount: 1, path: '/main.flatppl', bundle: { sources: {
    '/helpers.flatppl': `
step(a, x, p) = p[1] * a + x + p[2]
plus(a, x, p) = a + x + p
pred(row, p) = row.value > p.cut
`,
  } } });
  for (const [name, expected] of [['a', [2, 7, 18]], ['b', [0, 1, 5]], ['c', [2, 7, 18]],
    ['fold', [14]], ['result', [30]]] as const) {
    assert.deepEqual(Array.from((await ctx.getMeasure(name)).value.data), expected);
  }
});

test('scan and filter parameters remain live inside a scored callable', async () => {
  const { ctx } = makeMatCtx(`
p = elementof(reals)
states = scan((a, x, q) -> a + x * q, 0, [1, 2, 3], p)
selected = filter((x, q) -> x > q, [1, 2, 3], p)
M = Normal(states[3] + sum(selected), 1)
score = functionof(logdensityof(M, 0), p)
a = score(1)
b = score(2)
c = score(1)
`, { sampleCount: 1 });
  for (const [name, mean] of [['a', 11], ['b', 15], ['c', 11]] as const) {
    const expected = -Math.log(2 * Math.PI) / 2 - mean * mean / 2;
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('pushforward parameters bind forward inverse and volume callbacks', async () => {
  const { ctx } = makeMatCtx(`
p = elementof(cartpow(reals, 2))
B = bijection((x, q) -> q[1] * exp(x) + q[2],
              (y, q) -> log((y - q[2]) / q[1]),
              (x, q) -> x + log(q[1]))
M = pushfwd(B, Normal(0, 1), p)
score = functionof(logdensityof(M, 5), p)
a = score([2, 1])
b = score([4, 1])
c = score([2, 1])
D = pushfwd(B, Dirac(0), [2, 1])
sample, unused = rand(D, rnginit(42))
`, { sampleCount: 1 });
  for (const [name, scale] of [['a', 2], ['b', 4], ['c', 2]] as const) {
    const x = Math.log(4 / scale);
    const expected = -Math.log(2 * Math.PI) / 2 - x * x / 2 - Math.log(4);
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
  assert.equal((await ctx.getMeasure('sample')).value.data[0], 3);
});

test('affine pushforward parameters preserve the analytic Normal density', async () => {
  const { ctx } = makeMatCtx(`
p = elementof(cartpow(reals, 2))
M = pushfwd((x, q) -> q[1] * x + q[2], Normal(0, 1), p)
score = functionof(logdensityof(M, 2), p)
a = score([2, 1])
b = score([0.5, -1])
c = score([2, 1])
`, { sampleCount: 1 });
  for (const [name, scale, mean] of [['a', 2, 1], ['b', 0.5, -1], ['c', 2, 1]] as const) {
    const expected = -Math.log(scale) - Math.log(2 * Math.PI) / 2 - (2 - mean) ** 2 / (2 * scale ** 2);
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('heterogeneous parameter tuples preserve the analytic product of Normal measures', async () => {
  const { ctx } = makeMatCtx(`
mu = elementof(reals)
A = Normal(0, 1)
B = Normal(mu, 2)
C = Normal(1, 3)
M = normalize(logweighted((x, p) -> logdensityof(p[1], x) + logdensityof(p[2], x), A, (B, C)))
score = functionof(logdensityof(M, 1), mu)
a = score(2)
b = score(-1)
c = score(2)
`, { sampleCount: 1 });
  // Precision adds: 1 + 1/4 + 1/9 = 49/36. The mean is the
  // precision-weighted mean, (mu/4 + 1/9) / (49/36).
  const precision = 49 / 36;
  for (const [name, mu] of [['a', 2], ['b', -1], ['c', 2]] as const) {
    const mean = (mu / 4 + 1 / 9) / precision;
    const expected = (Math.log(precision) - Math.log(2 * Math.PI)) / 2
      - precision * (1 - mean) ** 2 / 2;
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('mixed measure and scalar parameters preserve additive log weights', async () => {
  const { ctx } = makeMatCtx(`
mu = elementof(reals)
theta = elementof(reals)
M = logweighted((x, p) -> logdensityof(p[1], x) + p[2]*x,
  Normal(mu, 1), (Normal(2*mu, 2), theta))
score = functionof(logdensityof(M, 0.3), mu, theta)
a = score(0.2, 0.4)
b = score(-0.5, -0.3)
c = score(0.2, 0.4)
`, { sampleCount: 1 });
  for (const [name, mu, theta] of [['a', 0.2, 0.4], ['b', -0.5, -0.3], ['c', 0.2, 0.4]] as const) {
    const expected = -Math.log(2 * Math.PI) - Math.log(2)
      - (0.3 - mu) ** 2 / 2 - (0.3 - 2*mu) ** 2 / 8 + 0.3*theta;
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('kernel and likelihood parameters preserve the same conditional density', async () => {
  const { ctx } = makeMatCtx(`
K(x) = Normal(x, 2)
L = likelihoodof(K, 3)
mu = elementof(reals)
implicit = likelihoodof(Normal(mu, 2), 3)
sigma = elementof(posreals)
recordlikelihood = likelihoodof(Normal(mu, sigma), 3)
a = logweighted((x, p) -> logdensityof(p[1](x), p[2]), Normal(0, 1), (K, 3))
b = weighted((x, p) -> densityof(p, x), Normal(0, 1), L)
c = logweighted((x, p) -> logdensityof(p, x), Normal(0, 1), L)
d = logweighted((x, p) -> logdensityof(p, x), Normal(0, 1), implicit)
recordweighted = weighted((x, p) -> densityof(p, x),
  joint(mu = Normal(0, 1), sigma = Dirac(2)), recordlikelihood)
la = logdensityof(a, 1)
lb = logdensityof(b, 1)
lc = logdensityof(c, 1)
ld = logdensityof(d, 1)
lr = logdensityof(recordweighted, (mu = 1, sigma = 2))
`, { sampleCount: 1 });
  // log phi(1) + log Normal(1, 2)(3).
  const expected = -Math.log(2 * Math.PI) - Math.log(2) - 1;
  for (const name of ['la', 'lb', 'lc', 'ld', 'lr']) {
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('collection callbacks can receive a fixed function as their parameter', async () => {
  const { ctx } = makeMatCtx(`
double(x) = 2*x
predicate(x) = x > 2
states = scan((a, x, p) -> p(a+x), 0, [1, 2, 3], double)
selected = filter((x, p) -> p(x), [1, 2, 3], predicate)
`, { sampleCount: 1 });
  assert.deepEqual(Array.from((await ctx.getMeasure('states')).value.data), [2, 8, 22]);
  assert.deepEqual(Array.from((await ctx.getMeasure('selected')).value.data), [3]);
});

test('collection parameter forms preserve vector and accumulator domains', () => {
  const { processSource } = require('../index.ts');
  for (const source of [
    'f(a,x,p)=Normal(a+x,p)\nout=scan(f,0,[1,2],1)',
    'f(a,x,p)=(value=a+x)\nout=scan(f,0,[1,2],1)',
    'f(a,x,p)=a+x+p\nout=scan(f,0,rowstack([[1,2],[3,4]]),1)',
    'f(x)=x>1\nout=filter(f,[1,2],mystery=5)',
    'f(x,p)=x>p\nout=filter(f,[1,2],1,p=2)',
  ]) assert.ok(processSource(source).diagnostics.some((d: any) => d.severity === 'error'), source);
});

test('parameter specialization preserves generated builtins under user shadowing', async () => {
  for (const prefix of ['logweighted(a,b)=Dirac(100)', 'exp(x)=100',
    'functionof(x)=0', 'logdensityof(a,b)=100']) {
    const { ctx } = makeMatCtx(prefix + `
K(x) = Normal(x, 2)
L = likelihoodof(K, 3)
M = weighted((x, p) -> densityof(p, x), Normal(0, 1), L)
lp = base.logdensityof(M, 1)
`, { sampleCount: 1 });
    const expected = -Math.log(2 * Math.PI) - Math.log(2) - 1;
    assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0] - expected) < 1e-12, prefix);
  }
});
