'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function values(source: string, names: string[], opts?: any) {
  const { ctx } = makeMatCtx(source, { sampleCount: 4, ...opts });
  const out: number[][] = [];
  for (const name of names) {
    const m = await ctx.getMeasure(name);
    out.push(Array.from(m.value?.data || m.samples));
  }
  return out;
}
const errors = (source: string) => processSource(source).diagnostics.filter((d: any) => d.severity === 'error');

// The result fields deliberately differ from the unary formal's name.
test('unary functions and fn receive records and tables whole', async () => {
  assert.deepEqual(await values(`
f(r) = r.x + r.nested.y
r = (x = 2, nested = (y = 3))
h = fn(_)
a = f(h(r))
b = h(table(x = [2, 5])).x
c = identity(r).nested.y
choose(g) = g
transported = choose(f)
d = transported(r)
`, ['a', 'b', 'c', 'd']), [[5], [2, 5], [3], [5]]);
});

test('explicit unary and inferred keyword-only inputs bind different record levels', async () => {
  const source = `
r = elementof(cartprod(x = reals))
explicit = functionof(r.x + 1, r)
inferred = functionof(r.x + 1)
a = explicit((x = 3))
b = inferred((r = (x = 3)))
alias = inferred
c = alias(r = (x = 3))
choose(f) = f
g = choose(inferred)
d = g((r = (x = 2)))
`;
  assert.deepEqual(await values(source, ['a', 'b', 'c', 'd']), [[4], [4], [4], [3]]);
  assert.ok(errors(source + '\ne = inferred((x = 3))').length);
  assert.ok(errors('x = elementof(reals); f = functionof(2*x); y = f(3)').length);
});

test('unary domain failure does not retry a matching record as keywords', async () => {
  assert.deepEqual(await values('p = Dirac((value = 2)); y = logdensityof(p, (value = 2))', ['y']), [[0, 0, 0, 0]]);
  assert.ok(errors('y = exp((x = 2))').length);
  assert.ok(errors('m = Poisson((rate = 2))').length);
});

test('multi-input tables bind columns by name and retain each column whole', async () => {
  assert.deepEqual(await values(`
f(a, b) = sum(a) - sum(b)
t = table(b = [1, 2], a = [5, 7])
y = f(t)
`, ['y']), [[9]]);
});

test('optional builtin inputs survive aliases and whole-argument composition', async () => {
  assert.deepEqual(await values(`
A = rowstack([[1, 2], [3, 4]])
d = diag
first(x) = x
h = fchain(first, d)
a = h(A)
b = d(k = 1, A = A)
`, ['a', 'b']), [[1, 4], [2]]);
  assert.ok(errors('y = diag((A = rowstack([[1, 2], [3, 4]])))').length);
});

test('composition and broadcasted aliases preserve whole row records', async () => {
  assert.deepEqual(await values(`
first(x) = (y = x)
second(r) = r.y + 1
h = fchain(first, second)
g = broadcasted(second)
a = h(2)
b = g(table(y = [2, 3]))
two(a, b) = a - 2*b
c = broadcast(two, table(b = [1, 2], a = [5, 7]))
d = broadcast(sub, b = [1, 2], a = [5, 7])
e = broadcast(sub, table(b = [1, 2], a = [5, 7]))
x = elementof(reals)
inferred = functionof(x + 1)
rows = broadcasted(inferred)
f = rows(table(x = [2, 3]))
`, ['a', 'b', 'c', 'd', 'e', 'f']), [[3], [3, 4], [3, 3], [4, 5], [4, 5], [3, 4]]);
});

test('imported aliases retain explicit and inferred input interfaces', async () => {
  assert.deepEqual(await values(`
m = load_module("callee.flatppl")
g = m.whole
h = m.named
a = g((x = 2))
b = h((r = (x = 2)))
`, ['a', 'b'], { path: 'main.flatppl', bundle: { sources: { 'callee.flatppl': `
r = elementof(cartprod(x = reals))
whole = functionof(3*r.x, r)
named = functionof(3*r.x)
` } } }), [[6], [6]]);
});

test('checked uses the ordinary binary interface', async () => {
  assert.deepEqual(await values(`
a = checked(3, true)
b = checked(condition = true, value = 3)
c = checked((condition = true, value = 3))
`, ['a', 'b', 'c']), [[3], [3], [3]]);
  assert.ok(errors('y = checked(3, condition = true)').length);
  assert.ok(errors('x ~ Normal(0,1); y = checked((value = 3, condition = x > 0))').length);
});

test('joint kernels expose the union of named inputs even when unary', async () => {
  const [score] = await values(`
K(mu) = Normal(mu, 1)
J = joint(a = K, b = K)
y = logdensityof(J((mu = 2)), (a = 1, b = -1))
`, ['y']);
  // Independent N(2,1) scores at +/-1: -log(2*pi) - (1+9)/2.
  for (const x of score) assert.ok(Math.abs(x + Math.log(2*Math.PI) + 5) < 1e-12);
});

test('record and positional disintegration preserve the analytic joint density', async () => {
  const [named, positional] = await values(`
a ~ Normal(0, 1)
b ~ Normal(a, 2)
Kr, Pr = disintegrate(["b"], lawof((a = a, b = b)))
Kv, Pv = disintegrate([2], lawof([a, b]))
r = logdensityof(jointchain(Pr, Kr), (a = 1, b = 2))
v = logdensityof(Pv, [1]) + logdensityof(Kv(complement = [1]), [2])
`, ['r', 'v']);
  const expected = -Math.log(2*Math.PI) - Math.log(2) - 0.5 - 1/8;
  for (const x of [...named, ...positional]) assert.ok(Math.abs(x - expected) < 1e-12);
});

test('indexed disintegration preserves shared selected draws', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0, 1)
b ~ Normal(a, 2)
c ~ Normal(b, 3)
K, P = disintegrate([2, 3], lawof([a, b, c]))
score = logdensityof(K([1]), [2, 4])
`, { sampleCount: 16 });
  const expected = -Math.log(2*Math.PI) - Math.log(6) - 1/8 - 2/9;
  for (const score of (await ctx.getMeasure('score')).samples) {
    assert.ok(Math.abs(score - expected) < 1e-12);
  }
});

test('record chain transitions consume the whole accumulated record in scoring and sampling', async () => {
  const { ctx } = makeMatCtx(`
K(r) = joint(b = Dirac(r.a + 1))
M = jointchain(joint(a = Dirac(2)), K)
x ~ M
score = logdensityof(M, (a = 2, b = 3))
`, { sampleCount: 4 });
  const x = await ctx.getMeasure('x');
  assert.deepEqual(Array.from(x.fields.a.samples), [2, 2, 2, 2]);
  assert.deepEqual(Array.from(x.fields.b.samples), [3, 3, 3, 3]);
  assert.deepEqual(Array.from((await ctx.getMeasure('score')).samples), [0, 0, 0, 0]);
});

test('likelihood points bind a unary input and named inputs by name', async () => {
  const { ctx } = makeMatCtx(`
K(x) = Normal(x, 1)
L = likelihoodof(K, 3)
x = elementof(reals)
named = functionof(Normal(x, 1))
Ln = likelihoodof(named, 3)
prior = joint(x = Normal(0, 1))
posterior = bayesupdate(L, prior)
a = logdensityof(L, (x = 2))
b = logdensityof(Ln, (x = 2))
c = logdensityof(posterior, (x = 2))
`, { sampleCount: 4 });
  const normal = (x: number) => -0.5*Math.log(2*Math.PI) - x*x/2;
  for (const name of ['a', 'b']) {
    const m = await ctx.getMeasure(name);
    assert.ok(Math.abs((m.samples || m.value.data)[0] - normal(1)) < 1e-12);
  }
  const c = await ctx.getMeasure('c');
  assert.ok(Math.abs(c.samples[0] - normal(1) - normal(2)) < 1e-12);
  const scorer = await require('../mcmc-density.ts').buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.logPi({ x: 2 }) - normal(1) - normal(2)) < 1e-12);
  assert.ok(Array.from((await ctx.getMeasure('posterior')).logWeights).every(Number.isFinite));
});
