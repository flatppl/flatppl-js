'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const close = (actual: number, expected: number, tol = 2e-12) =>
  assert.ok(Math.abs(actual - expected) <= tol, `${actual} != ${expected}`);
const odds = (p: number, theta: number) => p * Math.exp(theta) / (1 - p + p * Math.exp(theta));
const q = odds(0.2, 0.3);
const rate = 1.2 * Math.exp(0.3);
const geometric = 1 - 0.6 * Math.exp(0.3);
const beta = 2.5 * Math.exp(-0.3) - 1;
const mu2 = 6 * Math.exp(0.3) / (5 - 3 * Math.exp(0.3));
// Independently simplify §08 densities, including the inverse of the 2x2
// covariance (determinant 1.75). These do not call another engine scorer.
const cases = [
  { base: 'Normal(0.4, 1.5)', target: 'Normal(1.075, 1.5)', x: '0.2',
    expected: -Math.log(1.5 * Math.sqrt(2 * Math.PI)) - (0.2 - 1.075) ** 2 / 4.5 },
  { base: 'Gamma(2, 1.7)', target: 'Gamma(2, 1.4)', x: '0.8',
    expected: 2 * Math.log(1.4) + Math.log(0.8) - 1.4 * 0.8 },
  { base: 'Exponential(2)', target: 'Exponential(1.7)', x: '0.8',
    expected: Math.log(1.7) - 1.7 * 0.8 },
  { base: 'ChiSquared(4)', target: 'Gamma(2, 0.2)', x: '0.8',
    expected: 2 * Math.log(0.2) + Math.log(0.8) - 0.2 * 0.8 },
  { base: 'Bernoulli(0.2)', target: `Bernoulli(${q})`, x: 'true', expected: Math.log(q) },
  { base: 'Binomial(4, 0.2)', target: `Binomial(4, ${q})`, x: '2',
    expected: Math.log(6) + 2 * Math.log(q) + 2 * Math.log1p(-q) },
  { base: 'Poisson(1.2)', target: `Poisson(${rate})`, x: '2',
    expected: 2 * Math.log(rate) - rate - Math.log(2) },
  { base: 'MvNormal([0.4,-0.2],rowstack([[2,0.5],[0.5,1]]))',
    target: 'MvNormal([0.8,-0.45],rowstack([[2,0.5],[0.5,1]]))', x: '[0.2,0.1]',
    expected: -Math.log(2 * Math.PI) - Math.log(1.75) / 2
      - ((-0.6) ** 2 - (-0.6) * 0.55 + 2 * 0.55 ** 2) / 3.5 },
  { base: 'Geometric(0.4)', target: `Geometric(${geometric})`, x: '2',
    expected: Math.log(geometric) + 2 * Math.log1p(-geometric) },
  { base: 'NegativeBinomial(2,1.5)', target: `NegativeBinomial(2,${beta})`, x: '2',
    expected: Math.log(3) + 2 * Math.log(beta) - 4 * Math.log1p(beta) },
  { base: 'NegativeBinomial2(3,2)', target: `NegativeBinomial2(${mu2},2)`, x: '2',
    expected: Math.log(3) + 2 * Math.log(2) + 2 * Math.log(mu2) - 4 * Math.log(2 + mu2) },
];

test('affine log-weight laws share analytic density and RNG continuation', async () => {
  for (const [i, row] of cases.entries()) {
    const vector = i === 7;
    const weight = vector ? 'sum([0.3,-0.4] .* _) + 710' : '0.3*_ + 710';
    const reference = vector ? 'Lebesgue(cartpow(reals,2))'
      : i >= 4 && i !== 7 ? 'Counting(integers)' : 'Lebesgue(reals)';
    const { ctx } = makeMatCtx(`
B = ${row.base}
alias = B
w = fn(${weight})
E = logweighted(w, ${reference})
T = normalize(logweighted(w, alias))
U = tilt(E, alias)
P = ${row.target}
lp = logdensityof(T, ${row.x})
up = logdensityof(U, ${row.x})
s = rnginit(42)
x, sx = rand(T, s)
y, sy = rand(U, s)
z, sz = rand(P, s)
ax, _ = rand(iid(Normal(0,1),4), sx)
ay, _ = rand(iid(Normal(0,1),4), sy)
az, _ = rand(iid(Normal(0,1),4), sz)
`, { sampleCount: 3 });
    close((await ctx.getMeasure('lp')).samples[0], row.expected);
    close((await ctx.getMeasure('up')).samples[0], row.expected);
    const expected = await ctx.getMeasure('z');
    for (const n of ['x', 'y']) {
      const got = await ctx.getMeasure(n);
      // The inherited named-composite rand route seeds internal constructors
      // by name. Scalar leaves have exact replay; vector continuations below
      // agree, while vector sampling moments are checked separately.
      if (vector) continue;
      const data = (m: any) => m.value?.data || m.samples;
      // Parameter expressions can round by one ulp relative to the decimal
      // target. The same Philox stream and transformed leaf must still agree.
      assert.equal(data(got).length, data(expected).length);
      for (let j = 0; j < data(got).length; j++) close(data(got)[j], data(expected)[j]);
    }
    assert.deepEqual((await ctx.getMeasure('ax')).value.data, (await ctx.getMeasure('az')).value.data);
    assert.deepEqual((await ctx.getMeasure('ay')).value.data, (await ctx.getMeasure('az')).value.data);
  }
});

test('inferred Normal parameters and repeated tilts retain their score derivatives', async () => {
  const { ctx } = makeMatCtx(`
theta = elementof(reals)
mu = elementof(reals)
sigma = elementof(posreals)
T = normalize(logweighted(fn(theta * _ - 740), Normal(mu, sigma)))
U = normalize(logweighted(fn(0.2 * _), T))
lp = logdensityof(U, 0.4)
f = functionof(lp, theta = theta, mu = mu, sigma = sigma)
a = f(0.3, -0.2, 1.5)
b = f(-0.4, 0.7, 0.8)
lo = f(0.29999, -0.2, 1.5)
hi = f(0.30001, -0.2, 1.5)
`, { sampleCount: 2 });
  const score = async (n: string) => (await ctx.getMeasure(n)).samples[0];
  const oracle = (t: number, mu: number, sigma: number) =>
    -Math.log(sigma * Math.sqrt(2 * Math.PI)) - (0.4 - mu - (t + 0.2) * sigma ** 2) ** 2 / (2 * sigma ** 2);
  close(await score('a'), oracle(0.3, -0.2, 1.5));
  close(await score('b'), oracle(-0.4, 0.7, 0.8));
  close(((await score('hi')) - (await score('lo'))) / 0.00002, 0.4 - (-0.2 + 0.5 * 1.5 ** 2), 2e-10);
});

test('vector tilting materialises the analytic non-diagonal covariance law', async () => {
  const row = cases[7];
  const { ctx } = makeMatCtx(`
T = normalize(logweighted(fn(sum([0.3,-0.4] .* _)), ${row.base}))
lp = logdensityof(T, ${row.x})
`, { sampleCount: 4096 });
  close((await ctx.getMeasure('lp')).samples[0], row.expected);
  const data = (await ctx.getMeasure('T')).value.data;
  for (const [j, mean, variance] of [[0, 0.8, 2], [1, -0.45, 1]]) {
    let sum = 0;
    for (let i = j; i < data.length; i += 2) sum += data[i];
    close(sum / 4096, mean, 6 * Math.sqrt(variance / 4096));
  }
});

test('extreme odds retain rare-event scores, endpoints and repeated-tilt cancellation', async () => {
  const { ctx } = makeMatCtx(`
theta = elementof(reals)
p = elementof(unitinterval)
y = elementof(booleans)
T = normalize(logweighted(fn(theta * _), Bernoulli(p)))
lp = logdensityof(T, y)
f = functionof(lp, theta = theta, p = p, y = y)
a = f(1000, 0.2, false)
b = f(-1000, 0.2, true)
c = f(1000, 0, false)
d = f(-1000, 1, true)
e = f(1000, 0, true)
g = f(-1000, 1, false)
U = normalize(logweighted(fn(-1000 * _), normalize(logweighted(fn(1000 * _), Bernoulli(0.2)))))
u = logdensityof(U, true)
V = normalize(logweighted(fn(1000 * _), Binomial(4, 0.2)))
v = logdensityof(V, 2)
G = normalize(logweighted(fn(-1000 * _), Geometric(0.4)))
g1 = logdensityof(G, 1)
H = normalize(logweighted(fn(1000 * _), G))
h = logdensityof(H, 1)
`, { sampleCount: 2 });
  const score = async (n: string) => (await ctx.getMeasure(n)).samples[0];
  close(await score('a'), -1000 + Math.log(4));
  close(await score('b'), -1000 - Math.log(4));
  close(await score('c'), 0); close(await score('d'), 0);
  assert.equal(await score('e'), -Infinity); assert.equal(await score('g'), -Infinity);
  close(await score('u'), Math.log(0.2));
  close(await score('v'), Math.log(6) - 2000 + 2 * Math.log(4));
  close(await score('g1'), -1000 + Math.log(0.6));
  close(await score('h'), Math.log(0.4) + Math.log(0.6));
});

test('dynamic invalid laws and divergent Gamma-family tilts refuse before sampling', async () => {
  const laws = [
    ['Gamma(2, rate)', '-2', '-1'], // invalid original rate would become +1
    ['Exponential(rate)', '2', '2'], // MGF pole
    ['ChiSquared(rate)', '0.6', '4'], // divergent side
    ['Normal(0, rate)', '0', '-1'], // invalid source scale
    ['Bernoulli(rate)', '1000', '2'], // invalid p could round to a valid endpoint
    ['Geometric(rate)', '0', '0'],
    ['NegativeBinomial(2,rate)', '1', '1'],
    ['NegativeBinomial2(rate,2)', '1', '3'],
    ['Binomial(rate,0.2)', '0', '1.5'], // count must be an integer
    ['Normal(0,rate)', 'inf', '1'], // a nonfinite slope has no normalized law
  ];
  for (const [base, theta, rate] of laws) {
    const { ctx } = makeMatCtx(`
rate = elementof(reals)
theta = elementof(reals)
T = normalize(logweighted(fn(theta * _), ${base}))
K = functionof(T, rate = rate, theta = theta)
P = K(${rate}, ${theta})
x, _ = rand(P, rnginit(42))
`, { sampleCount: 2 });
    await assert.rejects(ctx.getMeasure('x'));
  }
  const { ctx } = makeMatCtx(`
c = elementof(reals)
T = normalize(logweighted(fn(_ + c), Normal(0,1)))
lp = logdensityof(T,0)
f = functionof(lp,c=c)
bad = f(inf)
good = f(710)
`, { sampleCount: 2 });
  await assert.rejects(ctx.getMeasure('bad'));
  close((await ctx.getMeasure('good')).samples[0], -0.5 * Math.log(2 * Math.PI) - 0.5);
});

test('zero tilt preserves large Normal scales and zero Poisson rate', async () => {
  const { ctx } = makeMatCtx(`
T = normalize(logweighted(fn(0 * _), Normal(0,1e200)))
a = logdensityof(T,0)
P = normalize(logweighted(fn(1000 * _), Poisson(0)))
b = logdensityof(P,0)
x, _ = rand(P,rnginit(42))
`, { sampleCount: 2 });
  close((await ctx.getMeasure('a')).samples[0], -200 * Math.log(10) - 0.5 * Math.log(2 * Math.PI));
  close((await ctx.getMeasure('b')).samples[0], 0);
  close((await ctx.getMeasure('x')).value.data[0], 0);
});

test('log odds survive serialized worker IR, aliases and renamed boundaries', async () => {
  const { orchestrator } = require('..');
  const density = require('../density.ts');
  const { ctx } = makeMatCtx(`
theta = elementof(reals)
p = elementof(unitinterval)
x = elementof(booleans)
w = functionof(theta*x, variate=x)
wa = w
T = normalize(logweighted(wa,Bernoulli(p)))
lp = logdensityof(T,false)
f = functionof(lp, slope=theta, probability=p)
a = f(1000,0.2)
`, { sampleCount: 2 });
  const expected = -1000 + Math.log(4);
  close((await ctx.getMeasure('a')).samples[0], expected);
  const ir = JSON.parse(JSON.stringify(orchestrator.expandMeasure('T', ctx)));
  close(density.logDensity(ir, false, { theta: 1000, p: 0.2 }), expected);
});

test('nonlinear closures and source-reference factors retain their generic refusal', async () => {
  for (const source of [
    'x=elementof(reals)\na=x*x\nw=functionof(a*x,x=x)\nT=normalize(logweighted(w,Normal(0,1)))',
    'T=tilt(logweighted(fn(_),Lebesgue(interval(-1,1))),Normal(0,1))',
    'T=tilt(logweighted(fn(_),Normal(0,1)),Normal(0,1))',
    'T=tilt(logweighted(fn(_),Lebesgue(posreals)),Exponential(2))',
  ]) {
    const { ctx } = makeMatCtx(source + '\nz,_=rand(T,rnginit(42))', { sampleCount: 2 });
    await assert.rejects(ctx.getMeasure('z'));
  }
});


test('log parameter tails retain finite scores and compose past sampler rounding', async () => {
  const { ctx } = makeMatCtx(`
P=normalize(logweighted(fn(-1000*_),Poisson(1)))
p=logdensityof(P,1)
p0=logdensityof(P,0)
Q=normalize(logweighted(fn(1000*_),P))
q=logdensityof(Q,1)
G=normalize(logweighted(fn(-40*_),Geometric(0.5)))
g0=logdensityof(G,0)
g1=logdensityof(G,1)
N=normalize(logweighted(fn(-1000*_),NegativeBinomial(2,2)))
n=logdensityof(N,1)
R=normalize(logweighted(fn(1000*_),N))
r=logdensityof(R,1)
V=normalize(logweighted(fn(0*_),NegativeBinomial2(1e-200,1e200)))
v0=logdensityof(V,0)
v1=logdensityof(V,1)
W=normalize(logweighted(fn(-1000*_),NegativeBinomial2(1,2)))
w=logdensityof(W,1)
w0=logdensityof(W,0)
X=normalize(logweighted(fn(1000*_),W))
wx=logdensityof(X,1)
x,_=rand(N,rnginit(42))
y,_=rand(P,rnginit(42))
A=normalize(logweighted(fn(-1000*_),NegativeBinomial(1e308,2)))
ax,_=rand(iid(A,128),rnginit(42))
a=logdensityof(A,1)
a0=logdensityof(A,0)
C=normalize(logweighted(fn(-1000*_),Binomial(1e308,0.5)))
c=logdensityof(C,1)
c0=logdensityof(C,0)
Z=normalize(logweighted(fn(0*_),NegativeBinomial2(1e200,1e-200)))
z=logdensityof(Z,0)
`, { sampleCount: 2 });
  const score = async (n: string) => (await ctx.getMeasure(n)).samples[0];
  close(await score('p'), -1000);
  close(await score('p0'), 0);
  close(await score('q'), -1);
  close(await score('g0'), -0.5*Math.exp(-40), 1e-30);
  close(await score('g1'), -40-Math.LN2);
  close(await score('n'), -1000 + Math.log(2/3));
  close(await score('r'), Math.log(8/27));
  close(await score('v0'), -1e-200, 1e-211);
  close(await score('v1'), -200*Math.log(10));
  close(await score('w'), -1000 + Math.log(2/3));
  close(await score('w0'), 0);
  close(await score('wx'), Math.log(8/27));
  close((await ctx.getMeasure('x')).value.data[0], 0);
  close((await ctx.getMeasure('y')).value.data[0], 0);
  // Capping beta while leaving the sampler oblivious to its log parameter
  // would produce mean 0.556 here, giving a nonzero count in these 128 draws.
  assert.ok((await ctx.getMeasure('ax')).value.data.every((v: number) => v === 0));
  close(await score('a'), 308*Math.log(10)-1000-Math.log(3));
  close(await score('a0'), -Math.exp(308*Math.log(10)-1000-Math.log(3)), 1e-137);
  close(await score('c'), 308*Math.log(10)-1000);
  close(await score('c0'), -Math.exp(308*Math.log(10)-1000), 1e-137);
  close(await score('z'), -400*Math.log(10)*1e-200, 1e-210);
});

test('generated constructors ignore local builtin shadowing and binomial endpoints retain one-sided scores', async () => {
  const { ctx } = makeMatCtx(`
Gamma=fn(_+1)
T=normalize(logweighted(fn(0.3*_),ChiSquared(4)))
t=logdensityof(T,0.8)
p=elementof(unitinterval)
B=normalize(logweighted(fn(0.3*_),Binomial(4,p)))
a=logdensityof(B,0)
b=logdensityof(B,4)
f=functionof(a,p=p)
g=functionof(b,p=p)
f0=f(0)
f1=f(0.0000001)
g0=g(1)
g1=g(0.9999999)
`, { sampleCount: 2 });
  const score = async (n: string) => (await ctx.getMeasure(n)).samples[0];
  close(await score('t'), cases[3].expected);
  close(((await score('f1'))-(await score('f0')))/1e-7, -4*Math.exp(0.3), 2e-6);
  close(((await score('g0'))-(await score('g1')))/1e-7, 4*Math.exp(-0.3), 2e-6);
});

test('named vector tilts compose before affine constructor lowering', async () => {
  const { ctx } = makeMatCtx(`
B=MvNormal([0.4,-0.2],rowstack([[2,0.5],[0.5,1]]))
T=normalize(logweighted(fn(sum([0.3,-0.4] .* _)),B))
U=normalize(logweighted(fn(sum([-0.3,0.4] .* _)),T))
lp=logdensityof(U,[0.2,0.1])
`, { sampleCount: 2 });
  close((await ctx.getMeasure('lp')).samples[0], -Math.log(2*Math.PI)-Math.log(1.75)/2
    - ((-0.2)**2-(-0.2)*0.3+2*0.3**2)/3.5);
});

test('formal expressions stay scoped through normalized exponential tilts', async () => {
  const { ctx } = makeMatCtx(`
K=functionof(normalize(logweighted(fn(0.3*_),Normal(_m_+1,1))),m=_m_)
P=K(2)
lp=logdensityof(P,0)
x,_=rand(P,rnginit(42))
Q=Normal(3.3,1)
y,_=rand(Q,rnginit(42))
`, { sampleCount: 2 });
  close((await ctx.getMeasure('lp')).samples[0], -0.5*Math.log(2*Math.PI)-3.3**2/2);
  close((await ctx.getMeasure('x')).value.data[0], (await ctx.getMeasure('y')).value.data[0]);
});


test('a scalar or length-one vector slope broadcasts over a multivariate event', async () => {
  const { ctx } = makeMatCtx(`
B=MvNormal([0.4,-0.2],rowstack([[2,0.5],[0.5,1]]))
T=normalize(logweighted(fn(sum(2 .* _)),B))
U=normalize(logweighted(fn(sum([2] .* _)),B))
t=logdensityof(T,[0.2,0.1])
u=logdensityof(U,[0.2,0.1])
`, { sampleCount: 2 });
  const expected = -Math.log(2*Math.PI)-Math.log(1.75)/2
    - ((-5.2)**2-(-5.2)*(-2.7)+2*(-2.7)**2)/3.5;
  close((await ctx.getMeasure('t')).samples[0], expected);
  close((await ctx.getMeasure('u')).samples[0], expected);
});


test('named callable captures expose inferred slopes before later reification', async () => {
  for (const definition of ['fn(theta*_+710)', 'functionof(theta*x+710,variate=x)']) {
    const { ctx } = makeMatCtx(`
theta=elementof(reals)
x=elementof(reals)
w=${definition}
wa=w
T=normalize(logweighted(wa,Normal(0,1)))
U=tilt(logweighted(wa,Lebesgue(reals)),Normal(0,1))
lp=logdensityof(T,0.4)
up=logdensityof(U,0.4)
f=functionof(lp,theta=theta)
g=functionof(up,theta=theta)
a=f(0.3)
b=g(-0.7)
`, { sampleCount: 2 });
    close((await ctx.getMeasure('a')).samples[0], -0.5*Math.log(2*Math.PI)-0.1**2/2);
    close((await ctx.getMeasure('b')).samples[0], -0.5*Math.log(2*Math.PI)-1.1**2/2);
  }
});


test('named vector weights expose live inferred slopes through affine lowering', async () => {
  const { ctx } = makeMatCtx(`
theta=elementof(cartpow(reals,2))
w=fn(sum(theta .* _))
B=MvNormal([0.4,-0.2],rowstack([[2,0.5],[0.5,1]]))
T=normalize(logweighted(w,B))
lp=logdensityof(T,[0.2,0.1])
f=functionof(lp,theta=theta)
a=f(theta=[0.3,-0.4])
b=f(theta=[-0.3,0.4])
`, { sampleCount: 2 });
  close((await ctx.getMeasure('a')).samples[0], cases[7].expected);
  close((await ctx.getMeasure('b')).samples[0], -Math.log(2*Math.PI)-Math.log(1.75)/2
    - (0.2**2-0.2*0.05+2*0.05**2)/3.5);
});

test('extreme count laws retain log parameters in fixed and conditional sampling', async () => {
  const { ctx } = makeMatCtx(`
A=normalize(logweighted(fn(0*_),NegativeBinomial2(1e-200,1e200)))
B=normalize(logweighted(fn(0*_),NegativeBinomial2(1e200,1e-200)))
a,_=rand(iid(A,128),rnginit(42))
b,_=rand(iid(B,128),rnginit(42))
alpha~Dirac(1e308)
mu~Dirac(1e-200)
c~normalize(logweighted(fn(-1000*_),NegativeBinomial(alpha,2)))
d~normalize(logweighted(fn(0*_),NegativeBinomial2(mu,1e200)))
`, { sampleCount: 128 });
  // Their zero-event masses are exp(-1e-200) and approximately
  // exp(-1e-200 * log(1e400)); both round to one at sampling precision.
  for (const name of ['a', 'b']) {
    const data = (await ctx.getMeasure(name)).value.data;
    assert.equal(data.length, 128);
    assert.ok(data.every((x: number) => x === 0));
  }
  for (const name of ['c', 'd']) {
    const samples = (await ctx.getMeasure(name)).samples;
    assert.equal(samples.length, 128);
    assert.ok(samples.every((x: number) => x === 0));
  }
});

test('broadcast tail scores retain support and zero-event probabilities', async () => {
  const { ctx } = makeMatCtx(`
P=normalize(logweighted(fn(-1000*_),Poisson(1)))
G=normalize(logweighted(fn(-40*_),Geometric(0.5)))
B=normalize(logweighted(fn(1000*_),Binomial(4,0.2)))
p=broadcast(logdensityof,P,[-1,0,0.5,1])
g=broadcast(logdensityof,G,[-1,0,0.5,1])
b=broadcast(logdensityof,B,[-1,0.5,5,2])
`, { sampleCount: 2 });
  const expected = [
    ['p', [-Infinity, 0, -Infinity, -1000]],
    ['g', [-Infinity, -0.5*Math.exp(-40), -Infinity, -40-Math.LN2]],
    ['b', [-Infinity, -Infinity, -Infinity, Math.log(6)-2000+2*Math.log(4)]],
  ] as const;
  for (const [name, scores] of expected) {
    const actual = (await ctx.getMeasure(name)).samples;
    assert.equal(actual.length, scores.length);
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] === -Infinity) assert.equal(actual[i], -Infinity);
      else close(actual[i], scores[i], name === 'g' && i === 1 ? 1e-30 : 2e-12);
    }
  }
});

test('a dynamic nonfinite vector slope refuses before sampling', async () => {
  const { ctx } = makeMatCtx(`
theta=elementof(cartpow(reals,2))
T=normalize(logweighted(fn(sum(theta .* _)),MvNormal([0,0],rowstack([[1,0],[0,1]]))))
K=functionof(T,theta=theta)
P=K(theta=[inf,0])
x,_=rand(P,rnginit(42))
`, { sampleCount: 2 });
  await assert.rejects(ctx.getMeasure('x'));
});
