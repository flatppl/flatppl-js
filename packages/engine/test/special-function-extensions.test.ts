'use strict';

// Independent mpmath 70-digit values, exact exponential/polynomial tails,
// and half-integer Bessel identities. Existing ordinary functions and their
// SciPy grid remain covered by special-functions.test.ts.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const sm = require('../standard-modules.ts');
const SF = 'sf=standard_module("special-functions","0.1")\n';

function close(actual: number, expected: number, rtol = 3e-12): void {
  assert.ok(Math.abs(actual - expected) <= rtol * Math.abs(expected) + 8 * Number.MIN_VALUE,
    `${actual} != ${expected}`);
}

async function values(expression: string): Promise<number[]> {
  const { ctx } = ctxFor(SF + `result=${expression}`, 1);
  const m = await ctx.getMeasure('result');
  return Array.from(m.samples ?? m.value?.data ?? m.value);
}

test('direct complements and log tails survive rounding and probability underflow', async () => {
  const got = await values(`[
    sf.gammaincc(a=1,x=40), sf.log_gammainc(2,1e-300), sf.log_gammaincc(2,1000),
    sf.gammaincc(0.00000001,1), sf.log_gammaincc(0.00000001,1),
    sf.betaincc(a=2,b=3,x=0.9999), sf.log_betainc(1000,2,0.001), sf.log_betaincc(2,1000,0.999),
    sf.log_gammainc(10000000,9800000), sf.log_gammaincc(10000000,10200000)]`);
  const expected = [Math.exp(-40), -1382.2442029769873, -993.0912452206848,
    2.1938393664027067e-9, -19.937612692722347, 3.999699999998678e-12,
    -6900.847523703155, -6900.847523703154, -2032.1393834049694, -1978.7932564033725];
  // The last two references integrate the normalized Gamma density at 70
  // digits around the tail endpoint; they do not use Temme's approximation.
  got.forEach((x, i) => close(x, expected[i]));
});

test('beta and its logarithm avoid products and differences of overflowing gamma values', async () => {
  const got = await values('[sf.beta(2,3),sf.logbeta(a=1000,b=2000),sf.logbeta(1e-12,2e-12)]');
  close(got[0], 1 / 12);
  close(got[1], -1911.8746142144516);
  close(got[2], Math.log(1.5e12));
});

test('tiny-shape log tails retain the correction to a probability of one', async () => {
  // B(a,2) gives P=x^a*(1+a*(1-x)) exactly. Gamma's leading lower
  // series is x^a/Gamma(a+1); later terms here are below binary64 precision.
  const got = await values(`[
    sf.betaincc(1e-20,2,0.01),sf.log_betainc(1e-20,2,0.01),sf.log_betaincc(1e-20,2,0.01),
    sf.log_gammainc(0.2,1e-300),sf.log_gammaincc(0.2,1e-300)]`);
  const coefficient = Math.log(0.01)+0.99;
  close(got[0],-1e-20*coefficient);
  close(got[1],1e-20*coefficient);
  close(got[2],Math.log(-1e-20*coefficient));
  close(got[3],-138.06973148963942);
  close(got[4],-1.0891244210583364e-60);
  for (const a of [0.009999,0.010001]) {
    const logP=a*Math.log(.2)+Math.log1p(a*.8);
    const joined=await values(`[sf.log_betainc(${a},2,.2),sf.betaincc(${a},2,.2)]`);
    close(joined[0],logP);
    close(joined[1],-Math.expm1(logP));
  }
});

test('scaled and log Bessel functions remain finite across the power-series and asymptotic joins', async () => {
  const got = await values(`[
    real(sf.bessel_i_scaled(0,1000)),sf.bessel_k_scaled(0,1000),
    sf.log_bessel_i(0,1000),sf.log_bessel_k(0,1000),
    real(sf.bessel_i_scaled(1000,1000000)),sf.log_bessel_i(1000,1000000),
    sf.bessel_k_scaled(1000,1000000),
    real(sf.bessel_i_scaled(63.999999,100)),real(sf.bessel_i_scaled(64,100)),
    real(sf.bessel_i_scaled(64.000001,100)),
    sf.bessel_k_scaled(0.5,1e-10),sf.log_bessel_k(50,1e-300),sf.log_bessel_i(0,1e-10)]`);
  const expected = [0.012617240455891257,0.03962832160075422,995.6273088898695,-1003.2282112244113,
    0.00024197070435489396,999991.6733061045,0.0020663648160767,
    8.737968619135429e-11,8.73796333175097e-11,8.737958044369636e-11,
    Math.sqrt(Math.PI / 2e-10),34717.306350704464,2.5e-21];
  got.forEach((x, i) => close(x, expected[i]));
});

test('scaled I preserves the complex principal branch and negative-order reflection', async () => {
  const got = await values(`[
    real(sf.bessel_i_scaled(0.5,-2)),imag(sf.bessel_i_scaled(v=0.5,z=-2)),
    real(sf.bessel_i_scaled(-0.5,2)),real(sf.bessel_i_scaled(1,-2))]`);
  assert.equal(got[0],0);
  close(got[1],(1-Math.exp(-4))/Math.sqrt(4*Math.PI));
  close(got[2],(1+Math.exp(-4))/Math.sqrt(4*Math.PI));
  close(got[3],-0.21526928924893765);
});

test('normal log CDF retains both the far lower tail and tiny upper correction', async () => {
  const got = await values('sf.log_ndtr.([-10000,-40,0,10,37])');
  const expected = [-50000010.12927891,-804.6084420137538,-Math.LN2,
    -7.619853024160525e-24,-5.725571222524577e-300];
  got.forEach((x, i) => close(x, expected[i]));
});

test('extended-real endpoint limits are retained by each companion family', async () => {
  const got = await values(`[
    sf.gammaincc(2,0),sf.log_gammainc(2,0),sf.log_gammaincc(2,0),
    sf.gammaincc(2,inf),sf.log_gammainc(2,inf),sf.log_gammaincc(2,inf),
    sf.betaincc(2,3,0),sf.log_betainc(2,3,0),sf.log_betaincc(2,3,0),
    sf.betaincc(2,3,1),sf.log_betainc(2,3,1),sf.log_betaincc(2,3,1),
    real(sf.bessel_i_scaled(0,0)),real(sf.bessel_i_scaled(1,0)),
    sf.log_bessel_i(0,0),sf.log_bessel_i(1,0),sf.bessel_k_scaled(0,0),sf.log_bessel_k(0,0),
    sf.log_ndtr(-inf),sf.log_ndtr(inf)]`);
  assert.deepEqual(got.map(x => x === 0 ? 0 : x), [1,-Infinity,0,0,0,-Infinity,
    1,-Infinity,0,0,0,-Infinity,1,0,0,-Infinity,Infinity,Infinity,-Infinity,0]);
});

test('all new callables evaluate at live inputs through aliases and named arguments', async () => {
  const { ctx } = makeMatCtx(SF + `
t ~ Normal(0,1)
x=exp(t)
p=invlogit(t)
f=sf.gammaincc
a=f(x=x,a=1)
b=sf.log_gammainc(1,x)
c=sf.log_gammaincc(1,x)
d=sf.betaincc(1,1,p)
e=sf.log_betainc(1,1,p)
f0=sf.log_betaincc(1,1,p)
g=sf.beta(1,x)
h=sf.logbeta(1,x)
i=real(sf.bessel_i_scaled(0.5,x))
j=sf.bessel_k_scaled(0.5,x)
k=sf.log_bessel_i(0.5,x)
l=sf.log_bessel_k(0.5,x)
n=sf.log_ndtr(t)
`, { sampleCount: 16, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples;
  const columns = await Promise.all(['a','b','c','d','e','f0','g','h','i','j','k','l','n']
    .map(async name => (await ctx.getMeasure(name)).samples));
  for (let r = 0; r < t.length; r++) {
    const x = Math.exp(t[r]), p = 1/(1+Math.exp(-t[r]));
    const logI = x - Math.log(2*Math.PI*x)/2 + Math.log1p(-Math.exp(-2*x));
    const expected = [Math.exp(-x),Math.log(-Math.expm1(-x)),-x,1-p,Math.log(p),Math.log1p(-p),
      1/x,-Math.log(x),-Math.expm1(-2*x)/Math.sqrt(2*Math.PI*x),Math.sqrt(Math.PI/(2*x)),
      logI,-x+Math.log(Math.PI/(2*x))/2];
    expected.forEach((v, c) => close(columns[c][r],v));
    // An independent Normal CDF identity, evaluated here away from its tail.
    close(Math.exp(columns[12][r]),0.5*sm.lookupStandardModule('special-functions','0.1')
      .bindings.get('erfc').impl(-t[r]/Math.SQRT2));
  }
});

test('loaded-module reexports retain named calls and scalar broadcasting', async () => {
  const { ctx } = makeMatCtx('m=load_module("functions.flatppl")\nresult=m.tail.(x=[1,2],a=1)', {
    sampleCount: 1, bundle: { sources: { 'functions.flatppl': SF+'tail=sf.log_gammaincc' } },
  });
  Array.from((await ctx.getMeasure('result')).samples as ArrayLike<number>)
    .forEach((x,i) => close(x,-(i+1)));
});

test('source finite differences match analytic derivatives of log tails and scaled Bessel functions', async () => {
  const source = (x: number) => `[
    sf.log_gammaincc(1,${x}),sf.log_betaincc(2,3,${x}/4),sf.logbeta(2,${x}),
    sf.log_bessel_i(0.5,${x}),sf.log_bessel_k(0.5,${x}),
    real(sf.bessel_i_scaled(0.5,${x})),sf.bessel_k_scaled(0.5,${x}),sf.log_ndtr(${x})]`;
  const x=2, step=1e-5, lo=await values(source(x-step)), hi=await values(source(x+step));
  const scaledI = -Math.expm1(-2*x)/Math.sqrt(2*Math.PI*x);
  const expected = [-1,-1.2,-1/x-1/(x+1),1/Math.tanh(x)-1/(2*x),-1-1/(2*x),
    scaledI*(2/Math.expm1(2*x)-1/(2*x)),-Math.sqrt(Math.PI/(2*x))/(2*x),
    0.05524786267898996];
  expected.forEach((v,i) => assert.ok(Math.abs((hi[i]-lo[i])/(2*step)-v)<2e-8));
});

test('invalid public domains fail without substituting a probability or scale', async () => {
  for (const expression of ['sf.gammaincc(-1,2)','sf.log_gammainc(1,-1)',
    'sf.log_gammaincc(inf,2)','sf.betaincc(1,2,2)','sf.log_betainc(0,1,0.5)',
    'sf.log_betaincc(1,-1,0.5)','sf.beta(-1,2)','sf.logbeta(1,inf)',
    'sf.log_bessel_i(-1,2)','sf.bessel_k_scaled(1,-1)','sf.log_bessel_k(1,-1)',
    'real(sf.bessel_i_scaled(1,inf))']) await assert.rejects(values(expression));
});
