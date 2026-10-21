'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { orchestrator } = require('../index.ts');
const density = require('../density.ts');
const values = require('../value.ts');

const close = (actual: number, expected: number, tolerance = 1e-11) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
async function evaluate(source: string): Promise<Float64Array> {
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  return (await ctx.getMeasure('score')).samples;
}
const betaLaw = `response(mu, phi, zi) = normalize(superpose(weighted(zi, Dirac(0.0)),
  weighted(1-zi, Beta(mu*phi, (1-mu)*phi))))`;

test('nested mixtures extend the common reference at distinct and repeated atoms', async () => {
  const source = `
inner = superpose(weighted(.2, Dirac(0.0)), weighted(.5, Beta(.5,.5)))
M = normalize(superpose(inner, weighted(.1, Dirac(0.0)), weighted(.2, Dirac(1.0))))
score = logdensityof.(M, [0.0, .25, 1.0])
`;
  const actual = await evaluate(source);
  const expected = [Math.log(.3), Math.log(.5 / (Math.PI * Math.sqrt(.25 * .75))), Math.log(.2)];
  expected
    .forEach((expected, i) => close(actual[i], expected));
  // A rank-one view keeps its logical order, and the typed buffer itself
  // carries the nonzero offset. Density must not read the backing buffer.
  const points = values.transpose({ shape: [3], data: Float64Array.of(9,0,.25,1,9).subarray(1,4) });
  const { ctx } = makeMatCtx(source);
  const result = density.logDensityN(orchestrator.expandMeasure('M', ctx), points, {}, 3, { pointsBatched: true });
  expected.forEach((expected, i) => close(result[i], expected));
});

test('callback Beta mixtures suppress singular endpoint density and shape gradients', async () => {
  const score = async (p: number[], x: number) => (await evaluate(`${betaLaw}
score = logdensityof(response(${p.join(',')}), ${x})`))[0];
  const p = [.5, 1, .3], h = 1e-5;
  for (const x of [0, .25]) {
    close(await score(p, x), x === 0 ? Math.log(.3)
      : Math.log(.7 / (Math.PI * Math.sqrt(x * (1-x)))));
    // Beta(1/2,1/2): digamma(1)-digamma(1/2)=2*log(2).
    const gradient = x === 0 ? [0, 0, 1/.3]
      : [Math.log(x/(1-x)), .5*Math.log(x*(1-x)) + 2*Math.log(2), -1/.7];
    for (let j = 0; j < p.length; j++) {
      const lo = p.slice(), hi = p.slice();
      lo[j] -= h; hi[j] += h;
      close(((await score(hi, x)) - (await score(lo, x))) / (2*h), gradient[j], 1e-8);
    }
  }
  const rows = await evaluate(`${betaLaw}
score = logdensityof(response.([.5,.5], [1,1], [.2,.3]), [0.0,.25])`);
  close(rows[0], Math.log(.2) + Math.log(.7 / (Math.PI*Math.sqrt(.25*.75))));
});

test('Gamma mixtures retain interior scores and zero atomic shape derivatives', async () => {
  const score = async (p: number[], x: number) => (await evaluate(`
response(alpha, rate, zi) = normalize(superpose(weighted(zi, Dirac(0.0)),
  weighted(1-zi, Gamma(alpha, rate))))
score = logdensityof(response(${p.join(',')}), ${x})`))[0];
  const p = [.5, .6, .3], h = 1e-5, euler = .5772156649015329;
  for (const x of [0, 1.2]) {
    close(await score(p, x), x === 0 ? Math.log(.3)
      : Math.log(.7) + .5*Math.log(.6) - .5*Math.log(Math.PI) - .5*Math.log(x) - .6*x);
    const gradient = x === 0 ? [0, 0, 1/.3]
      : [Math.log(.6*x) + euler + 2*Math.log(2), .5/.6-x, -1/.7];
    for (let j = 0; j < p.length; j++) {
      const lo = p.slice(), hi = p.slice();
      lo[j] -= h; hi[j] += h;
      close(((await score(hi, x)) - (await score(lo, x))) / (2*h), gradient[j], 1e-8);
    }
  }
});

test('pointwise Gamma scores retain the atomic mass through held measure refs', async () => {
  const actual = await evaluate(`
point(alpha,x) = logdensityof(normalize(superpose(weighted(.3,Dirac(0.0)),
  weighted(.7,Gamma(alpha,.6)))),x)
score = point.([.5,.5],[0.0,1.2])`);
  close(actual[0], Math.log(.3));
  close(actual[1], Math.log(.7) + .5*Math.log(.6) - .5*Math.log(Math.PI)
    - .5*Math.log(1.2) - .6*1.2);
});

test('pointwise Beta scores retain the atomic mass through held measure refs', async () => {
  const actual = await evaluate(`
point(alpha,x) = logdensityof(normalize(superpose(weighted(.3,Dirac(0.0)),
  weighted(.7,Beta(alpha,.5)))),x)
score = point.([.5,.5],[0.0,.25])`);
  close(actual[0], Math.log(.3));
  close(actual[1], Math.log(.7 / (Math.PI*Math.sqrt(.25*.75))));
});

test('pointwise mixed density still refuses an unproven moving atom', async () => {
  await assert.rejects(() => evaluate(`
point(a,x) = logdensityof(normalize(superpose(weighted(.3,Dirac(a)),
  weighted(.7,Beta(.5,.5)))),x)
score = point.([.2,.2],[.2,.25])`), /incommensurable reference/);
});

test('mixed iid likelihoods preserve record posterior scores and shared parameters', async () => {
  const score = async (p: number) => (await evaluate(`${betaLaw}
p ~ Beta(1,1)
y ~ iid(response(.5,1,p), 2)
P = lawof(record(p=p))
K = kernelof(record(y=y), p=p)
L = likelihoodof(pars -> K(p = pars.p), record(y=[0.0,.25]))
post = bayesupdate(L,P)
score = logdensityof(post, record(p=${p}))`))[0];
  close(await score(.3), Math.log(.3*.7 / (Math.PI*Math.sqrt(.25*.75))));
  close(((await score(.30001)) - (await score(.29999))) / .00002, 1/.3-1/.7, 1e-8);
});

test('retaining the selector uses the same fixed conditional reference', async () => {
  const source = `c ~ Bernoulli(.3)
y ~ ifelse(c, Normal(0,1), Dirac(0.0))
P = lawof(record(c=c,y=y))`;
  assert.equal((await evaluate(`${source}\nscore=logdensityof(P,record(c=true,y=0.0))`))[0], -Infinity);
  close((await evaluate(`${source}\nscore=logdensityof(P,record(c=false,y=0.0))`))[0], Math.log(.7));
  close((await evaluate(`${source}\nscore=logdensityof(P,record(c=true,y=.5))`))[0],
    Math.log(.3) - .5*Math.log(2*Math.PI) - .125);
  await assert.rejects(() => evaluate(`a~Normal(0,1)
c~Bernoulli(.3)
y~ifelse(c,Normal(0,1),Dirac(a))
P=lawof(record(a=a,c=c,y=y))
score=logdensityof(P,record(a=.2,c=false,y=.2))`), /incommensurable reference/);
});

test('fixed-reference scoring preserves normalization validity and refuses moving atoms', async () => {
  const source = `f(s)=normalize(superpose(weighted(.3,Dirac(0.0)),
    weighted(.7,normalize(weighted(s,Beta(.5,.5))))))`;
  close((await evaluate(`${source}\nscore=logdensityof(f.([1.0,1.0]),[0.0,0.0])`))[0], 2*Math.log(.3));
  await assert.rejects(() => evaluate(`${source}\nscore=logdensityof(f.([0.0,1.0]),[0.0,0.0])`), /ZERO total mass/);
  await assert.rejects(() => evaluate(`a~Normal(0,1)
score=logdensityof(superpose(weighted(.3,Dirac(a)),weighted(.7,Beta(.5,.5))),0.0)`), /incommensurable reference/);
});
