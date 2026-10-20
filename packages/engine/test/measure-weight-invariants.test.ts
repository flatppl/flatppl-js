'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

function context(source: string, count = 64) {
  const { proc, ctx } = ctxFor(source, count);
  const errors = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errors.length, 0, errors.map((d: any) => d.message).join('\n'));
  ctx.rootKey = [4711, 0x9e3779b9];
  return ctx;
}

test('an affine conditional measure carries its parameter weight once', async () => {
  // §06 locscale retains the current theta in both the affine shift and the
  // base. Its N(0,1) proposal targets N(1,1), with centered log ratio theta_i-theta_0.
  for (const base of ['Normal(0,1)', 'Normal(theta,1)']) {
    const ctx = context(`theta ~ normalize(weighted(fn(exp(_)), Normal(0,1)))
x ~ locscale(${base}, theta, 1)`);
    const theta = await ctx.getMeasure('theta');
    const x = await ctx.getMeasure('x');
    for (let i = 0; i < x.samples.length; i++) {
      const actual = (x.logWeights?.[i] ?? 0) - (x.logWeights?.[0] ?? 0);
      assert.ok(Math.abs(actual - theta.samples[i] + theta.samples[0]) < 1e-12);
    }
  }
});

test('callable reweighting counts the base mass once for each variate shape', async () => {
  // §06 dν=f dM: every shape is one atom of mass 3 with weight 2.
  for (const [base, weight] of [
    ['Dirac(2)', 'fn(_)'],
    ['joint(a=Dirac(2))', 'fn(_.a)'],
    ['iid(Dirac(2),2)', 'fn(_[1])'],
  ]) {
    const ctx = context(`M=weighted(${weight}, weighted(3,${base}))
z=totalmass(M)`, 8);
    assert.ok(Math.abs((await ctx.getMeasure('z')).samples[0] - 6) < 1e-12);
  }
  const zero = context('M=weighted(fn(_),weighted(3,Dirac(0)))\nz=totalmass(M)', 8);
  assert.equal((await zero.getMeasure('z')).samples[0], 0);
});

test('normalizing a unit vector atom preserves its subsequent variate weight', async () => {
  // §06 normalization is the identity on this unit atom [2,2].
  const ctx = context(`M=weighted(fn(_[1]),normalize(iid(Dirac(2),2)))
z=totalmass(M)`, 8);
  assert.ok(Math.abs((await ctx.getMeasure('z')).samples[0] - 2) < 1e-12);
});

test('a callable weight carries its explicit parameter law', async () => {
  // §06 conditional normalization gives x|theta ~ N(theta,1), now sampled
  // exactly. Only the supplied theta proposal carries importance weight;
  // the conditional law must not reapply its old theta*x-theta²/2 ratio.
  const ctx = context(`theta~normalize(weighted(fn(exp(_)),Normal(0,1)))
x~normalize(logweighted((v,t)->t*v,Normal(0,1),theta))`);
  const theta = await ctx.getMeasure('theta');
  const x = await ctx.getMeasure('x');
  const logRatio = (i: number) => theta.samples[i];
  for (let i = 0; i < x.samples.length; i++) {
    const actual = x.logWeights[i] - x.logWeights[0];
    assert.ok(Math.abs(actual - logRatio(i) + logRatio(0)) < 1e-12);
  }
});

test('a truncation indicator preserves shared parameter weight provenance', async () => {
  // GeneralizedNormal uses the filter fallback. The conditional chain retains
  // its base t's exp(t) importance event exactly once on accepted atoms.
  const ctx = context(`theta~normalize(weighted(fn(exp(_)),Normal(0,1)))
K=t->truncate(GeneralizedNormal(t,1,2),interval(-1,1))
J=jointchain(t=lawof(theta),y=K)`);
  const joint = await ctx.getMeasure('J');
  const theta = joint.fields.t.samples;
  const accepted = Array.from(joint.logWeights, (_w: number, i: number) => i)
    .filter((i: number) => Number.isFinite(joint.logWeights[i]));
  assert.ok(accepted.length > 1);
  const first = accepted[0];
  for (const i of accepted) {
    const actual = joint.logWeights[i] - joint.logWeights[first];
    assert.ok(Math.abs(actual - theta[i] + theta[first]) < 1e-12);
  }
});

test('a conditional superposition retains point values and their weighting event', async () => {
  // Both branches are Dirac(theta), including the affine spelling. One reified
  // law retains x=theta and the parameter's weighting event once (§04).
  for (const component of ['Dirac(theta)', 'locscale(Dirac(0),theta,1)']) {
    const ctx = context(`theta~normalize(weighted(fn(_+1),Bernoulli(.5)))
M=normalize(superpose(${component},${component}))
x~M
J=lawof(record(t=theta,x=x))`);
    const joint = await ctx.getMeasure('J');
    const t = joint.fields.t.samples, x = joint.fields.x.samples;
    for (let i = 0; i < t.length; i++) {
      assert.equal(x[i], t[i]);
      const actual = joint.logWeights[i] - joint.logWeights[0];
      assert.ok(Math.abs(actual - (t[i] - t[0]) * Math.log(2)) < 1e-12);
    }
  }
});

test('unequal conditional mixture branches retain both independent parameter laws', async () => {
  // Both priors belong to the joint, including the unselected branch's.
  const ctx = context(`a~normalize(weighted(fn(1+_),Bernoulli(.5)))
b~normalize(weighted(fn(1+3*_),Bernoulli(.5)))
M=normalize(superpose(Dirac(2*a),weighted(3,Dirac(2*b+1))))
x~M
J=lawof(record(a=a,b=b,x=x))`);
  const joint = await ctx.getMeasure('J');
  const a = joint.fields.a.samples, b = joint.fields.b.samples, x = joint.fields.x.samples;
  for (let i = 0; i < x.length; i++) {
    assert.equal(x[i], x[i] % 2 ? 2 * b[i] + 1 : 2 * a[i]);
    const expected = (a[i] - a[0]) * Math.log(2) + (b[i] - b[0]) * Math.log(4);
    assert.ok(Math.abs(joint.logWeights[i] - joint.logWeights[0] - expected) < 1e-12);
  }
});

test('a zero conditional mixture slice retains zero weight', async () => {
  // θ has target P(θ=1)=2/3. Weighting its joint law with an independent
  // Bernoulli variate by 2θ gives the zero slice and total mass 4/3 (§06).
  // A zero-mass slice admits neither normalization nor a draw.
  const ctx = context(`theta~normalize(weighted(fn(1+_),Bernoulli(.5)))
u~Bernoulli(.5)
J=weighted(r->ifelse(r.t,2.0,0.0),lawof(record(t=theta,x=u)))`);
  const joint = await ctx.getMeasure('J');
  const theta = joint.fields.t.samples;
  assert.ok(theta.includes(0) && theta.includes(1));
  for (let i = 0; i < theta.length; i++) {
    if (theta[i] === 0) assert.equal(joint.logWeights[i], -Infinity);
    else {
      assert.ok(Number.isFinite(joint.logWeights[i]));
      assert.ok(joint.fields.x.samples[i] === 0 || joint.fields.x.samples[i] === 1);
    }
  }
});

test('iid of a variate-weighted mixture retains its product law', async () => {
  // Component masses are {false:1/2,true:1} and {false:1}; after
  // normalization M=Bernoulli(2/5), so its product assigns 4/25 to [1,1].
  const ctx = context(`A=weighted(fn(1+_),Bernoulli(.5))
M=normalize(superpose(A,Bernoulli(0)))
V=iid(M,2)`, 8192);
  const scalar = await ctx.getMeasure('M');
  const product = await ctx.getMeasure('V');
  const x = product.samples, lw = product.logWeights;
  const max = lw ? Math.max(...lw) : 0;
  let mass = 0, first = 0, second = 0, both = 0;
  for (let i = 0; i < x.length / 2; i++) {
    const w = Math.exp((lw?.[i] ?? 0) - max);
    mass += w;
    first += w * x[2 * i];
    second += w * x[2 * i + 1];
    both += w * x[2 * i] * x[2 * i + 1];
  }
  const scalarMean = scalar.samples.reduce((a: number, b: number) => a + b, 0)
    / scalar.samples.length;
  assert.ok(Math.abs(scalarMean - .4) < .025);
  assert.ok(Math.abs(first / mass - .4) < .025);
  assert.ok(Math.abs(second / mass - .4) < .025);
  assert.ok(Math.abs(both / mass - .16) < .025);
});

test('a conditional iid draw preserves parameter lineage in one joint law', async () => {
  for (const base of ['Normal(theta,1)', 'locscale(Dirac(0),theta,1)']) {
    const ctx = context(`theta~normalize(weighted(fn(_+1),Bernoulli(.5)))
V=iid(${base},2)
x~V
J=lawof(record(t=theta,x=x))`);
    const joint = await ctx.getMeasure('J');
    const t = joint.fields.t.samples;
    // The normalized parameter law has target/proposal ratio proportional
    // to 1+theta. Neither a conditional product nor its joint repeats it.
    for (let i = 0; i < t.length; i++) {
      const actual = joint.logWeights[i] - joint.logWeights[0];
      assert.ok(Math.abs(actual - (t[i] - t[0]) * Math.log(2)) < 1e-12);
    }
  }
});
