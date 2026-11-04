'use strict';

// A length-1 vector latent (iid(D, 1), a pushfwd of it, a 1-D MvNormal) is a
// vector, not a scalar (spec §03, §06). Its numbers equal the matching scalar
// model's, with the variate kept as a 1-element vector.
//
// Oracles (scipy 1.18): x ~ N(1, 2²), y | x ~ N(x, 1), y = 0 observed.
// log p(x = 0.5, y = 0) = norm.logpdf(0.5, 1, 2) + norm.logpdf(0, 0.5, 1).
// Evidence log Z = norm.logpdf(0, 1, √5). Under pushfwd(fn(2.0 * _), ·) the
// prior is N(2, 4²), so log Z = norm.logpdf(0, 2, √17).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { materialiser } = require('..');
const { buildPriorTransform } = require('../prior-transform.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');

const model = (M: string) => `
x ~ ${M}
prior = lawof(record(x = x))
y ~ Normal(x[1], 1.0)
K = kernelof(record(y = y), x = x)
L = likelihoodof(K, record(y = 0.0))
posterior = bayesupdate(L, prior)
`;

const IID = 'iid(Normal(1.0, 2.0), 1)';
const PUSH = 'pushfwd(fn(2.0 * _), iid(Normal(1.0, 2.0), 1))';
const MV = 'MvNormal([1.0], rowstack([[4.0]]))';
const U = 0.8413447460685429;   // Φ(1)

test('model view scores a length-1 vector latent as a vector', async () => {
  const cases: [string, number][] = [
    [IID, -2.6872742469692907],
    [PUSH, -3.4194839275292357],
    [MV, -2.6872742469692907],
  ];
  for (const [M, want] of cases) {
    const { ctx } = ctxFor(model(M), 200);
    const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
    assert.deepEqual(mv.latentShapes[0].dims, [1], M);
    const got = mv.logPosterior(Float64Array.of(0.5));
    assert.ok(Math.abs(got - want) < 1e-12, `${M}: ${got}, want ${want}`);
  }
});

test('prior transform realises a length-1 vector latent as a 1-element vector', () => {
  const cases: [string, number][] = [[IID, 3], [PUSH, 6], [MV, 3]];
  for (const [M, want] of cases) {
    const { ctx } = ctxFor(model(M), 10);
    const pt = buildPriorTransform(ctx, ctx.derivations.posterior);
    assert.equal(pt.dim, 1);
    const x = pt.transform(Float64Array.of(U)).x;
    assert.ok(x instanceof Float64Array && x.length === 1, `${M}: ${x}`);
    assert.ok(Math.abs(x[0] - want) < 1e-12, `${M}: ${x[0]}, want ${want}`);
  }
});

test('prior transform realises iid(D, 3) and a pushfwd of it as 3-vectors', () => {
  // iid: 1 + 2·Φ⁻¹(u) per coordinate; the pushfwd doubles it (scipy 1.18).
  const cases: [string, number][] = [['iid(Normal(1.0, 2.0), 3)', 1], ['pushfwd(fn(2.0 * _), iid(Normal(1.0, 2.0), 3))', 2]];
  for (const [M, scale] of cases) {
    const { ctx } = ctxFor(model(M), 10);
    const x = buildPriorTransform(ctx, ctx.derivations.posterior).transform(Float64Array.of(0.2, 0.5, U)).x;
    assert.equal(x.length, 3);
    const want = [-0.68324247, 1, 3].map((v) => scale * v);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(x[i] - want[i]) < 1e-8, `${M}: x[${i}] = ${x[i]}`);
  }
});

test('prior transform refuses iid of a length-1 vector measure as a non-scalar element', () => {
  const { ctx } = ctxFor(model('iid(iid(Normal(0.0, 1.0), 1), 3)'), 10);
  assert.throws(() => buildPriorTransform(ctx, ctx.derivations.posterior),
    /iid inner measure must be scalar per element/);
});

test('mh and nested posteriors of a length-1 vector latent match the conjugate Normal', async () => {
  // Prior N(m, s²), one unit-variance observation at 0: posterior variance
  // v = 1/(1/s² + 1), mean v·m/s². IID: m = 1, s = 2. PUSH: m = 2, s = 4.
  const cases: [string, number, number, number][] = [
    [IID, 0.2, 0.8, -1.823657489421723],
    [PUSH, 2 / 17, 16 / 17, -2.45319226405631],
  ];
  const N = 4000;
  for (const backend of ['mh', 'nested']) {
    for (const [M, mean, variance, logZ] of cases) {
      const { ctx } = ctxFor(model(M), N);
      const m = await materialiser.materialiseMeasure('posterior', ctx,
        { backend, chains: 4, warmup: 500, draws: 1000, nLive: 400, seed: 3 });
      const x = m.fields.x.value;
      assert.deepEqual(x.shape, [N, 1]);
      const ess = m.diagnostics.perParam['x[0]'].essBulk;
      let s = 0, s2 = 0;
      for (const v of x.data) { s += v; s2 += v * v; }
      const est = s / N, estVar = s2 / N - est * est;
      const what = `${backend} ${M}`;
      assert.ok(Math.abs(est - mean) < 5 * Math.sqrt(variance / ess), `${what}: mean ${est}, want ${mean}`);
      assert.ok(Math.abs(estVar - variance) < 5 * variance * Math.sqrt(2 / ess),
        `${what}: variance ${estVar}, want ${variance}`);
      if (backend === 'nested') {
        const { logZ: got, logZerr } = m.diagnostics;
        assert.ok(Math.abs(got - logZ) < 5 * logZerr, `${what}: log Z ${got} ± ${logZerr}, want ${logZ}`);
      }
    }
  }
});
