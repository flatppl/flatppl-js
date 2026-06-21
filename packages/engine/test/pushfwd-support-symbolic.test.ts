'use strict';
const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const { ctxFor } = require('./density/regression-baseline.test.ts');
const ms         = require('../model-spec.ts');

function supOf(src: string, latentName: string): any {
  const { ctx } = ctxFor(src, 1);
  let d: any = null; for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) if (v && (v as any).kind === 'bayesupdate') d = v;
  const ls = ms.enumerateLatents(d, ctx);
  return ls.find((l: any) => l.name === latentName).support;
}
function model(decl: string): string {
  return decl + `
x ~ iid(m, 1)
k = (z) -> Normal(z, 1)
y ~ k.(x)
prior = lawof(record(x = x))
forward_kernel = kernelof(record(y = y), x = x)
L = likelihoodof(forward_kernel, record(y = [0.5]))
posterior = bayesupdate(L, prior)
`;
}

test('pushfwd image support: 0.1*exp over Exponential ⇒ greaterThan(0.1)', () => {
  const s = supOf(model(`m = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))`), 'x');
  assert.equal(s.kind, 'greaterThan');
  assert.ok(Math.abs(s.lo - 0.1) < 1e-12, `lo=${s.lo}`);
});
test('pushfwd image support: exp over a real base ⇒ positive', () => {
  const s = supOf(model(`m = pushfwd(fn(exp(_)), Normal(0, 1))`), 'x');
  assert.equal(s.kind, 'positive');
});
test('pushfwd image support: positive-affine over a real base ⇒ real', () => {
  const s = supOf(model(`m = pushfwd(fn(2.0 * _ + 1.0), Normal(0, 1))`), 'x');
  assert.equal(s.kind, 'real');
});
