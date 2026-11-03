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

// A throw from the forward function is not an infinite limit. The body below
// throws only at the far probe (u = 60). No source model reaches this, since
// lift moves literal vectors into anons, which throw at every probe.
test('pushfwd image support: a throw at a probe ends the derivation', () => {
  const u = { kind: 'ref', ns: '%local', name: 'u' };
  const lit = (value: any) => ({ kind: 'lit', value });
  const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
  const scaledExp = call('mul', lit(0.1), call('exp', u));
  const throwsAbove31 = call('get', lit([0.0]), call('add', lit(1), call('floor', call('divide', u, lit(31)))));
  assert.equal(ms.pushfwdImageSupport(call('add', scaledExp, throwsAbove31), 'u', { kind: 'positive' }), null);
  assert.deepEqual(ms.pushfwdImageSupport(scaledExp, 'u', { kind: 'positive' }), { kind: 'greaterThan', lo: 0.1 });
});
