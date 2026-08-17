'use strict';
// classifyPushfwd (derivations.ts) used to return bare `null` on every
// shape failure, so a malformed `pushfwd` produced NO diagnostic — the
// binding silently vanished and the eventual error named the CONSUMER,
// not the defect site. Two fixes:
//
//   1. A reversed-argument-order spelling (`pushfwd(M, f)` instead of the
//      spec §06 table's `pushfwd(f, M)`) now gets a located diagnostic
//      naming the argument order, instead of silently vanishing.
//   2. A bare BUILTIN-function identifier as `f` (the spec's own example,
//      `pushfwd(exp, mu)`, 06-measure-algebra.md) now classifies — lift.ts
//      desugars it to `fn(exp(_))` before classification, the same shape
//      the explicit-fn spelling already produces.
//
// Oracle for (2): pushfwd(exp, Normal(mu, sigma)) is LogNormal(mu, sigma)
// (closed form, independent of the engine):
//   logpdf(y) = -log(y) - log(sigma) - 0.5*log(2*pi) - (log(y)-mu)^2/(2*sigma^2)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

function logNormalLogpdf(y: number, mu: number, sigma: number): number {
  const ly = Math.log(y);
  return -ly - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI)
    - ((ly - mu) * (ly - mu)) / (2 * sigma * sigma);
}

test('reversed pushfwd(M, f) gets a located diagnostic naming the argument order', () => {
  const src = `
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(mu, exp)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  assert.equal(built.derivations.nu, undefined, 'nu still gets no derivation');
  const diags = (built.diagnostics || []).filter((d: any) => d.severity === 'error');
  const hit = diags.find((d: any) => /nu/.test(d.message) && /pushfwd/.test(d.message)
    && /reversed/i.test(d.message));
  assert.ok(hit, 'a located diagnostic names the reversed argument order');
  assert.ok(hit.loc, 'the diagnostic carries a source location');
  assert.match(hit.message, /\(f, M\)/, 'the message states the correct argument order');
});

test('a legal-but-not-callable f keeps falling through (no false-positive "reversed" diagnostic)', () => {
  // x is bound but not callable-like, mu is a real bound measure — the
  // :2373-class guard that classifyPushfwd already had. Must NOT be
  // reclassified as a reversed-argument malformed pushfwd.
  const src = `
x = 5.0
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(x, mu)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  assert.equal(built.derivations.nu, undefined, 'nu still gets no derivation (unsupported, not malformed)');
  const diags = (built.diagnostics || []);
  const falsePositive = diags.find((d: any) => /reversed/i.test(d.message));
  assert.equal(falsePositive, undefined, 'no reversed-argument-order diagnostic fires for this shape');
});

test('pushfwd(exp, mu) (spec §06\'s own bare-builtin spelling) classifies', () => {
  const src = `
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(exp, mu)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  assert.ok(built.derivations.nu, 'nu classifies to a derivation');
  assert.equal(built.derivations.nu.kind, 'pushfwd');
  assert.equal(built.derivations.nu.from, 'mu');
  assert.deepEqual(built.diagnostics || [], [], 'no diagnostics for the valid spelling');
});

test('pushfwd(exp, Normal(mu, sigma)) scores at the LogNormal oracle (independent closed form)', async () => {
  const mu = 0.0, sigma = 1.0;
  const src = `
m = Normal(mu = ${mu}, sigma = ${sigma})
nu = pushfwd(exp, m)
`;
  for (const y of [0.5, 1.0, 2.0, 3.0]) {
    const { ctx } = ctxFor(src + `lp = logdensityof(nu, ${y})\n`, 1);
    const lp = await ctx.getMeasure('lp');
    const got = lp.samples[0];
    const oracle = logNormalLogpdf(y, mu, sigma);
    assert.ok(Math.abs(got - oracle) < 1e-9,
      `logdensityof(nu, ${y}): got ${got}, LogNormal oracle ${oracle}`);
  }
});

test('pushfwd(exp, Normal(0,1)) samples like LogNormal(0,1) (regression, mirrors bijection-density.test.ts)', async () => {
  const src = `
m = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(exp, m)
`;
  const { ctx } = ctxFor(src, 4096);
  const nu = await ctx.getMeasure('nu');
  let mean = 0;
  for (let i = 0; i < nu.samples.length; i++) {
    assert.ok(nu.samples[i] > 0, 'pushfwd(exp, ...) sample non-positive: ' + nu.samples[i]);
    mean += nu.samples[i];
  }
  mean /= nu.samples.length;
  assert.ok(Math.abs(mean - Math.sqrt(Math.E)) < 0.1, 'sample mean off analytic E[LogNormal(0,1)]=√e: got ' + mean);
});
