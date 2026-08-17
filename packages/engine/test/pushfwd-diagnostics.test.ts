'use strict';
// classifyPushfwd (derivations.ts) returned bare `null` on every shape
// failure. For a reversed-argument-order `pushfwd(M, f)`, that null was
// NOT silent on `main` — buildDerivations' generic fixed-phase dead-end
// check already emitted "Fixed-phase binding 'nu' produced no value ...
// This is an engine gap — the expression is deterministic but
// unsupported." That message MISATTRIBUTES the defect: it isn't an
// engine gap, it's a swapped-argument typo. Two fixes:
//
//   1. A reversed-argument-order spelling (`pushfwd(M, f)` instead of the
//      spec §06 table's `pushfwd(f, M)`) now gets a located diagnostic
//      naming the argument order, replacing the misleading generic
//      "engine gap" message for exactly this shape. The detection rule
//      (`_mSlotIsFunctionShaped`) must NOT fire on the spec-legal
//      "Uniform kernel extension" (§06: `pushfwd(f, K)` — a kernel is a
//      legal M), or it would delete the honest engine-gap message for
//      code that is merely unimplemented, not malformed, and hand out an
//      invalid rewrite suggestion. `kernelof` is excluded unconditionally
//      (spec §sec:kernelof: its own body is a value, never a measure, so
//      the kernel it reifies is always measure-valued when called);
//      `fn`/`functionof` are excluded only when their body is itself
//      measure-valued (checked via `isMeasureExpr`); `bijection`/`fchain`
//      stay function-shaped unconditionally (neither can return a
//      measure).
//   2. A bare BUILTIN-function identifier as `f` (the spec's own example,
//      `pushfwd(exp, mu)`, 06-measure-algebra.md) now classifies — lift.ts
//      desugars it to `fn(exp(_))` before classification, the same shape
//      the explicit-fn spelling already produces. The diagnostic message
//      reads the ORIGINAL source name off the synthesized binding's
//      `builtinSourceName` tag, not the lift-synthesized anon name.
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

function engineGapMessage(diags: any[]) {
  return diags.find((d: any) => /engine gap/i.test(d.message));
}

function reversedMessage(diags: any[]) {
  return diags.find((d: any) => /reversed/i.test(d.message));
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
  const hit = reversedMessage(diags);
  assert.ok(hit, 'a located diagnostic names the reversed argument order');
  assert.ok(hit.loc, 'the diagnostic carries a source location');
  assert.match(hit.message, /\(f, M\)/, 'the message states the correct argument order');
  // The replaced generic message must not ALSO fire (no double diagnostic).
  assert.equal(engineGapMessage(diags), undefined,
    'the reversed-args diagnostic replaces the generic engine-gap message, not adds to it');
});

test('pushfwd(exp, log): the message names the SOURCE spelling, not the lift-synthesized anon binding', () => {
  const src = `
nu = pushfwd(exp, log)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const hit = reversedMessage(built.diagnostics || []);
  assert.ok(hit, 'a reversed-args diagnostic fires (log can never be a valid M)');
  assert.doesNotMatch(hit.message, /__anon/, 'no lift-synthesized anon name leaks into the message');
  assert.match(hit.message, /\bexp\b/, "the user's own spelling 'exp' appears");
  assert.match(hit.message, /\blog\b/, "the user's own spelling 'log' appears");
});

test('a legal-but-not-callable f keeps falling through (no false-positive "reversed" diagnostic)', () => {
  // x is bound but not callable-like, mu is a real bound measure — the
  // :2373-class guard that classifyPushfwd already had. Must NOT be
  // reclassified as a reversed-argument malformed pushfwd; the generic
  // engine-gap message (pre-existing, unrelated to this fix) still fires.
  const src = `
x = 5.0
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(x, mu)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  assert.equal(built.derivations.nu, undefined, 'nu still gets no derivation (unsupported, not malformed)');
  const diags = built.diagnostics || [];
  assert.equal(reversedMessage(diags), undefined, 'no reversed-argument-order diagnostic fires for this shape');
  assert.ok(engineGapMessage(diags), 'the honest engine-gap message still fires (unaffected by this fix)');
});

// Spec §06 "Uniform kernel extension": "Measure algebra operations accept
// both kernels in general and measures as a (very important) special
// case of kernels. On a kernel, the operation applies to the output
// measure at each input point: pushfwd(f, K) denotes θ ↦
// pushfwd(f, κ(θ))." A kernel in M-position is therefore LEGAL — even
// though the engine does not classify kernel-M pushfwd yet
// (typeinfer.ts:2469, TODO-flatppl-js.md). Each of these three spellings
// must NOT get the "reversed" diagnostic (it would be a false positive
// AND an invalid rewrite suggestion), and must keep the honest
// pre-existing "engine gap ... unsupported" message instead.
const KERNEL_M_SPELLINGS: Array<[string, string]> = [
  ['kernelof in M-position', `
mu = Normal(mu = 0.0, sigma = 1.0)
x ~ mu
y ~ Normal(mu = x, sigma = 1.0)
K = kernelof(y, x = x)
nu = pushfwd(exp, K)
`],
  ['measure-valued functionof in M-position', `
p = elementof(interval(0.0, 10.0))
K = functionof(Normal(mu = p, sigma = 1.0), p = p)
nu = pushfwd(exp, K)
`],
  ['fn-typed measure-valued callable in M-position', `
K = fn(Normal(mu = _, sigma = 1.0))
nu = pushfwd(exp, K)
`],
];

for (const [label, src] of KERNEL_M_SPELLINGS) {
  test(`legal pushfwd(f, K), ${label}: no false-positive "reversed" diagnostic`, () => {
    const { bindings } = processSource(src);
    const built = orchestrator.buildDerivations(bindings);
    assert.equal(built.derivations.nu, undefined, 'nu is still unclassified (kernel-M pushfwd is a tracked gap, not this fix)');
    const diags = built.diagnostics || [];
    assert.equal(reversedMessage(diags), undefined, 'legal kernel-M pushfwd must not be flagged as reversed');
    assert.ok(engineGapMessage(diags), 'the honest engine-gap message is preserved, not deleted');
  });
}

test('legal pushfwd(bijection, K): a bijection-typed f does not affect the M-slot check', () => {
  const src = `
p = elementof(interval(0.0, 10.0))
K = functionof(Normal(mu = p, sigma = 1.0), p = p)
eb = bijection(fn(exp(_)), fn(log(_)), fn(_))
nu = pushfwd(eb, K)
`;
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const diags = built.diagnostics || [];
  assert.equal(reversedMessage(diags), undefined, 'a measure-valued M is legal regardless of what kind of callable f is');
  assert.ok(engineGapMessage(diags), 'the honest engine-gap message is preserved');
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
