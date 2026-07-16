'use strict';
// Spec §06 "Known-bijection registry" (case 1): logdensityof(pushfwd(f, M), y)
// must evaluate analytically WITHOUT an explicit bijection(...) annotation for
// the built-in registry bijections. The synthesized inverse is produced by
// bijection-registry.invertExpr and attached in derivations.ts (case 'pushfwd').
//
// Regression for #260 (a): the synthesized inverse's substituted output ref was
// minted with ns:'self' instead of ns:'%local', violating the CLM subset
// invariant (ir-walk.ts: formal-parameter refs are '%local'); the logdensityof
// derivation was then silently cascade-pruned ("no derivation for '__score__'"),
// breaking EVERY already-registered op (exp/log/affine/pow).
//
// #260 (b): the remaining §07 elementary unary functions (sqrt, log10,
// log1p/expm1, logit/invlogit, probit/invprobit, atan, sinh/asinh, tanh) were
// added to bijection-registry.ts's ELEMENTARY_BIJECTIONS, mirroring the
// oracle-verified Rust determiniser (flatppl-rust crates/determinizer
// src/invert.rs `bare_bijection`). Every oracle value below is independently
// derived via scipy (base logpdf minus the forward log|f'| at the inverted
// point) — see the per-test comment for the exact base/point.
//
// #260 (c): a domain-restricted forward (log/log10/sqrt/log1p/logit/probit)
// additionally requires the base measure's support to lie within that
// domain; where the base's support is PROVABLY outside it, `processSource`
// now raises a static error diagnostic (typeinfer.ts
// `checkPushfwdDomainContracts`) rather than silently scoring a
// sub-probability measure.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('..');

async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  return m.value ? m.value.data[0] : m.samples[0];
}

// Oracles: scipy, independently derived.
test('§06 registry: annotation-free pushfwd(fn(exp(_)), Normal) ≡ LogNormal', async () => {
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(exp(_)), Normal(0, 1))
__score__ = logdensityof(m, 2.0)
`);
  assert.ok(Math.abs(v - (-1.8523122207237186)) < 1e-9, `exp@2.0 got ${v}`);
});

test('§06 registry: annotation-free affine pushfwd(fn(2*_ + 1), Normal)', async () => {
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(2 * _ + 1), Normal(0, 1))
__score__ = logdensityof(m, 3.0)
`);
  assert.ok(Math.abs(v - (-2.112085713764618)) < 1e-9, `affine@3.0 got ${v}`);
});

test('§06 registry: annotation-free pow pushfwd(fn(pow(_, 3)), Normal)', async () => {
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(pow(_, 3)), Normal(0, 1))
__score__ = logdensityof(m, 2.0)
`);
  assert.ok(Math.abs(v - (-3.2733494682301787)) < 1e-9, `pow(_,3)@2.0 got ${v}`);
});

// ---------------------------------------------------------------------
// #260 (b): registry extension — the remaining §07 elementary unary
// bijections. Each oracle: scipy `base.logpdf(x0) - log|f'(x0)|`, x0 =
// f_inv(y), independently derived (not copied from the engine).

test('§06 registry: sqrt pushfwd(fn(sqrt(_)), Exponential(1))', async () => {
  // oracle: x0 = 1.5**2 = 2.25; expon(scale=1).logpdf(2.25) = -2.25;
  // log|f'(x0)| = log(0.5) - 0.5*log(2.25) = -1.0986122886681096.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(sqrt(_)), Exponential(1))
__score__ = logdensityof(m, 1.5)
`);
  assert.ok(Math.abs(v - (-1.1513877113318904)) < 1e-9, `sqrt@1.5 got ${v}`);
});

test('§06 registry: log10 pushfwd(fn(log10(_)), Exponential(2))', async () => {
  // oracle: x0 = 10**0.3; expon(scale=0.5).logpdf(x0) = -3.2973774493778136;
  // log|f'(x0)| = -(log(x0) + log(log(10))) = -1.5248079731461694.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(log10(_)), Exponential(2))
__score__ = logdensityof(m, 0.3)
`);
  assert.ok(Math.abs(v - (-1.7725694762316442)) < 1e-9, `log10@0.3 got ${v}`);
});

test('§06 registry: log1p pushfwd(fn(log1p(_)), Exponential(1))', async () => {
  // oracle: x0 = expm1(0.5) = 0.6487212707001282; expon(scale=1).logpdf(x0)
  // = -0.6487212707001282; log|f'(x0)| = -log1p(x0) = -0.5.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(log1p(_)), Exponential(1))
__score__ = logdensityof(m, 0.5)
`);
  assert.ok(Math.abs(v - (-0.1487212707001282)) < 1e-9, `log1p@0.5 got ${v}`);
});

test('§06 registry: expm1 pushfwd(fn(expm1(_)), Normal(0,1))', async () => {
  // oracle: x0 = log1p(0.4) = 0.33647223662121295; norm(0,1).logpdf(x0)
  // = -0.9755453162131135; log|f'(x0)| = x0 (d/dx(e^x-1)=e^x, log(e^x0)=x0).
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(expm1(_)), Normal(0, 1))
__score__ = logdensityof(m, 0.4)
`);
  assert.ok(Math.abs(v - (-1.3120175528343263)) < 1e-9, `expm1@0.4 got ${v}`);
});

test('§06 registry: logit pushfwd(fn(logit(_)), Beta(2,3))', async () => {
  // oracle: x0 = invlogit(0.7) = 0.6681877721681662; beta(2,3).logpdf(x0)
  // = -0.12465149686837362; log|f'(x0)| = -(log(x0)+log(1-x0)) = 1.506372097770916.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(logit(_)), Beta(2, 3))
__score__ = logdensityof(m, 0.7)
`);
  assert.ok(Math.abs(v - (-1.6310235946392897)) < 1e-9, `logit@0.7 got ${v}`);
});

test('§06 registry: invlogit pushfwd(fn(invlogit(_)), Normal(0,1))', async () => {
  // oracle: x0 = logit(0.6) = 0.4054651081081642; norm(0,1).logpdf(x0)
  // = -1.0011395101512552; s=invlogit(x0)=0.6; log|f'(x0)|=log(s)+log(1-s)
  // = -1.4271163556401456.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(invlogit(_)), Normal(0, 1))
__score__ = logdensityof(m, 0.6)
`);
  assert.ok(Math.abs(v - 0.42597684548889037) < 1e-9, `invlogit@0.6 got ${v}`);
});

test('§06 registry: probit pushfwd(fn(probit(_)), Beta(2,3))', async () => {
  // oracle: x0 = Phi(0.65) = 0.7421538891941353; beta(2,3).logpdf(x0)
  // = -0.5240766932231051; log|f'(x0)| = 0.5*log(2*pi) + 0.5*probit(x0)^2
  // = 1.1301885332046726.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(probit(_)), Beta(2, 3))
__score__ = logdensityof(m, 0.65)
`);
  assert.ok(Math.abs(v - (-1.6542652264277777)) < 1e-9, `probit@0.65 got ${v}`);
});

test('§06 registry: invprobit pushfwd(fn(invprobit(_)), Normal(2,3))', async () => {
  // oracle: x0 = probit(0.55) = 0.12566134685507416; norm(2,3).logpdf(x0)
  // = -2.2127255655768456; log|f'(x0)| = -(0.5*log(2*pi)+0.5*x0^2)
  // = -0.9268339202513883.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(invprobit(_)), Normal(2, 3))
__score__ = logdensityof(m, 0.55)
`);
  assert.ok(Math.abs(v - (-1.2858916453254574)) < 1e-9, `invprobit@0.55 got ${v}`);
});

test('§06 registry: atan pushfwd(fn(atan(_)), Normal(0,1))', async () => {
  // oracle: x0 = tan(0.9) = 1.260158217550339; norm(0,1).logpdf(x0)
  // = -1.7129378998344964; log|f'(x0)| = -log1p(x0^2) = -0.9508848871716284.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(atan(_)), Normal(0, 1))
__score__ = logdensityof(m, 0.9)
`);
  assert.ok(Math.abs(v - (-0.762053012662868)) < 1e-9, `atan@0.9 got ${v}`);
});

test('§06 registry: sinh pushfwd(fn(sinh(_)), Normal(0,1))', async () => {
  // oracle: x0 = asinh(1.2) = 1.015973134179692; norm(0,1).logpdf(x0)
  // = -1.435039237892126; log|f'(x0)| = log(cosh(x0)) = 0.44599901965255523.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(sinh(_)), Normal(0, 1))
__score__ = logdensityof(m, 1.2)
`);
  assert.ok(Math.abs(v - (-1.8810382575446811)) < 1e-9, `sinh@1.2 got ${v}`);
});

test('§06 registry: asinh pushfwd(fn(asinh(_)), Normal(0,1))', async () => {
  // oracle: x0 = sinh(0.8) = 0.888105982187623; norm(0,1).logpdf(x0)
  // = -1.3133046510033939; log|f'(x0)| = -0.5*log1p(x0^2) = -0.29075356032839356.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(asinh(_)), Normal(0, 1))
__score__ = logdensityof(m, 0.8)
`);
  assert.ok(Math.abs(v - (-1.0225510906750004)) < 1e-9, `asinh@0.8 got ${v}`);
});

test('§06 registry: tanh pushfwd(fn(tanh(_)), Normal(0,1))', async () => {
  // oracle: x0 = atanh(0.3) = 0.30951960420311175; norm(0,1).logpdf(x0)
  // = -0.9668397258976982; log|f'(x0)| = log(1-tanh(x0)^2) = -0.09431067947124142.
  const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(tanh(_)), Normal(0, 1))
__score__ = logdensityof(m, 0.3)
`);
  assert.ok(Math.abs(v - (-0.8725290464264568)) < 1e-9, `tanh@0.3 got ${v}`);
});

// ---------------------------------------------------------------------
// #260 (c): domain-restriction refusal. `log` requires a posreals-ish
// base support (spec §06); Normal(0,1)'s support is REALS, which is
// PROVABLY outside that domain — processSource must raise an error
// diagnostic rather than silently score a finite-but-wrong density.

test('§06 domain guard: pushfwd(fn(log(_)), Normal(0,1)) REFUSES (off-domain)', () => {
  const proc = processSource(`
flatppl_compat = "0.1"
m = pushfwd(fn(log(_)), Normal(0, 1))
__score__ = logdensityof(m, 2.0)
`);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errs.length > 0, 'expected an error diagnostic for off-domain log pushfwd');
  assert.ok(
    errs.some((d: any) => /pushfwd/.test(d.message) && /'log'/.test(d.message) && /support/.test(d.message)),
    `expected a pushfwd-domain-specific message, got: ${JSON.stringify(proc.diagnostics)}`,
  );
  // Not the generic case-3 "requires a bijection annotation" message —
  // this must be the domain-specific refusal, not the fallback.
  assert.ok(
    !errs.some((d: any) => /requires a bijection annotation/.test(d.message)),
    'expected the domain-specific message, not the generic "requires a bijection annotation" fallback',
  );
});

test('§06 domain guard: pushfwd(fn(sqrt(_)), Exponential(1)) does NOT refuse (in-domain)', () => {
  const proc = processSource(`
flatppl_compat = "0.1"
m = pushfwd(fn(sqrt(_)), Exponential(1))
__score__ = logdensityof(m, 1.5)
`);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], `expected no error diagnostics, got: ${JSON.stringify(errs)}`);
});
