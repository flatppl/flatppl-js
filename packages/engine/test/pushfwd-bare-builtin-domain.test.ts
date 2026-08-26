'use strict';
// Spec §06 "Engine contract for `pushfwd` density evaluation", case 1
// (known-bijection registry): "A domain-restricted forward — `log`/`log10` on
// `posreals`, `sqrt` (and `pow`) on `nonnegreals`, `log1p` on
// `interval(-1, inf)`, `logit`/`probit` on `interval(0, 1)` — additionally
// requires the base measure's support to lie within that domain; where it does
// not, density evaluation is refused rather than yielding a silently
// sub-probability measure."
//
// The guard (typeinfer.ts `checkPushfwdDomainContracts`) only saw the explicit
// `fn(op(_))` spelling: the BARE-BUILTIN spelling `pushfwd(log, M)` reaches PIR
// as a `ref` to the builtin, not as a `functionof` reification, so
// `_resolveForwardReif` returned null and the whole pushfwd went unguarded.
// `pushfwd(log, Normal(0, 1))` then scored a finite density.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('..');

async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  return m.value ? m.value.data[0] : m.samples[0];
}
function errorsOf(src: string): any[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

// ---------------------------------------------------------------------
// Refusals: a bare domain-restricted builtin over a provably off-domain base.

test('§06 domain guard: bare pushfwd(log, Normal(0,1)) REFUSES', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
m = pushfwd(log, Normal(0, 1))
__score__ = logdensityof(m, 0.5)
`);
  assert.ok(
    errs.some((d: any) => /pushfwd/.test(d.message) && /'log'/.test(d.message) && /support/.test(d.message)),
    `expected the pushfwd domain refusal, got: ${JSON.stringify(errs)}`,
  );
  assert.ok(
    !errs.some((d: any) => /requires a bijection annotation/.test(d.message)),
    'expected the domain-specific message, not the generic case-3 fallback',
  );
  const d = errs.find((e: any) => /pushfwd/.test(e.message));
  assert.ok(d.loc && d.loc.start && typeof d.loc.start.line === 'number',
    `refusal must be located, got loc=${JSON.stringify(d.loc)}`);
});

test('§06 domain guard: bare sqrt / logit / log10 / probit over a real base REFUSE', () => {
  for (const op of ['sqrt', 'logit', 'log10', 'probit']) {
    const errs = errorsOf(`
flatppl_compat = "0.1"
m = pushfwd(${op}, Normal(0, 1))
__score__ = logdensityof(m, 0.5)
`);
    assert.ok(
      errs.some((e: any) => /pushfwd/.test(e.message) && new RegExp(`'${op}'`).test(e.message)),
      `expected a domain refusal for bare ${op}, got: ${JSON.stringify(errs)}`,
    );
  }
});

test('§06 domain guard: an alias of a bare builtin REFUSES too', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
f = log
m = pushfwd(f, Normal(0, 1))
__score__ = logdensityof(m, 0.5)
`);
  assert.ok(
    errs.some((d: any) => /pushfwd/.test(d.message) && /'log'/.test(d.message)),
    `expected the domain refusal through the alias, got: ${JSON.stringify(errs)}`,
  );
});

// ---------------------------------------------------------------------
// Non-refusals. The guard must not fire on an unrestricted forward, nor on a
// restricted forward whose base support IS within the domain.

test('§06 domain guard: bare log over Exponential (nonnegreals) does NOT refuse', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
m = pushfwd(log, Exponential(1))
__score__ = logdensityof(m, -0.5)
`);
  assert.deepEqual(errs, [], `in-domain base must not refuse, got: ${JSON.stringify(errs)}`);
});

test('§06 domain guard: bare logit over Beta (unit interval) does NOT refuse', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
m = pushfwd(logit, Beta(2, 3))
__score__ = logdensityof(m, 0.5)
`);
  assert.deepEqual(errs, [], `in-domain base must not refuse, got: ${JSON.stringify(errs)}`);
});

// A `bijection(...)`-annotated forward is the USER's asserted inverse and
// volume element — spec §06 `bijection`: "FlatPPL implementations are not
// required to verify this." The domain guard covers the known-bijection
// registry (case 1), so an annotation must not be second-guessed here.
test('§06: a bijection-annotated log forward is NOT domain-guarded', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
b = bijection(log, exp, fn(neg(log(abs(_)))))
m = pushfwd(b, Normal(0, 1))
__score__ = logdensityof(m, 0.5)
`);
  assert.ok(
    !errs.some((d: any) => /pushfwd: forward function uses/.test(d.message)),
    `an explicit annotation must not trip the registry domain guard, got: ${JSON.stringify(errs)}`,
  );
});

// LogNormal exactness. `exp` is unrestricted (domain ℝ) so the guard must leave
// `pushfwd(exp, Normal(0, 1))` = LogNormal(0, 1) alone, in BOTH spellings.
// Oracle: scipy.stats.lognorm.logpdf(y, s=1) — equivalently the closed form
// -log y - ½log(2π) - (log y)²/2.
const LOGNORMAL_01: Array<[number, number]> = [
  [0.5, -0.4660178596038281],
  [2.0, -1.8523122207237188],
  [3.0, -2.6210253022790733],
];

test('§06: bare pushfwd(exp, Normal(0,1)) scores LogNormal(0,1) exactly', async () => {
  for (const [y, want] of LOGNORMAL_01) {
    const src = `
flatppl_compat = "0.1"
m = pushfwd(exp, Normal(0, 1))
__score__ = logdensityof(m, ${y})
`;
    assert.deepEqual(errorsOf(src), [], `exp is unrestricted; must not refuse at y=${y}`);
    const v = await score(src);
    assert.ok(Math.abs(v - want) < 1e-9, `bare exp@${y} got ${v}, want ${want}`);
  }
});

test('§06: fn(exp(_)) spelling scores the same LogNormal values', async () => {
  for (const [y, want] of LOGNORMAL_01) {
    const v = await score(`
flatppl_compat = "0.1"
m = pushfwd(fn(exp(_)), Normal(0, 1))
__score__ = logdensityof(m, ${y})
`);
    assert.ok(Math.abs(v - want) < 1e-9, `fn exp@${y} got ${v}, want ${want}`);
  }
});

// Affine control: unrestricted, and the guard must not perturb its density.
// Oracle: norm.logpdf((3 - 1) / 2) - log 2.
test('§06: affine pushfwd(fn(2*_ + 1), Normal(0,1)) still scores', async () => {
  const src = `
flatppl_compat = "0.1"
m = pushfwd(fn(2 * _ + 1), Normal(0, 1))
__score__ = logdensityof(m, 3.0)
`;
  assert.deepEqual(errorsOf(src), [], 'affine is unrestricted; must not refuse');
  const v = await score(src);
  assert.ok(Math.abs(v - (-2.112085713764618)) < 1e-9, `affine@3.0 got ${v}`);
});
