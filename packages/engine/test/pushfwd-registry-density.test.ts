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
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

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
