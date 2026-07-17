'use strict';
// #310: parity-aware pow domain guard for pushfwd density.
// pow(_, k) is bijective on all of ℝ only for an ODD (positive) integer k;
// for an even or non-integer k it is not injective on ℝ, so the single-branch
// change-of-variables (y^(1/k)) is a silently-wrong density unless the base
// measure's support is provably non-negative. Spec §06/§07: pow (like sqrt) on
// nonnegreals; refuse-don't-mislower off-domain.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('..');

function errs(src: string): string[] {
  return (processSource(src).diagnostics || []).filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}
async function score(src: string, x: number): Promise<number> {
  const { ctx } = ctxFor(src + `\n__score__ = logdensityof(m, ${x})\n`, 1);
  const mm = await ctx.getMeasure('__score__');
  return mm.value ? mm.value.data[0] : mm.samples[0];
}
const H = 'flatppl_compat = "0.1"\n';

test('#310: even pow over a reals base REFUSES (would be silently wrong)', () => {
  const e = errs(H + 'm = pushfwd(fn(pow(_, 2)), Normal(0, 1))\n__score__ = logdensityof(m, 2.25)\n');
  assert.ok(e.length >= 1, 'expected a domain-refusal diagnostic');
  assert.ok(e.some((s) => /pow/.test(s) && /non-negative|nonnegreals/.test(s)), `got ${JSON.stringify(e)}`);
});

test('#310: non-integer pow over a reals base REFUSES', () => {
  const e = errs(H + 'm = pushfwd(fn(pow(_, 0.5)), Normal(0, 1))\n__score__ = logdensityof(m, 0.5)\n');
  assert.ok(e.some((s) => /pow/.test(s)), `expected pow domain refusal; got ${JSON.stringify(e)}`);
});

test('#310: odd-integer pow over a reals base STILL scores (bijective on ℝ)', async () => {
  const v = await score(H + 'm = pushfwd(fn(pow(_, 3)), Normal(0, 1))', 2.0);
  assert.ok(Math.abs(v - (-3.2733494682301787)) < 1e-9, `pow(_,3)@2.0 got ${v}`);
});

test('#310: even pow over a NON-NEGATIVE base scores correctly (single-branch valid)', async () => {
  const v = await score(H + 'm = pushfwd(fn(pow(_, 2)), Exponential(1.0))', 2.25);
  assert.ok(Math.abs(v - (-2.59861228866811)) < 1e-9, `pow(_,2)@2.25 over Exponential got ${v}`);
});
