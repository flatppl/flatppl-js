'use strict';

// Value-level `x in S` membership (spec §07 "Membership"): `x in S` returns a
// boolean. The deterministic floor reaches it via the determiniser's truncate
// lowering `truncate(M, S) → ifelse(in(v, S), density, neg(inf))`, so the
// engine must evaluate `in` (and the `interval(lo,hi)` / named-set operands) as
// a VALUE, not merely as a measure-layer set descriptor. Regression for the
// gap where `in`/`interval` had no evaluable-op registration and any binding
// containing `x in S` yielded "no derivation".

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function scoreValue(src: string, name: string): Promise<any> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure(name);
  if (m && m.value && m.value.data) return m.value.data[0];
  if (m && m.samples && m.samples.length) return m.samples[0];
  throw new Error(`no value for binding '${name}'`);
}

test('`x in interval(lo,hi)` gates ifelse — closed interval (spec §03/§07)', async () => {
  // in-support → then-branch; out-of-support → else-branch.
  assert.equal(await scoreValue('lp = ifelse(0.5 in interval(-1.0, 2.0), 10.0, 20.0)', 'lp'), 10.0);
  assert.equal(await scoreValue('lp = ifelse(2.5 in interval(-1.0, 2.0), 10.0, 20.0)', 'lp'), 20.0);
  // `interval` is the CLOSED interval [lo, hi] — endpoints are members.
  assert.equal(await scoreValue('lp = ifelse(2.0 in interval(-1.0, 2.0), 10.0, 20.0)', 'lp'), 10.0);
  assert.equal(await scoreValue('lp = ifelse(-1.0 in interval(-1.0, 2.0), 10.0, 20.0)', 'lp'), 10.0);
});

test('`x in <named set>` membership via ifelse', async () => {
  assert.equal(await scoreValue('lp = ifelse(1.5 in posreals, 1.0, 0.0)', 'lp'), 1.0);
  assert.equal(await scoreValue('lp = ifelse(0.0 in posreals, 1.0, 0.0)', 'lp'), 0.0);      // strict > 0
  assert.equal(await scoreValue('lp = ifelse(0.0 in nonnegreals, 1.0, 0.0)', 'lp'), 1.0);   // >= 0
});

test('truncate in-support density lowers through the `in` gate', async () => {
  // The determiniser's truncate density: ifelse(in(v, S), logdensity, neg(inf)).
  // In-support, this returns the Normal log-density; verify against closed form.
  const v = await scoreValue(
    'lp = ifelse(0.5 in interval(-1.0, 2.0), '
    + 'builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), 0.5), neg(inf))',
    'lp',
  );
  const expect = -0.5 * Math.log(2 * Math.PI) - 0.5 * 0.25; // log N(0.5; 0, 1)
  assert.ok(Math.abs(v - expect) < 1e-12, `${v} vs ${expect}`);
});
