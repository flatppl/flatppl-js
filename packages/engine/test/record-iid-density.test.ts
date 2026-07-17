'use strict';
// G1: iid over a record-valued measure scored against a table observation.
// Spec: 06-measure-algebra.md — iid(M, size) is the product measure over
// arrays of M's variate; logdensityof(iid(M,n), x) = Σ_i logdensityof(M, x_i).
// 03-value-types.md — an array of records is a table (each row is a record).
//
// Expected values are INDEPENDENTLY derived (hand closed-form + scipy, recorded
// per case) — never pinned to engine output.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

// The models below carry no free parameters (the data is fully given), so the
// measure's density is scored directly with logdensityof(iid(M,n), table).
async function score(src: string): Promise<number> {
  const { ctx, diagnostics } = ctxFor('flatppl_compat = "0.1"\n' + src, 1);
  const errs = (diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length) throw new Error('analysis: ' + errs.map((d: any) => d.message).join(' | '));
  const mm = await ctx.getMeasure('__score__');
  return mm.value ? mm.value.data[0] : mm.samples[0];
}

test('#231/G1: iid over a 2-field record variate vs closed form', async () => {
  // Σ logpdf(N(0,1), v) over {0.1, 0.2, 0.3, 0.4}
  //   = 4·(−0.9189385332046727) − 0.15 = −3.8257541328186907
  // scipy: sum(norm.logpdf(v,0,1) for v in [0.1,0.2,0.3,0.4]) = -3.8257541328186906
  const v = await score(`
gxy = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 0.0, sigma = 1.0))
data = table(x = [0.1, 0.2], y = [0.3, 0.4])
d2 = iid(gxy, 2)
__score__ = logdensityof(d2, data)
`);
  assert.ok(Math.abs(v - (-3.8257541328186907)) < 1e-12, `got ${v}, want -3.8257541328186907`);
});

test('#231/G1: iid over a 3-field record variate vs closed form', async () => {
  // scipy: sum(norm.logpdf(v,0,1) for v in [0.1..0.6]) = -5.968631199228036
  const v = await score(`
g3 = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 0.0, sigma = 1.0), z = Normal(mu = 0.0, sigma = 1.0))
data = table(x = [0.1, 0.2], y = [0.3, 0.4], z = [0.5, 0.6])
d3 = iid(g3, 2)
__score__ = logdensityof(d3, data)
`);
  assert.ok(Math.abs(v - (-5.968631199228036)) < 1e-12, `got ${v}, want -5.968631199228036`);
});

test('#231/G1: row-count mismatch fails loud (3 copies, 2 rows)', async () => {
  await assert.rejects(async () => score(`
gxy = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 0.0, sigma = 1.0))
data = table(x = [0.1, 0.2], y = [0.3, 0.4])
d2 = iid(gxy, 3)
__score__ = logdensityof(d2, data)
`));
});
