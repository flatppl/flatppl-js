'use strict';

// =====================================================================
// discrete-truncate-normalizer.test.ts — Buffy #73 (discrete-truncate)
// =====================================================================
//
// `normalize(truncate(<discrete base>, S))`: the truncate normalizer
// Z = base-mass-over-S is computed by continuous machinery on every scoring
// path — mat-density.resolveNormalizeMasses materialises a worker CDF-diff
// (F(hi)−F(lo), drops the lower endpoint for a discrete base);
// mat-density.truncateLogMass quadratures the base log-density (pmf-as-
// density); the broadcast-kernel walkNormalize helper fell through to Z=1.
// Each is silently wrong for a DISCRETE base (the true mass is Σ pmf over S).
// §06 defines no discrete-truncate normalizer — fail loud rather than score
// silently wrong. (Sampling of a truncated discrete leaf is untouched; only
// the normalizer fails loud.)

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function scoreThrows(src: string): Promise<{ threw: boolean; value: number | null; msg: string }> {
  try {
    const { ctx } = ctxFor(src, 1);
    const m = await ctx.getMeasure('__score__');
    const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
    return { threw: false, value: s ? s[0] : null, msg: '' };
  } catch (e: any) {
    return { threw: true, value: null, msg: String(e && e.message) };
  }
}

test('discrete-base normalize(truncate(Poisson, interval)) fails loud (was silently mis-normalised to a fixed wrong Z)', async () => {
  const src = `m = normalize(truncate(Poisson(3.0), interval(0.0, 5.0)))
__score__ = logdensityof(m, 2.0)
`;
  const r = await scoreThrows(src);
  assert.ok(r.threw,
    `scoring a discrete-base truncate normalizer must throw (was silently mis-scored to ${r.value}); got no throw`);
  assert.ok(/discret|pmf|probability mass|not implemented|not supported/i.test(r.msg),
    `the error should name the discrete-truncate limitation; got: ${r.msg}`);
});

test('discrete-base truncate normalizer fails loud through a kernelof/likelihood surface', async () => {
  // A discrete normalize(truncate(...)) inside a kernelof body reaches the same
  // resolveNormalizeMasses choke as the direct case (it carries a massFrom ref,
  // so resolveTruncateNormalizers — the continuous-only inline path — skips it).
  // Exercises the guard through the likelihood-scoring plumbing.
  const src = `mu ~ Poisson(3.0)
k = kernelof(record(y = normalize(truncate(Poisson(3.0), interval(0.0, 5.0)))), mu = mu)
L = likelihoodof(k, record(y = 2.0))
__score__ = logdensityof(L, record(mu = 3.0))
`;
  const r = await scoreThrows(src);
  assert.ok(r.threw && /discret|pmf|probability mass/i.test(r.msg),
    `kernelof/likelihood discrete truncate must fail loud; threw=${r.threw} value=${r.value} msg=${r.msg}`);
});

test('discrete-base truncate normalizer fails loud on the per-row broadcast-kernel path (walkNormalize)', async () => {
  // Per-row normalize(truncate(discrete)) broadcast kernel → walkNormalize's
  // truncate-normalizer helper (the third distinct path).
  const src = `w = [3.0, 4.0]
energy = p -> normalize(truncate(Poisson(p), interval(0.0, 5.0)))
E ~ energy.(w)
__score__ = logdensityof(E, [2.0, 3.0])
`;
  const r = await scoreThrows(src);
  assert.ok(r.threw && /discret|pmf|probability mass/i.test(r.msg),
    `broadcast-kernel discrete truncate must fail loud; threw=${r.threw} value=${r.value} msg=${r.msg}`);
});

test('guard: a CONTINUOUS-base normalize(truncate(Normal, interval)) still scores (fail-loud is discrete-only)', async () => {
  const src = `m = normalize(truncate(Normal(0.0, 1.0), interval(-1.0, 2.0)))
__score__ = logdensityof(m, 0.5)
`;
  const r = await scoreThrows(src);
  assert.ok(!r.threw && Number.isFinite(r.value),
    `continuous truncate must still score; threw=${r.threw} value=${r.value} msg=${r.msg}`);
});
