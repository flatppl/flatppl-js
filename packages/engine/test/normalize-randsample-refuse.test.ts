'use strict';

// wave SMALLS-JS: `normalize` was entirely absent from sampler.ts's
// MEASURE_OP_WALKERS, so `rand(state, normalize(weighted(w, M)))` refused
// even though §07 sec:random makes sampling a normalized measure legal in
// principle. Fixed for the INLINE form in sampler.ts (walkNormalizeRefuse).
//
// This file covers the SECOND, materialiser-side route, mirroring wave-j1's
// weighted fix: a NAMED `nw = normalize(w); rand(state, nw)` never reaches
// sampler.walk at all — `classifyRandTuple` (lift.ts) resolves `nw` to a
// binding ref and classifies the draw as a `randsample` derivation, which
// routes through `matRandSample` (materialiser.ts) instead. Before this
// fix, `matRandSample` had no `normalize` case: it called
// `ctx.getMeasure(d.from)` (dispatching to `matNormalize`, which — for a
// weighted base — legitimately builds a self-normalized-importance-sampling
// empirical measure with non-uniform logWeights) and then discarded those
// logWeights (`empirical.arrayMeasure(data, variateDims, null)`), silently
// handing back a plain draw from the UNNORMALIZED base. Reproduced live
// against pre-fix HEAD: `rand(state, normalize(w))` for
// `w = weighted(2.0, Normal(0, 1))` returned a bare Normal(0,1) draw with no
// throw — the same weight-dropping defect class wave-j1's `walkWeightedRefuse`
// / `matRandSample` weighted-check closes, one `normalize(...)` wrapper away.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx } = require('./_materialise-helpers.ts');

const RNGSEED_SRC = `
rngseed = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
rstate = rnginit(rngseed)
`;

test('randsample: named normalize(weighted(...)) binding refuses rather than dropping the weight', async () => {
  const src = RNGSEED_SRC + `
w = weighted(2.0, Normal(0, 1))
nw = normalize(w)
s, rstate2 = rand(rstate, nw)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample',
    'named-binding rand is a randsample derivation, not evaluate');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'normalize' cannot be sampled.*requires reweighting/,
  );
});

test('randsample: named normalize(logweighted(...)) binding refuses too (same wall)', async () => {
  const src = RNGSEED_SRC + `
w = logweighted(0.5, Normal(0, 1))
nw = normalize(w)
s, rstate2 = rand(rstate, nw)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'normalize' cannot be sampled.*requires reweighting/,
  );
});

test('randsample: normalize(alias(weighted(...))) still refuses (alias/normalize chain peeled)', async () => {
  const src = RNGSEED_SRC + `
w = weighted(2.0, Normal(0, 1))
w2 = w
nw = normalize(w2)
s, rstate2 = rand(rstate, nw)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 8 });
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'normalize' cannot be sampled.*requires reweighting/,
  );
});

test('randsample: named normalize(m) identity binding samples — no regression', async () => {
  const src = RNGSEED_SRC + `
m = Normal(0, 1)
nm = normalize(m)
s, rstate2 = rand(rstate, iid(nm, 5000))
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 5000 });
  assert.equal(built.derivations.s.kind, 'randsample');
  const measure = await ctx.getMeasure('s');
  const data: Float64Array = measure.value.data;
  const n = data.length;
  assert.equal(n, 5000);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += data[i];
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < n; i++) sq += (data[i] - mean) * (data[i] - mean);
  const varr = sq / n;
  assert.ok(Math.abs(mean) < 0.1, `mean ${mean} should be ~0 (Standard Normal)`);
  assert.ok(Math.abs(varr - 1) < 0.15, `var ${varr} should be ~1 (Standard Normal)`);
});

// =====================================================================
// Density path unchanged (this wave only touches sampler.ts / the
// matRandSample rand-gate, not mat-density.ts / matNormalize). One pin
// each for the identity shape and the reweighted shape, oracle-derived
// closed-form: normalize(Normal(0,1)) density is exact (Z=1, no shift);
// normalize(weighted(w, Normal(0,1))) absorbs the constant weight exactly
// (Z=w, shift is -log(w) which then cancels the weight's own log(w) shift).
// =====================================================================

function normalLogpdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * z * z;
}

test('density: normalize(Normal(0,1)) identity — logdensityof unchanged (Z=1)', async () => {
  const src = `
m = normalize(Normal(0, 1))
lp = logdensityof(m, 0.3)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 4 });
  const measure = await ctx.getMeasure('lp');
  const expected = normalLogpdf(0.3, 0, 1);
  assert.ok(Math.abs(measure.samples[0] - expected) < 1e-9,
    `normalize(Normal(0,1)) logp(0.3): got ${measure.samples[0]}, expected ${expected}`);
});

test('density: normalize(weighted(3, Normal(0,1))) — logdensityof unchanged (constant weight absorbed, Z=3)', async () => {
  const src = `
m = normalize(weighted(3.0, Normal(0, 1)))
lp = logdensityof(m, 0.3)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 4 });
  const measure = await ctx.getMeasure('lp');
  // A constant weight w scales the base's mass by w (Z = w · 1 = 3), so
  // normalize divides it straight back out: the constant cancels exactly
  // and the normalized density equals the base Normal(0,1) density.
  const expected = normalLogpdf(0.3, 0, 1);
  assert.ok(Math.abs(measure.samples[0] - expected) < 1e-9,
    `normalize(weighted(3, Normal)) logp(0.3): got ${measure.samples[0]}, expected ${expected}`);
});
