'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { buildRegion } = require('../mlfriends.ts');

function lcg(seed: number) { let s = seed >>> 0 || 1; return () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; }; }
// mulberry32: a small, well-known 32-bit PRNG with much better multi-
// dimensional equidistribution than the glibc-style LCG above. The cluster
// metric's falsifiable check below is a stress test for exactly the LCG's
// weak spot: a SKEWED few-way categorical split (picking among a handful of
// unequal-size leaf clusters) sampled via a FIXED per-attempt draw count —
// fixed-stride subsequences of an LCG are themselves low-quality LCGs (a
// classic defect), which measurably biased this test's pick frequencies and
// sample mean under `lcg()` across many seeds even though the underlying
// region math is correct (verified independently: Math.random() reproduces
// a grid-integral ground truth for the union's mean coordinate to within
// 0.01 across ten random cluster configurations, and mulberry32 matches it
// across fifteen seeds — see mlfriends.ts's comment above euclidDist2 for
// the full analysis). The ball metrics' own tests don't need this: picking
// uniformly among K≈hundreds of EQUALLY-weighted centers doesn't expose the
// same fixed-stride correlation the way a 2–12-way skewed split does.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function (): number {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('mlfriends: region contains all its live points', () => {
  const prng = lcg(1);
  const live = Array.from({ length: 60 }, () => Float64Array.from([0.3 + 0.2 * prng(), 0.5 + 0.2 * prng()]));
  const region = buildRegion(live, prng);
  for (const u of live) assert.ok(region.contains(u), `region must contain its own live point ${u}`);
  assert.ok(region.radius > 0);
});

test('mlfriends: sampled points lie inside the cube and the region', () => {
  const prng = lcg(2);
  const live = Array.from({ length: 80 }, () => Float64Array.from([prng(), prng()]));
  const region = buildRegion(live, prng);
  let got = 0;
  for (let i = 0; i < 500 && got < 100; i++) {
    const u = region.sample();
    if (!u) continue;
    got++;
    for (const c of u) assert.ok(c >= 0 && c <= 1, `coord in cube: ${c}`);
    assert.ok(region.contains(u), 'sample must be inside the region');
  }
  assert.ok(got > 0, 'region produced at least some samples');
});

test('mlfriends: uniform-in-region draws are unbiased (overlap correction)', () => {
  // Live points widely spread and asymmetrically clustered: a DENSE cluster
  // near x≈0.3 (tight spread → heavy ball overlap) and a SPARSE cluster near
  // x≈0.75 (wider spread → little overlap). Without a correct overlap
  // correction, region.sample() over-represents the heavy-overlap cluster and
  // its x-mean is pulled toward 0.3. The independent reference — brute-force
  // rejection sampling in the unit square, accepting iff region.contains(u) —
  // is uniform over the union BY CONSTRUCTION (no overlap weighting), so it
  // is a ground truth the sampler's output must match.
  const prng = lcg(3);
  const dense = Array.from({ length: 40 }, () => Float64Array.from([0.3 + 0.02 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const sparse = Array.from({ length: 12 }, () => Float64Array.from([0.75 + 0.15 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const live = dense.concat(sparse);
  const region = buildRegion(live, prng);

  let refN = 0, refSx = 0;
  for (let i = 0; i < 400000 && refN < 4000; i++) {
    const u = Float64Array.from([prng(), prng()]);
    if (region.contains(u)) { refN++; refSx += u[0]; }
  }
  assert.ok(refN >= 4000, `brute-force reference must collect enough accepts (got ${refN})`);
  const refMeanX = refSx / refN;

  let sampN = 0, sampSx = 0;
  for (let i = 0; i < 400000 && sampN < 4000; i++) {
    const u = region.sample();
    if (u) { sampN++; sampSx += u[0]; }
  }
  assert.ok(sampN >= 4000, `region.sample() must collect enough accepts (got ${sampN})`);
  const sampMeanX = sampSx / sampN;

  assert.ok(
    Math.abs(sampMeanX - refMeanX) < 0.03,
    `sample x-mean ${sampMeanX} vs brute-force uniform-union reference x-mean ${refMeanX}`
  );
});

test('mlfriends: identity metric (RadFriends) draws are also unbiased (overlap correction)', () => {
  // Same falsifiable check as the whitened-metric test above, but with
  // opts.metric:'identity' — the RadFriends ball geometry (raw Euclidean
  // radius, no Cholesky/covariance). The overlap correction must still make
  // region.sample() uniform over the union; if it didn't, this would show up
  // as the same heavy-cluster bias the whitened test guards against.
  const prng = lcg(4);
  const dense = Array.from({ length: 40 }, () => Float64Array.from([0.3 + 0.02 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const sparse = Array.from({ length: 12 }, () => Float64Array.from([0.75 + 0.15 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const live = dense.concat(sparse);
  const region = buildRegion(live, prng, { metric: 'identity' });

  let refN = 0, refSx = 0;
  for (let i = 0; i < 400000 && refN < 4000; i++) {
    const u = Float64Array.from([prng(), prng()]);
    if (region.contains(u)) { refN++; refSx += u[0]; }
  }
  assert.ok(refN >= 4000, `brute-force reference must collect enough accepts (got ${refN})`);
  const refMeanX = refSx / refN;

  let sampN = 0, sampSx = 0;
  for (let i = 0; i < 400000 && sampN < 4000; i++) {
    const u = region.sample();
    if (u) { sampN++; sampSx += u[0]; }
  }
  assert.ok(sampN >= 4000, `region.sample() must collect enough accepts (got ${sampN})`);
  const sampMeanX = sampSx / sampN;

  assert.ok(
    Math.abs(sampMeanX - refMeanX) < 0.03,
    `sample x-mean ${sampMeanX} vs brute-force uniform-union reference x-mean ${refMeanX}`
  );
});

test('mlfriends: cluster metric draws are also unbiased (overlap correction)', () => {
  // Same falsifiable check as the whitened/identity tests, but with
  // opts.metric:'cluster' — the per-cluster local-ellipsoid union (recursive
  // 2-means + MultiNest/nestle volume-decrease split test). The dense/sparse
  // asymmetric-cluster fixture is exactly the shape this region is FOR (it
  // should end up fitting one ellipsoid per cluster); the overlap correction
  // must still make region.sample() uniform over the union of ellipsoids —
  // a biased region here would silently corrupt logZ in the sampler.
  const prng = mulberry32(5);
  const dense = Array.from({ length: 40 }, () => Float64Array.from([0.3 + 0.02 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const sparse = Array.from({ length: 12 }, () => Float64Array.from([0.75 + 0.15 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const live = dense.concat(sparse);
  const region = buildRegion(live, prng, { metric: 'cluster' });

  let refN = 0, refSx = 0;
  for (let i = 0; i < 400000 && refN < 4000; i++) {
    const u = Float64Array.from([prng(), prng()]);
    if (region.contains(u)) { refN++; refSx += u[0]; }
  }
  assert.ok(refN >= 4000, `brute-force reference must collect enough accepts (got ${refN})`);
  const refMeanX = refSx / refN;

  let sampN = 0, sampSx = 0;
  for (let i = 0; i < 400000 && sampN < 4000; i++) {
    const u = region.sample();
    if (u) { sampN++; sampSx += u[0]; }
  }
  assert.ok(sampN >= 4000, `region.sample() must collect enough accepts (got ${sampN})`);
  const sampMeanX = sampSx / sampN;

  assert.ok(
    Math.abs(sampMeanX - refMeanX) < 0.03,
    `sample x-mean ${sampMeanX} vs brute-force uniform-union reference x-mean ${refMeanX}`
  );
});

test('mlfriends: cluster metric region contains all its live points', () => {
  const prng = mulberry32(6);
  const live = Array.from({ length: 60 }, () => Float64Array.from([0.3 + 0.2 * prng(), 0.5 + 0.2 * prng()]));
  const region = buildRegion(live, prng, { metric: 'cluster' });
  for (const u of live) assert.ok(region.contains(u), `region must contain its own live point ${u}`);
  assert.ok(region.nCenters >= 1);
});
