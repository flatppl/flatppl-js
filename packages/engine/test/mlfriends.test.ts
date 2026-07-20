'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { buildRegion } = require('../mlfriends.ts');

function lcg(seed: number) { let s = seed >>> 0 || 1; return () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; }; }

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
