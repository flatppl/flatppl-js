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
  // Live points clustered so balls overlap heavily on one side; the overlap
  // correction must NOT over-represent the overlap region. Check the sample mean
  // matches the region's geometric mean within tolerance over many draws.
  const prng = lcg(3);
  const live = Array.from({ length: 100 }, () => Float64Array.from([0.5 + 0.05 * (prng() - 0.5), 0.5 + 0.3 * (prng() - 0.5)]));
  const region = buildRegion(live, prng);
  let n = 0, sx = 0;
  for (let i = 0; i < 5000 && n < 2000; i++) { const u = region.sample(); if (u) { n++; sx += u[0]; } }
  assert.ok(n > 100);
  assert.ok(Math.abs(sx / n - 0.5) < 0.05, `x-mean ${sx / n} vs 0.5 (overlap bias check)`);
});
