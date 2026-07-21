'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { deriveRegionOpts } = require('../mat-density.ts');

// deriveRegionOpts maps the viewer's single-field regionMetric contract
// ('off' | 'whitened' | 'identity' | 'cluster' — see types.d.ts
// Ctx.inferenceOpts and the region selector in render-controls.ts) onto
// runNested's {useRegion, region} pair. 'off' (region-free) is the shipped
// default — see mlfriends.ts's header comment for why the other metrics
// stay opt-in.

test('regionMetric "off" (the default) is region-free: useRegion false, region undefined', () => {
  assert.deepEqual(deriveRegionOpts({ regionMetric: 'off' }), { useRegion: false, region: undefined });
});

test('an inferenceOpts object with no regionMetric at all defaults to region-free too', () => {
  assert.deepEqual(deriveRegionOpts({}), { useRegion: false, region: undefined });
});

test('regionMetric "whitened" turns region on with metric:"whitened" (global MLFriends ellipsoid)', () => {
  assert.deepEqual(deriveRegionOpts({ regionMetric: 'whitened' }), { useRegion: true, region: { metric: 'whitened' } });
});

test('regionMetric "identity" turns region on with metric:"identity" (RadFriends balls)', () => {
  assert.deepEqual(deriveRegionOpts({ regionMetric: 'identity' }), { useRegion: true, region: { metric: 'identity' } });
});

test('regionMetric "cluster" turns region on with metric:"cluster" (per-cluster local ellipsoids)', () => {
  assert.deepEqual(deriveRegionOpts({ regionMetric: 'cluster' }), { useRegion: true, region: { metric: 'cluster' } });
});

// Back-compat: direct runNested callers / saved viewer state that predate
// the regionMetric field still work via the raw useRegion/region passthrough.
test('back-compat: regionMetric undefined falls back to raw useRegion/region', () => {
  assert.deepEqual(
    deriveRegionOpts({ useRegion: true, region: { metric: 'identity' } }),
    { useRegion: true, region: { metric: 'identity' } },
  );
});

test('back-compat: regionMetric undefined + useRegion unset is region-free', () => {
  assert.deepEqual(deriveRegionOpts({ region: { metric: 'whitened' } }), { useRegion: false, region: { metric: 'whitened' } });
});

test('a null/undefined opts object is region-free (defensive default)', () => {
  assert.deepEqual(deriveRegionOpts(null), { useRegion: false, region: undefined });
  assert.deepEqual(deriveRegionOpts(undefined), { useRegion: false, region: undefined });
});
