// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { test } from 'node:test';
import assert from 'node:assert/strict';

const { densityContours } = await import('./contour2d.ts');

// Deterministic PRNG (mulberry32) + Box-Muller so the blob is fixed across
// runs — no Math.random, so the contour geometry is reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussianBlob(n, seed) {
  const rnd = mulberry32(seed);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
    const r = Math.sqrt(-2 * Math.log(u1));
    xs[i] = r * Math.cos(2 * Math.PI * u2);
    ys[i] = r * Math.sin(2 * Math.PI * u2);
  }
  return { xs, ys };
}

// Axis-aligned bounding-box area of a set of segments — a cheap proxy for
// "which contour encloses more area" without a polygon-area walk.
function bboxArea(segments) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [[x0, y0], [x1, y1]] of segments) {
    xmin = Math.min(xmin, x0, x1); xmax = Math.max(xmax, x0, x1);
    ymin = Math.min(ymin, y0, y1); ymax = Math.max(ymax, y0, y1);
  }
  if (!(xmax > xmin) || !(ymax > ymin)) return 0;
  return (xmax - xmin) * (ymax - ymin);
}

test('degenerate (all points identical) yields empty contours', () => {
  const n = 500;
  const xs = new Float64Array(n).fill(2.5);
  const ys = new Float64Array(n).fill(-1.0);
  const levels = densityContours(xs, ys, [0.68, 0.95]);
  assert.equal(levels.length, 2);
  for (const l of levels) assert.equal(l.segments.length, 0);
});

test('too few points → empty (below minPoints)', () => {
  const { xs, ys } = gaussianBlob(10, 1);
  const levels = densityContours(xs, ys, [0.68, 0.95], { minPoints: 30 });
  for (const l of levels) assert.equal(l.segments.length, 0);
});

test('fracs are echoed back in order', () => {
  const { xs, ys } = gaussianBlob(2000, 7);
  const levels = densityContours(xs, ys, [0.68, 0.95]);
  assert.deepEqual(levels.map((l) => l.frac), [0.68, 0.95]);
});

test('Gaussian blob: both credible contours are traced', () => {
  const { xs, ys } = gaussianBlob(4000, 42);
  const levels = densityContours(xs, ys, [0.68, 0.95]);
  assert.ok(levels[0].segments.length > 0, '68% contour has segments');
  assert.ok(levels[1].segments.length > 0, '95% contour has segments');
});

test('inner (68%) contour encloses less area than outer (95%)', () => {
  const { xs, ys } = gaussianBlob(4000, 42);
  const [l68, l95] = densityContours(xs, ys, [0.68, 0.95]);
  const a68 = bboxArea(l68.segments);
  const a95 = bboxArea(l95.segments);
  assert.ok(a68 > 0 && a95 > 0);
  assert.ok(a68 < a95, `68% bbox (${a68.toFixed(3)}) should be smaller than 95% (${a95.toFixed(3)})`);
});

test('contour segments stay within the data range', () => {
  const { xs, ys } = gaussianBlob(4000, 42);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    xmin = Math.min(xmin, xs[i]); xmax = Math.max(xmax, xs[i]);
    ymin = Math.min(ymin, ys[i]); ymax = Math.max(ymax, ys[i]);
  }
  const eps = 1e-6;
  for (const l of densityContours(xs, ys, [0.68, 0.95])) {
    for (const [[x0, y0], [x1, y1]] of l.segments) {
      for (const x of [x0, x1]) assert.ok(x >= xmin - eps && x <= xmax + eps);
      for (const y of [y0, y1]) assert.ok(y >= ymin - eps && y <= ymax + eps);
    }
  }
});

test('a standard-normal 2D blob 68% contour ~ radius 1.5 (2D HPD)', () => {
  // For an isotropic unit Gaussian, the 68%-mass HPD region is the disk of
  // radius sqrt(-2 ln(1-0.68)) ≈ 1.51. The traced 68% contour's mean radius
  // from the origin should land near there (loose bound — histogram + blur).
  const { xs, ys } = gaussianBlob(8000, 123);
  const [l68] = densityContours(xs, ys, [0.68, 0.95]);
  let rsum = 0, cnt = 0;
  for (const [[x0, y0], [x1, y1]] of l68.segments) {
    rsum += Math.hypot((x0 + x1) / 2, (y0 + y1) / 2); cnt++;
  }
  const meanR = rsum / cnt;
  const expected = Math.sqrt(-2 * Math.log(1 - 0.68));  // ≈ 1.51
  assert.ok(Math.abs(meanR - expected) < 0.5,
    `mean 68% contour radius ${meanR.toFixed(3)} vs expected ${expected.toFixed(3)}`);
});
