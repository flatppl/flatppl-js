// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { binGrid } = await import('./contour2d.ts');
const { jointDensityField, jointMeanField } = await import('./render-field.ts');

// Deterministic PRNG (mulberry32) + Box-Muller, same as contour2d.test.ts, so
// the oracle comparisons below are reproducible run to run.
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

// ---- binGrid -------------------------------------------------------------

test('binGrid: four corner points land one per bin at G=2', () => {
  const g = binGrid([0, 1, 0, 1], [0, 0, 1, 1], 2);
  // x range [0,1], dx = 0.5 → centres 0.25, 0.75. Index is by*G+bx.
  assert.deepEqual(Array.from(g.gx), [0.25, 0.75]);
  assert.deepEqual(Array.from(g.gy), [0.25, 0.75]);
  assert.deepEqual(Array.from(g.field), [1, 1, 1, 1]);
  assert.equal(g.total, 4);
  assert.equal(g.count, 4);
});

test('binGrid: weights accumulate per bin instead of counts', () => {
  const g = binGrid([0, 1, 0, 1], [0, 0, 1, 1], 2, [1, 2, 3, 4]);
  assert.deepEqual(Array.from(g.field), [1, 2, 3, 4]);
  assert.equal(g.total, 10);
  assert.equal(g.count, 4);
});

test('binGrid: negative weights are kept (signed numerator)', () => {
  const g = binGrid([0, 1], [0, 1], 2, [-3, 4]);
  assert.equal(g.field[0], -3);
  assert.equal(g.field[3], 4);
  assert.equal(g.total, 1);
});

test('binGrid: a degenerate axis yields null', () => {
  assert.equal(binGrid([1, 1, 1], [0, 1, 2], 4), null);
  assert.equal(binGrid([0, 1, 2], [5, 5, 5], 4), null);
});

test('binGrid: non-finite coordinates and weights are skipped', () => {
  const g = binGrid([0, 1, NaN, 0.5], [0, 1, 0.5, Infinity], 2, [1, 1, 1, 1]);
  assert.equal(g.count, 2);
  const gw = binGrid([0, 1, 0.5], [0, 1, 0.5], 2, [1, 1, NaN]);
  assert.equal(gw.count, 2);
  assert.equal(gw.total, 2);
});

// ---- jointDensityField ---------------------------------------------------

test('jointDensityField: below 30 points yields null', () => {
  const { xs, ys } = gaussianBlob(20, 3);
  assert.equal(jointDensityField(xs, ys), null);
});

test('jointDensityField: integrates to 1 over the grid', () => {
  const { xs, ys } = gaussianBlob(20000, 11);
  const f = jointDensityField(xs, ys);
  const dx = f.xs[1] - f.xs[0], dy = f.ys[1] - f.ys[0];
  let mass = 0;
  for (let i = 0; i < f.z.length; i++) mass += f.z[i] * dx * dy;
  assert.ok(Math.abs(mass - 1) < 1e-9, `grid mass ${mass} should be 1`);
});

// Field value at the bin containing (x, y), or NaN when outside the grid.
function at(f, x, y) {
  const dx = f.xs[1] - f.xs[0], dy = f.ys[1] - f.ys[0];
  const c = Math.round((x - f.xs[0]) / dx);
  const r = Math.round((y - f.ys[0]) / dy);
  if (c < 0 || c >= f.xs.length || r < 0 || r >= f.ys.length) return NaN;
  return f.z[r * f.xs.length + c];
}

test('jointDensityField: unit-Gaussian shape matches the closed form', () => {
  // For x, y ~ N(0,1) independent the joint density is exp(-(x²+y²)/2) / 2π.
  // The ratio between two points is normalisation-free, so it tests the shape
  // without depending on how much tail the clip trimmed.
  const { xs, ys } = gaussianBlob(40000, 42);
  const f = jointDensityField(xs, ys);
  const centre = at(f, 0, 0);
  for (const [px, py] of [[1.5, 0], [0, 1.5], [1, 1]]) {
    const ratio = at(f, px, py) / centre;
    const closedForm = Math.exp(-(px * px + py * py) / 2);
    assert.ok(Math.abs(ratio - closedForm) / closedForm < 0.1,
      `density ratio at (${px}, ${py}) was ${ratio.toFixed(4)}, closed form ${closedForm.toFixed(4)}`);
  }
  // Absolute scale: 1/(2π) ≈ 0.15915, divided by the mass the clip retained
  // (the central 99% of each axis, so ~0.99² = 0.980) because the surface is
  // renormalised over the window it draws.
  const expected = (1 / (2 * Math.PI)) / (0.99 * 0.99);
  assert.ok(Math.abs(centre - expected) / expected < 0.05,
    `centre density ${centre.toFixed(5)} vs ${expected.toFixed(5)}`);
});

test('jointDensityField: importance weights move the distribution', () => {
  // w(x) = exp(x - 1/2) is the density ratio N(1,1)/N(0,1) in x, so the
  // weighted sample targets N(1,1) × N(0,1). The surface's centroid must then
  // be (1, 0) rather than the unweighted (0, 0). Centroid, not argmax: a
  // far-tail atom carries weight e^3.5 and can own a single bin.
  const { xs, ys } = gaussianBlob(40000, 77);
  const w = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) w[i] = Math.exp(xs[i] - 0.5);
  const f = jointDensityField(xs, ys, w);
  const nx = f.xs.length;
  let mass = 0, mx = 0, my = 0;
  for (let r = 0; r < f.ys.length; r++) {
    for (let c = 0; c < nx; c++) {
      const v = f.z[r * nx + c];
      mass += v; mx += v * f.xs[c]; my += v * f.ys[r];
    }
  }
  mx /= mass; my /= mass;
  assert.ok(Math.abs(mx - 1) < 0.15, `weighted centroid x ${mx.toFixed(3)} should be near 1`);
  assert.ok(Math.abs(my) < 0.15, `weighted centroid y ${my.toFixed(3)} should be near 0`);
});

test('jointDensityField: a heavy tail does not set the grid range', () => {
  // y is a standard Cauchy (ratio of two normals): its sample range runs to
  // thousands while its mass sits within a few units. The grid must follow the
  // mass, or every draw lands in one row of bins.
  const a = gaussianBlob(20000, 55);
  const b = gaussianBlob(20000, 56);
  const ys = new Float64Array(a.xs.length);
  for (let i = 0; i < ys.length; i++) ys[i] = a.ys[i] / b.ys[i];
  let dataMax = 0;
  for (let i = 0; i < ys.length; i++) dataMax = Math.max(dataMax, Math.abs(ys[i]));
  const f = jointDensityField(a.xs, ys);
  const gridSpan = f.ys[f.ys.length - 1] - f.ys[0];
  assert.ok(dataMax > 100, `expected a heavy tail, max |y| was ${dataMax}`);
  assert.ok(gridSpan < 300, `grid y span ${gridSpan.toFixed(1)} should track the mass, not ${dataMax.toFixed(0)}`);
  // The mode still has to be in the middle rows, not pinned to an edge.
  let peak = -Infinity, pr = -1;
  const nx = f.xs.length;
  for (let r = 0; r < f.ys.length; r++) {
    for (let c = 0; c < nx; c++) if (f.z[r * nx + c] > peak) { peak = f.z[r * nx + c]; pr = r; }
  }
  assert.ok(pr > 2 && pr < f.ys.length - 3, `peak row ${pr} of ${f.ys.length} is at an edge`);
});

// ---- jointMeanField ------------------------------------------------------

test('jointMeanField: a constant third quantity is reproduced exactly', () => {
  const { xs, ys } = gaussianBlob(5000, 5);
  const v = new Float64Array(xs.length).fill(7);
  const f = jointMeanField(xs, ys, v);
  let occupied = 0;
  for (let i = 0; i < f.z.length; i++) {
    if (Number.isNaN(f.z[i])) continue;
    occupied++;
    assert.equal(f.z[i], 7);
  }
  assert.ok(occupied > 100, `expected many occupied bins, got ${occupied}`);
});

test('jointMeanField: empty bins are NaN', () => {
  // Two tight clusters at opposite corners leave the middle of the grid empty.
  const n = 2000;
  const xs = new Float64Array(n), ys = new Float64Array(n), v = new Float64Array(n);
  const rnd = mulberry32(9);
  for (let i = 0; i < n; i++) {
    const far = i % 2 === 0;
    xs[i] = (far ? 10 : 0) + rnd() * 0.1;
    ys[i] = (far ? 10 : 0) + rnd() * 0.1;
    v[i] = 1;
  }
  const f = jointMeanField(xs, ys, v);
  let nan = 0;
  for (let i = 0; i < f.z.length; i++) if (Number.isNaN(f.z[i])) nan++;
  assert.ok(nan > f.z.length / 2, `expected mostly empty bins, got ${nan}/${f.z.length}`);
});

test('jointMeanField: mean of z = x tracks the bin x centre', () => {
  // z depends on x alone, so the per-bin weighted mean must be the mean of x
  // over that bin's x span — within half a bin width of its centre. This is
  // the geometric check: a swapped axis or a numerator binned on a different
  // grid than the denominator breaks it.
  const { xs, ys } = gaussianBlob(40000, 21);
  const f = jointMeanField(xs, ys, xs);
  const dx = f.xs[1] - f.xs[0];
  const nx = f.xs.length;
  let checked = 0;
  for (let r = 0; r < f.ys.length; r++) {
    for (let c = 0; c < nx; c++) {
      const m = f.z[r * nx + c];
      if (Number.isNaN(m)) continue;
      checked++;
      assert.ok(Math.abs(m - f.xs[c]) <= dx / 2 + 1e-9,
        `bin mean ${m} outside bin [${f.xs[c] - dx / 2}, ${f.xs[c] + dx / 2}]`);
    }
  }
  assert.ok(checked > 100);
});

test('jointMeanField: a weighted mean stays inside its bin and rises with it', () => {
  // Weighting cannot move a bin's mean of z = y outside that bin's y span, and
  // the means must increase down the rows. Compared within ONE weighted field:
  // the clip window is itself weight-dependent, so a weighted and an unweighted
  // field are on different grids and their cells are not comparable.
  const { xs, ys } = gaussianBlob(40000, 31);
  const w = new Float64Array(ys.length);
  for (let i = 0; i < ys.length; i++) w[i] = Math.exp(4 * ys[i]);
  const f = jointMeanField(xs, ys, ys, w);
  const dy = f.ys[1] - f.ys[0];
  const nx = f.xs.length;
  let compared = 0, rising = 0, rowPairs = 0;
  for (let r = 0; r < f.ys.length; r++) {
    for (let c = 0; c < nx; c++) {
      const m = f.z[r * nx + c];
      if (Number.isNaN(m)) continue;
      compared++;
      assert.ok(Math.abs(m - f.ys[r]) <= dy / 2 + 1e-9,
        `bin mean ${m} left bin ${f.ys[r]} ± ${dy / 2}`);
      if (r > 0) {
        const above = f.z[(r - 1) * nx + c];
        if (!Number.isNaN(above)) { rowPairs++; if (m > above) rising++; }
      }
    }
  }
  assert.ok(compared > 100);
  assert.ok(rising === rowPairs,
    `${rising}/${rowPairs} vertical pairs increase; every one should`);
});
