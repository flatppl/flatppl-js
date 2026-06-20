'use strict';

// =====================================================================
// optimizer-coords.test.ts
// =====================================================================
//
// The optimiser works in a normalised, mostly-unconstrained coordinate
// `z` (origin at the pivot, |z|≈1 ≙ one plot-scale displacement), while
// the objective is evaluated in the model's original parameter space `x`.
// `makeCoords` builds the x↔z maps + a feasibility projection from each
// axis's value-set domain:
//   - reals          → affine, no projection;
//   - interval(lo,hi)→ affine + box clamp (boundary maxima reachable);
//   - posreals       → log map (stays > 0, unbounded above).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeCoords } = require('../optimizer/coords.ts');

test('reals: affine round-trip; pivot maps to z=0; one scale unit = one z unit', () => {
  const c = makeCoords({ domains: [{ kind: 'real' }], scales: [10], x0: [5] });
  assert.deepEqual(c.toZ([5]), [0]);
  assert.ok(Math.abs(c.toX([1])[0] - 15) < 1e-12, `toX(1)=${c.toX([1])[0]}`);
  const rt = c.toX(c.toZ([42]));
  assert.ok(Math.abs(rt[0] - 42) < 1e-9, `round-trip ${rt[0]}`);
  assert.deepEqual(c.project([3.3]), [3.3]); // no projection for reals
});

test('interval: box-clamped; large |z| saturates to the bounds (boundary reachable)', () => {
  const c = makeCoords({ domains: [{ kind: 'interval', lo: 0, hi: 1 }], scales: [1], x0: [0.5] });
  const hi = c.toX(c.project([1e6]))[0];
  const lo = c.toX(c.project([-1e6]))[0];
  assert.ok(Math.abs(hi - 1) < 1e-9, `upper bound: ${hi}`);
  assert.ok(Math.abs(lo - 0) < 1e-9, `lower bound: ${lo}`);
  // interior point round-trips
  const rt = c.toX(c.project(c.toZ([0.7])))[0];
  assert.ok(Math.abs(rt - 0.7) < 1e-9, `interior round-trip ${rt}`);
});

test('posreals: log map keeps x > 0 for any z; pivot maps to z=0', () => {
  const c = makeCoords({ domains: [{ kind: 'posreals' }], scales: [2], x0: [3] });
  assert.deepEqual(c.toZ([3]), [0]);
  assert.ok(c.toX([-1e3])[0] > 0, 'stays positive for large negative z');
  assert.ok(c.toX([1e3])[0] > 0, 'stays positive for large positive z');
  const rt = c.toX(c.toZ([0.01]))[0];
  assert.ok(Math.abs(rt - 0.01) < 1e-9, `round-trip ${rt}`);
});

test('mixed multi-axis: independent per-axis maps; degenerate scale falls back', () => {
  const c = makeCoords({
    domains: [{ kind: 'real' }, { kind: 'interval', lo: -2, hi: 2 }, { kind: 'posreals' }],
    scales: [0, 4, 1], // first scale is degenerate → fallback to max(1,|x0|)
    x0: [7, 0, 1],
  });
  assert.equal(c.dim, 3);
  // axis 0 fallback scale = max(1, |7|) = 7
  assert.ok(Math.abs(c.toX([0, 0, 0])[0] - 7) < 1e-9);
  assert.ok(Math.abs(c.toX([1, 0, 0])[0] - 14) < 1e-9, 'fallback scale = 7');
  // axis 1 clamps
  assert.ok(Math.abs(c.toX(c.project([0, 1e6, 0]))[1] - 2) < 1e-9);
  // axis 2 positive
  assert.ok(c.toX([0, 0, -50])[2] > 0);
});
