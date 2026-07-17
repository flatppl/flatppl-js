'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { adaptiveCubature } = require('../quadrature.ts');

test('1-D constant integrates to 1', () => {
  const r = adaptiveCubature(() => 1.0, 1);
  assert.ok(Math.abs(r.Z - 1.0) < 1e-10, `Z=${r.Z}`);
});

test('1-D linear ∫u du = 0.5', () => {
  const r = adaptiveCubature((u: number[]) => u[0], 1);
  assert.ok(Math.abs(r.Z - 0.5) < 1e-10, `Z=${r.Z}`);
});

test('2-D ∫∫(u+v) = 1.0', () => {
  const r = adaptiveCubature((u: number[]) => u[0] + u[1], 2);
  assert.ok(Math.abs(r.Z - 1.0) < 1e-10, `Z=${r.Z}`);
});

test('1-D kink ∫√|u-0.5| du = 0.4714045207910317 (adaptive refines the kink)', () => {
  const EXACT = 0.4714045207910317; // = (4/3)·(0.5)^1.5
  const r = adaptiveCubature((u: number[]) => Math.sqrt(Math.abs(u[0] - 0.5)), 1);
  assert.ok(Math.abs(r.Z - EXACT) < 1e-6, `Z=${r.Z} Δ=${Math.abs(r.Z - EXACT)}`);
});

test('never evaluates at a boundary node (interior rule)', () => {
  let touchedBoundary = false;
  adaptiveCubature((u: number[]) => { if (u[0] <= 0 || u[0] >= 1) touchedBoundary = true; return 1; }, 1);
  assert.equal(touchedBoundary, false);
});

test('narrow interior peak is not stepped over (minimum initial subdivision)', () => {
  // A Gaussian spike (σ=0.01) at u=0.3 — deliberately off every root-cell
  // Gauss node (GL3 {0.113,0.5,0.887}, GL2 {0.211,0.789}). A single seed cell
  // would sample ~0 at all of them, report a falsely-tiny error, and terminate
  // at Z≈0. The minimum initial subdivision must catch it. Analytic ∫₀¹ ≈ σ√(2π)
  // (both bounds are ≥30σ from the peak, so the truncated tails are negligible).
  const EXACT = 0.01 * Math.sqrt(2 * Math.PI); // 0.025066282746310002
  const r = adaptiveCubature(
    (u: number[]) => Math.exp(-((u[0] - 0.3) ** 2) / (2 * 0.01 * 0.01)), 1);
  assert.ok(Math.abs(r.Z - EXACT) < 1e-6, `Z=${r.Z} Δ=${Math.abs(r.Z - EXACT)}`);
});
