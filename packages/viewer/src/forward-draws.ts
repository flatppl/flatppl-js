// @flatppl/viewer — forward-sampling draw controls state —
// DOM-free mutations of the forward-draw settings (SAMPLE_COUNT,
// MARGINALIZATION_COUNT, rootSeed) plus their cache-invalidation
// contract. The configUpdate handler (main.ts) and buildDrawControl
// (render-controls.ts) both go through here so cache invalidation
// lives in one place. No DOM / render imports — unit-tested directly.
import type { Ctx } from './types';

// A cached EmpiricalMeasure was sized to the old SAMPLE_COUNT /
// estimated at the old M / seeded from the old rootSeed, so any
// forward-draw change must drop both caches before re-rendering.
export function clearForwardCaches(ctx: Ctx): void {
  ctx.measureCache = new Map();
  ctx.histogramCache = new Map();
}

export function setSampleCount(ctx: Ctx, n: number): boolean {
  if (!Number.isFinite(n) || n <= 0 || (n | 0) === ctx.SAMPLE_COUNT) return false;
  ctx.SAMPLE_COUNT = n | 0;
  clearForwardCaches(ctx);
  return true;
}

export function setMarginalizationCount(ctx: Ctx, n: number): boolean {
  if (!Number.isFinite(n) || n < 1 || (n | 0) === ctx.MARGINALIZATION_COUNT) return false;
  ctx.MARGINALIZATION_COUNT = n | 0;
  clearForwardCaches(ctx);
  return true;
}

// Deterministic seed pin. Mirrors into the posterior seed only when it
// is still unset, so pinning the forward seed makes the WHOLE view
// reproducible without a second control — but never clobbers a seed the
// user set explicitly in the Sampler control.
export function pinForwardSeed(ctx: Ctx, v: number): void {
  ctx.rootSeed = v | 0;
  if (ctx.inferenceOpts && ctx.inferenceOpts.seed == null) {
    ctx.inferenceOpts.seed = v | 0;
  }
}

// Deterministic, reproducible re-roll: bump the seed by one and drop
// caches so the next render draws a fresh (but reproducible) sample.
export function rerollForwardSeed(ctx: Ctx): number {
  ctx.rootSeed = (ctx.rootSeed | 0) + 1;
  clearForwardCaches(ctx);
  return ctx.rootSeed;
}
