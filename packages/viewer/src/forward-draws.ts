// @flatppl/viewer — forward-sampling draw settings —
// DOM-free mutations of the forward-draw settings (SAMPLE_COUNT,
// MARGINALIZATION_COUNT) plus their cache-invalidation contract. The
// configUpdate handler (main.ts) routes through here so cache invalidation
// lives in one place. No DOM / render imports — unit-tested directly.
// (The in-panel cog writes SAMPLE_COUNT / rootSeed directly and re-renders
// via onInferenceChange, mirroring how the other sampler knobs work.)
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
