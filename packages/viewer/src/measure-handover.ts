// @ts-check
// @flatppl/viewer — cached samples that travel with the plots panel —
//
// When a host moves the plots panel into another window and back, the
// samples of the plotted binding go with it, so a slow posterior is not
// drawn again. exportMeasures takes them out of this mount's cache;
// importMeasures parks them in another mount, which adopts them on its next
// plot render if they were drawn for the same model and the same sampling
// settings. Anything else, a missing or oversized or non-cloneable handover
// included, falls back to sampling as before. Hosts that never call these
// functions (VS Code) leave ctx.pendingHandover unset, and nothing here runs.

import type { Ctx } from './types';

export interface MeasureHandover {
  binding: string;
  validity: { modelKey: string; settingsKey: string };
  entries: [string, any][];
  bytes: number;
  /** Why entries is empty: 'cap' (over the size cap) or 'clone' (not cloneable). */
  omitted?: 'cap' | 'clone';
}

export type HandoverStatus = 'none' | 'pending' | 'adopted' | 'rejected:cap' | 'rejected:clone'
  | 'rejected:model' | 'rejected:settings' | 'rejected:empty';

/** 64 MiB: about 160 scalar leaves at the default 50 000 draws. A host can
 *  set another cap with __FLATPPL_CONFIG__.handoverCapBytes. */
export const DEFAULT_HANDOVER_CAP = 64 * 1024 * 1024;

/** The cap this mount applies: the host's configured one, else the default. */
export function handoverCap(ctx: Ctx): number {
  const configured = ctx.CONFIG && ctx.CONFIG.handoverCapBytes;
  return typeof configured === 'number' && configured >= 0 ? configured : DEFAULT_HANDOVER_CAP;
}

/** FNV-1a over bytes or text: a content digest that is the same in every
 *  window, unlike a per-mount counter. */
function digest(value: unknown): string {
  let h = 0x811c9dc5;
  const mix = (b: number) => { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; };
  if (ArrayBuffer.isView(value)) {
    const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    for (let i = 0; i < bytes.length; i++) mix(bytes[i]);
  } else {
    const text = typeof value === 'string' ? value : JSON.stringify(value) ?? '';
    for (let i = 0; i < text.length; i++) { const c = text.charCodeAt(i); mix(c & 0xff); mix(c >>> 8); }
  }
  return h.toString(16);
}

/** The model the cached samples were drawn from: analysed source, path,
 *  dependency sources and data files, by content. */
export function handoverModelKey(ctx: Ctx): string {
  const files = ctx.currentBundleFiles || {};
  const fileKeys = Object.keys(files).sort().map((k) => [k, digest(files[k])]);
  return JSON.stringify([ctx.analyzedSource ?? null, ctx.currentPath ?? null,
    ctx.currentBundleSources ?? null, fileKeys]);
}

/** Every setting that changes a draw. */
export function handoverSettingsKey(ctx: Ctx): string {
  return JSON.stringify([ctx.SAMPLE_COUNT, ctx.rootSeed, ctx.MARGINALIZATION_COUNT,
    ctx.REJECTION_BUDGET, ctx.inferenceOpts]);
}

/** Bytes the entries' samples take when cloned: each typed-array buffer
 *  once, and 8 bytes per number in plain arrays (record and array leaves
 *  can hold plain arrays), nested arrays included. */
export function sampleBytes(value: unknown, seen: Set<ArrayBufferLike> = new Set()): number {
  if (!value || typeof value !== 'object') return 0;
  if (ArrayBuffer.isView(value)) {
    if (seen.has(value.buffer)) return 0;
    seen.add(value.buffer);
    return value.buffer.byteLength;
  }
  let n = 0;
  if (Array.isArray(value)) {
    for (const v of value) n += typeof v === 'number' ? 8 : sampleBytes(v, seen);
    return n;
  }
  for (const v of Object.values(value)) n += sampleBytes(v, seen);
  return n;
}

/** Take the cached samples of `binding` (default: the plotted binding) out
 *  of this mount's cache. Null when nothing is plotted. */
export function exportMeasures(ctx: Ctx, binding?: string | null, capBytes = handoverCap(ctx)): MeasureHandover | null {
  if (binding == null) binding = ctx.currentPlotBindingName;
  if (!binding) return null;
  const validity = { modelKey: handoverModelKey(ctx), settingsKey: handoverSettingsKey(ctx) };
  const keys = Array.from(ctx.measureCache.keys()).filter((k) => k === binding || k.startsWith(binding + '|'));
  const entries: [string, any][] = keys.map((k) => [k, ctx.measureCache.get(k)]);
  const bytes = sampleBytes(entries);
  if (bytes > capBytes) return { binding, validity, entries: [], bytes, omitted: 'cap' };
  let copy: [string, any][];
  try {
    copy = structuredClone(entries);
  } catch (err) {
    console.warn('FlatPPL: the cached samples of ' + binding + ' cannot be moved; they will be drawn again.', err);
    return { binding, validity, entries: [], bytes, omitted: 'clone' };
  }
  // The samples now belong to the receiving window.
  for (const k of keys) ctx.measureCache.delete(k);
  return { binding, validity, entries: copy, bytes };
}

/** Park a handover; the next plot render adopts it if it still fits. */
export function importMeasures(ctx: Ctx, handover: MeasureHandover | null): void {
  if (!handover) { ctx.pendingHandover = null; return; }
  if (handover.omitted) { ctx.pendingHandover = null; ctx.handoverStatus = 'rejected:' + handover.omitted as HandoverStatus; return; }
  if (!handover.entries.length) { ctx.pendingHandover = null; ctx.handoverStatus = 'rejected:empty'; return; }
  ctx.pendingHandover = handover;
  ctx.handoverStatus = 'pending';
}

/** Called before a plot render: move a matching handover into the cache. */
export function adoptPendingMeasures(ctx: Ctx): void {
  const handover = ctx.pendingHandover;
  if (!handover || !ctx.analyzedSource) return;
  ctx.pendingHandover = null;
  if (handover.validity.modelKey !== handoverModelKey(ctx)) { ctx.handoverStatus = 'rejected:model'; return; }
  if (handover.validity.settingsKey !== handoverSettingsKey(ctx)) { ctx.handoverStatus = 'rejected:settings'; return; }
  for (const [key, measure] of handover.entries) {
    // A draw this mount already started or finished wins.
    if (!ctx.measureCache.has(key)) ctx.measureCache.set(key, measure);
  }
  ctx.handoverStatus = 'adopted';
}
