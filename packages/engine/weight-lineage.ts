'use strict';

// Ancestry for importance-weight arrays.
//
// The engine represents a weighted empirical measure as
// `{ samples, logWeights }`, and a derived measure's atom-i weight is the
// JOINT importance weight of its parents' atom i — the product over the
// DISTINCT weighting events the parents bring in (spec §06 "Density
// reweighting", "Normalization and mass", "Joint composition"; §04
// "Reification to measures" for what `record(...)` reifies).
//
// `empirical.propagateLogWeights` used to recognise a shared event by
// reference identity of the `logWeights` array alone. That works only while an
// inheritance chain passes the same array forward. The moment two streams
// COMBINE, the sum is a fresh array and both ancestors are unrecoverable, so a
// later consumer that sees the constituents AND the merged descendant counts
// each constituent twice:
//
//   a ~ normalize(weighted(fn(exp(_)), Normal(0, 1)))     # weights A
//   b ~ normalize(weighted(fn(exp(0.5 * _)), Normal(0, 1)))  # weights B
//   y ~ Normal(a + b, 1.0)                                # weights A + B
//   r = record(a = a, b = b, y = y)                       # wanted A + B
//                                                         # got 2A + 2B
//
// The exact means are [1, 0.5, 1.5] (completing the square makes the tilted
// normals Normal(1, 1) and Normal(0.5, 1)); the record measured
// [2.01, 1.01, 3.03] at 400000 atoms, with the centered record weights exactly
// twice the centered `y` weights.
//
// So this module records what each published weight array is made OF. A
// non-null `logWeights` array is the sum of an ordered list of WEIGHTING
// EVENTS, each carrying its own per-atom contribution and constant offset, and
// a merge unions the events instead of adding the arrays.
//
// What is NOT here: any change to how a weight is COMPUTED. A site hands over
// the contribution it already calculated. Deriving a contribution as
// `output - parent` is forbidden — a mask's `-Infinity - -Infinity` is `NaN`,
// and the sign of a pooled shift would be lost.
//
// Unknown provenance degrades to the old behaviour: `lineageOf` mints one
// OPAQUE event per unseen array, keyed by that array, so a site this module
// does not know about — or an array that crossed the worker boundary, where a
// `WeakMap` cannot follow — behaves exactly as it did before. Teaching a site
// is therefore always a strict improvement and never a regression.

/**
 * One weighting event: an operation that multiplied the measure's atoms by a
 * weight. `values` is the per-atom log contribution over the engine's N-atom
 * axis, `null` when the event is a pure constant. `offset` is added at every
 * atom (a `normalize` pooled shift, a constant `weighted` factor, the
 * `-log(N)` empirical baseline).
 *
 * `id` is the identity: two events with equal numbers are still distinct
 * events unless they came from the same operation.
 */
export type WeightEvent = {
  id: number;
  values: Float64Array | null;
  offset: number;
};

export type WeightLineage = {
  events: readonly WeightEvent[];
};

let nextEventId = 1;

// Keyed by the published array, so an array that nothing references any more
// takes its lineage with it. Values are frozen event lists; a registered
// lineage is never mutated in place, since consumers hold on to the list.
const lineages: WeakMap<Float64Array, WeightLineage> = new WeakMap();

/**
 * Mint a fresh weighting event. `values` must be the contribution this
 * operation adds, not the resulting weights.
 */
export function newEvent(
  values: Float64Array | null,
  offset: number = 0,
): WeightEvent {
  return { id: nextEventId++, values, offset };
}

/**
 * The events an array is the sum of.
 *
 * An unregistered array becomes ONE opaque event holding the whole array. The
 * result is memoised against the array so that meeting the same unknown array
 * twice yields the same event id — that memoisation is what preserves the old
 * reference-identity dedupe for every site this module has not been taught.
 */
export function lineageOf(arr: Float64Array): WeightLineage {
  const known = lineages.get(arr);
  if (known) return known;
  const fresh: WeightLineage = { events: Object.freeze([newEvent(arr, 0)]) };
  lineages.set(arr, fresh);
  return fresh;
}

/**
 * Record that `arr` is the sum of `events`.
 *
 * The caller owns the invariant `arr[i] === Σ_e ((e.values?.[i] ?? 0) + e.offset)`
 * up to float association order. Registering a lineage that does not hold makes
 * every later merge wrong, so a site that cannot guarantee it must register
 * nothing and take the opaque fallback instead.
 *
 * Registration must happen AFTER the site finishes writing `arr`, and the
 * events' `values` arrays must not be mutated afterwards.
 *
 * Returns `arr`, so a site can register inline in its return expression.
 */
export function register(
  arr: Float64Array,
  events: readonly WeightEvent[],
): Float64Array {
  lineages.set(arr, { events: Object.freeze(events.slice()) });
  return arr;
}

/**
 * Register `out` as `parent`'s events plus one new event of its own — the shape
 * every weight-transforming operation has (`weighted`, `logweighted`,
 * `normalize`, `bayesupdate`): it starts from a parent's weights and adds its
 * own per-atom or constant correction.
 *
 * `parentWeights` is the array the site accumulated FROM (`null` when the
 * parent was implicit-uniform and the site supplied its own baseline, which
 * then belongs in `extraEvents`).
 */
export function derive(
  out: Float64Array,
  parentWeights: Float64Array | null,
  ownEvents: readonly WeightEvent[],
): Float64Array {
  const inherited = parentWeights ? lineageOf(parentWeights).events : [];
  return register(out, inherited.concat(ownEvents));
}

/**
 * The union of several arrays' events, in first-seen order.
 *
 * First-seen order keeps a re-summed array bit-comparable with the merge that
 * first produced it: float addition is deterministic, so the same events added
 * in the same order give the same bits.
 */
export function unionEvents(
  arrays: readonly Float64Array[],
): WeightEvent[] {
  const seen = new Set<number>();
  const out: WeightEvent[] = [];
  for (const arr of arrays) {
    for (const e of lineageOf(arr).events) {
      if (seen.has(e.id)) continue;
      seen.add(e.id);
      out.push(e);
    }
  }
  return out;
}

/**
 * True when `arr` is exactly the sum of `events` — same event set, no more and
 * no fewer. Order is irrelevant: the array already holds the sum, and addition
 * commutes over a set of events even when the bits depend on the order they
 * were added in.
 */
export function coversExactly(
  arr: Float64Array,
  events: readonly WeightEvent[],
): boolean {
  const own = lineageOf(arr).events;
  if (own.length !== events.length) return false;
  const ids = new Set<number>();
  for (const e of own) ids.add(e.id);
  for (const e of events) {
    if (!ids.has(e.id)) return false;
  }
  return true;
}

/**
 * Sum `events` into a fresh array of `length` atoms and register it.
 *
 * A constant event contributes only its offset; a per-atom event contributes
 * `values[i] + offset`. Contributions are added in event order.
 */
export function sumEvents(
  events: readonly WeightEvent[],
  length: number,
): Float64Array {
  const out = new Float64Array(length);
  for (const e of events) {
    const v = e.values;
    if (v) {
      if (e.offset === 0) {
        for (let i = 0; i < length; i++) out[i] += v[i];
      } else {
        for (let i = 0; i < length; i++) out[i] += v[i] + e.offset;
      }
    } else if (e.offset !== 0) {
      for (let i = 0; i < length; i++) out[i] += e.offset;
    }
  }
  return register(out, events);
}

/**
 * The atom-axis length an event spans, or -1 for a constant event. Used by the
 * merge to refuse events that do not share the engine's N-atom axis.
 */
export function eventLength(e: WeightEvent): number {
  return e.values ? e.values.length : -1;
}
