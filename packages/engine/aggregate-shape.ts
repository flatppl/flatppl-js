'use strict';

// =====================================================================
// aggregate-shape.ts — canonical form of an aggregate IR node
// =====================================================================
//
// Spec §04 §sec:aggregate. The IR shape
//
//     aggregate(reducer, output_axes, body)
//
// fully determines the contraction:
//   - reduce axes = axes used in body but not listed in output_axes
//     (insertion order from a depth-first body walk)
//   - canonical axis order = [output_axes..., reduce_axes...] — every
//     contraction lowering (single-point and atom-batched broadcast-
//     reduce, AGGREGATE_PATTERNS specialisers, future MLIR linalg
//     export) builds against this ordering
//   - axis lengths: typeinfer can resolve some statically; the runtime
//     resolves the rest from get(container, .axis) shape inspection at
//     evaluation time.
//
// Before this module existed (engine-concepts §11 / TODO P1) the same
// walk was duplicated in three places: typeinfer's inferAggregate, the
// single-point runtime (`_inferAggregateAxisLengths` +
// `_collectInScopeAxisNames`), and the atom-batched runtime
// (`_inferAxisLengthsN`). Each had silently-divergent edge cases. Now
// the canonical form is computed ONCE per IR node — annotated lazily on
// `ir.meta.aggregateCanonical` and consumed by every layer.
//
// MLIR linalg.generic carries `indexing_maps` + `iterator_types` +
// `iteration_space` on the op itself; JAX einsum decomposes once into
// `dot_general(contracting_dimensions, batch_dimensions)` with axis
// classification baked into the op. This module is the FlatPPL
// analogue.
//
// CJS module (mirrors sampler-aggregate / standard-modules — anything
// that consumes via require() and may participate in cycles).

export interface AggregateCanonical {
  /** Output axes in spec-declared order (args[1]'s vector entries). */
  outAxes: string[];
  /** Axes referenced in body but NOT in outAxes — order is first-
   *  occurrence in a depth-first body walk, so the canonical ordering
   *  is deterministic and stable across re-annotations. */
  reduceAxes: string[];
  /** Canonical iteration ordering: [outAxes..., reduceAxes...]. The
   *  trailing reduce block makes the final reduction a contiguous-
   *  block tail-reduce for every output slot. */
  canonicalAxes: string[];
  /** Best-effort static axis lengths from typeinfer's container shapes.
   *  Missing entries (and `%dynamic` entries) are resolved at runtime
   *  via `resolveAxisLengths`. The `axis -> length` map is keyed by
   *  axis name. */
  axisLengths: Record<string, number | '%dynamic'>;
  /** True iff every entry in canonicalAxes has a concrete integer
   *  length in axisLengths — runtime can skip its own resolution pass.
   *  When false, the runtime resolver consults values. */
  fullyResolved: boolean;
}

// ---------------------------------------------------------------------
// Pure structural helpers (no value or type lookup)
// ---------------------------------------------------------------------

/**
 * Collect axis names that appear in `exprIR`, NOT descending into
 * nested aggregate(...) bodies (each aggregate has its own closed
 * axis scope per spec §05).
 *
 * Order is depth-first first-occurrence — used as the canonical
 * ordering of axes in body so the post-output reduce-axes block has a
 * deterministic order across re-annotations.
 */
function collectAxesInScope(exprIR: any): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  function walk(n: any): void {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.kind === 'axis') {
      if (!seen.has(n.name)) { seen.add(n.name); ordered.push(n.name); }
      return;
    }
    if (n.kind === 'call' && n.op === 'aggregate') return;   // inner scope
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op'
          || k === 'name' || k === 'ns') continue;
      walk(n[k]);
    }
  }
  walk(exprIR);
  return ordered;
}

/**
 * Extract the (outAxes, reduceAxes, canonicalAxes) skeleton from an
 * aggregate IR node. Pure structural — no value or type lookup.
 *
 * Returns null when the IR shape isn't a recognised aggregate (wrong
 * arg count, output_axes not an array literal, etc.); callers fall
 * back to whatever defensive path makes sense.
 */
function structuralCanonicalAxes(ir: any): {
  outAxes: string[];
  reduceAxes: string[];
  canonicalAxes: string[];
} | null {
  if (!ir || ir.kind !== 'call' || ir.op !== 'aggregate') return null;
  const args = ir.args || [];
  if (args.length !== 3) return null;
  const axesIR = args[1];
  if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
  const outAxes: string[] = [];
  for (const a of axesIR.args || []) {
    if (!a || a.kind !== 'axis') return null;
    outAxes.push(a.name);
  }
  const allAxes = collectAxesInScope(args[2]);
  const outSet = new Set(outAxes);
  const reduceAxes: string[] = [];
  for (const a of allAxes) if (!outSet.has(a)) reduceAxes.push(a);
  return { outAxes, reduceAxes, canonicalAxes: [...outAxes, ...reduceAxes] };
}

// ---------------------------------------------------------------------
// Annotation: typeinfer populates, runtime consumes
// ---------------------------------------------------------------------

/**
 * Return the canonical form for an aggregate IR node, computing and
 * caching it on `ir.meta.aggregateCanonical` if not already present.
 *
 * `staticLengths` is the typeinfer-resolved lengths map (may have
 * `%dynamic` entries) — passed only when typeinfer is the caller; the
 * runtime gets undefined and builds the structural skeleton with no
 * static lengths.
 *
 * Returns null when the IR doesn't structurally match an aggregate
 * (wrong arg count, etc.); callers should fall back defensively.
 */
function annotate(
  ir: any,
  staticLengths?: Record<string, number | '%dynamic'>,
): AggregateCanonical | null {
  // Re-annotate if the IR's args have been swapped under us (the
  // dissolver rewrites aggregates) — detect via a quick check that
  // outAxes/reduceAxes still match the IR's current shape.
  const cached: AggregateCanonical | undefined = ir.meta && ir.meta.aggregateCanonical;
  const skel = structuralCanonicalAxes(ir);
  if (!skel) return null;
  if (cached) {
    let match = cached.outAxes.length === skel.outAxes.length
                && cached.reduceAxes.length === skel.reduceAxes.length;
    if (match) {
      for (let i = 0; i < skel.outAxes.length && match; i++) {
        if (cached.outAxes[i] !== skel.outAxes[i]) match = false;
      }
      for (let i = 0; i < skel.reduceAxes.length && match; i++) {
        if (cached.reduceAxes[i] !== skel.reduceAxes[i]) match = false;
      }
    }
    if (match) {
      // Same shape — fold in any new static lengths the caller learnt.
      if (staticLengths) {
        for (const a of Object.keys(staticLengths)) {
          if (!(a in cached.axisLengths)) {
            cached.axisLengths[a] = staticLengths[a];
          }
        }
        cached.fullyResolved = skel.canonicalAxes.every(
          (a) => typeof cached.axisLengths[a] === 'number');
      }
      return cached;
    }
    // Shape drifted — fall through and recompute.
  }
  const axisLengths: Record<string, number | '%dynamic'> = {};
  if (staticLengths) {
    for (const a of Object.keys(staticLengths)) axisLengths[a] = staticLengths[a];
  }
  const canonical: AggregateCanonical = {
    outAxes:       skel.outAxes,
    reduceAxes:    skel.reduceAxes,
    canonicalAxes: skel.canonicalAxes,
    axisLengths,
    fullyResolved: skel.canonicalAxes.every(
      (a) => typeof axisLengths[a] === 'number'),
  };
  if (!ir.meta) ir.meta = {};
  ir.meta.aggregateCanonical = canonical;
  return canonical;
}

/**
 * Read-only accessor: returns the cached annotation if present, else
 * computes the structural skeleton (no static lengths). Use this when
 * the caller doesn't carry a static-length resolver — the runtime
 * `resolveAxisLengths` will fill in the missing entries.
 */
function getCanonical(ir: any): AggregateCanonical | null {
  return annotate(ir);
}

/**
 * Resolve any `%dynamic` or absent axis lengths in `canonical` by
 * consulting the runtime body via `lengthOfAxisInBody(axisName)`. The
 * resolver is supplied by the caller — single-point runtime uses
 * `evaluateExpr` on `get(...)` containers, atom-batched runtime uses
 * a similar shape inspection over `refArrays`. Mutates the canonical
 * form's axisLengths in place when resolution succeeds; returns the
 * fully-resolved lengths map.
 *
 * Throws a clear diagnostic on any axis that can't be resolved.
 */
function resolveAxisLengths(
  ir: any,
  canonical: AggregateCanonical,
  lengthOfAxisInBody: (axisName: string) => number | null,
): Record<string, number> {
  const out: Record<string, number> = {};
  const missing: string[] = [];
  for (const axisName of canonical.canonicalAxes) {
    let v = canonical.axisLengths[axisName];
    if (typeof v !== 'number') {
      const resolved = lengthOfAxisInBody(axisName);
      if (resolved == null) { missing.push(axisName); continue; }
      v = resolved;
      canonical.axisLengths[axisName] = v;
    }
    out[axisName] = v;
  }
  if (missing.length > 0) {
    throw new Error(
      `aggregate: could not infer length of axis `
      + `${missing.map((a) => '.' + a).join(', ')} — each axis must `
      + `index a known array at least once in expr`);
  }
  canonical.fullyResolved = true;
  return out;
}

module.exports = {
  collectAxesInScope,
  structuralCanonicalAxes,
  annotate,
  getCanonical,
  resolveAxisLengths,
};
