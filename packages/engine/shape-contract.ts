'use strict';

// =====================================================================
// shape-contract.ts — the single source of truth for structural shape
// =====================================================================
//
// THE PROBLEM this module exists to eliminate ("shape/representation drift").
// Three engine layers each answer "what structural shape does construct X
// produce?" independently, and they DRIFT:
//
//   - TYPE      (typeinfer.ts)   — the inferred structural Type.
//   - LAYOUT    (materialiser)   — how the empirical measure stores it
//                                  (a flat array vs a record-of-columns vs a
//                                  tuple-of-elems; AoS-per-draw vs SoA-storage).
//   - CONSUME   (density.ts)     — how the density walker splits a variate
//                                  value back into per-component footprints.
//
// When these three disagree, the type advertises one shape while the sampler
// draws another and the density scores a third — silent `type ≠ runtime` bugs
// (positional `joint`, vector-per-entry table columns, cartprod, …). The fix
// is ONE owner per structural rule that emits a representation-neutral
// `ShapeSpec`, which each layer PROJECTS into its own view:
//
//         shape rule ──► ShapeSpec ──► typeOfShape        (type)
//                                 ├──► layoutOfShape       (materialiser)
//                                 └──► consumeOrderOfShape (density)
//
// so the three layers STOP re-deriving and can no longer drift.
//
// STAGING. This is the seed of a multi-stage migration (see
// flatppl-dev/TODO-flatppl-js.md "Shape-contract unification"). Stage S0 —
// this file — owns the CAT-SHAPE FAMILY and the TYPE projection:
//
//   - `catShape`      — the one owner of the cat rule shared by `cat` (values,
//                       spec §07), positional `cartprod` (sets, §03), and
//                       positional `joint` (variates, §06). Previously a
//                       nested `catShapeType` helper inside typeinfer,
//                       unreachable from the materialiser/density layers
//                       (which must NOT import typeinfer). Living here — a
//                       leaf module depending only on `types.ts` — it is the
//                       neutral home all three layers can consult.
//   - `typeOfShape`   — the TYPE projection (typeinfer's view). Landed.
//
// The `layoutOfShape` / `consumeOrderOfShape` projections land alongside their
// materialiser / density consumers in later stages (that is when the ShapeSpec
// `components` / field-order metadata carried here starts being read); adding
// them before a consumer would be dead scaffolding. The metadata is captured
// now so the spec need not be reshaped when those stages arrive.
//
// This module is a dependency-free leaf apart from `types.ts` (the type
// constructors + `unifyArith`), keeping it cycle-safe to import from
// typeinfer, materialiser, and density alike.

const T = require('./types.ts');

// ---------------------------------------------------------------------
// ShapeSpec — a representation-neutral description of a structural shape.
// ---------------------------------------------------------------------
//
// For the cat-shape family a spec is one of:
//
//   { kind: 'deferred' }
//       Under-resolved — at least one component is `deferred`/`any`, so the
//       cat result cannot be committed yet.
//
//   { kind: 'catArray', elem, length, components }
//       A flat rank-1 array: all-scalar components (each `footprint: 1`,
//       `length = #components`) or all rank-1 vectors (per-vector footprints,
//       `length = Σ footprints`). `elem` is the unified element type;
//       `length` is a positive integer or the literal '%dynamic'. The
//       `components` list carries each component's footprint width — unused by
//       `typeOfShape`, consumed by the density/materialiser projections that
//       land in later stages (a positional joint/cartprod/cat splits its flat
//       variate back into these widths).
//
//   { kind: 'mergedRecord', fields }
//       A record whose fields are the components' fields concatenated in
//       order (`fields` is an ordered array of `[name, Type]` pairs). Duplicate
//       field names are rejected upstream (`catShape` returns null).
//
// `catShape` returns `null` for a shape the cat rule does NOT permit (mixing
// structural kinds, a rank ≥ 2 array component, a duplicate record field). The
// CALLER owns the diagnostic — the shape layer stays diagnostic-free so it can
// be reused by callers with different error messages (cat vs cartprod vs
// joint).

/**
 * catShape(parts) — the shared cat-shape rule (spec §07 `cat` / §06 positional
 * `joint` / §03 positional `cartprod`).
 *
 *   - all scalars           → a vector of the unified element type
 *   - all rank-1 vectors     → a concatenated vector
 *   - all records            → a merged record (distinct field names)
 *   - any `deferred`/`any`   → `{ kind: 'deferred' }` (defer)
 *   - mixed kinds / rank ≥ 2 / duplicate field → `null` (caller diagnoses)
 *
 * `parts` are structural Types (the component variate/element/set types).
 */
function catShape(parts: any[]): any {
  if (!parts || !parts.length) return null;
  if (parts.some((t: any) => !t || t.kind === 'deferred' || t.kind === 'any')) {
    return { kind: 'deferred' };
  }
  // Leaf element join: promote two numeric leaf types to their common type
  // (bool ⊂ int ⊂ real → complex). Falls back to `deferred` if they don't
  // unify — matching the legacy helper's conservative behaviour.
  const joinLeaf = (a: any, b: any) => {
    const u = T.unifyArith(a, b, null);
    return (u && u.result) ? u.result : T.deferred();
  };

  if (parts.every((t: any) => t.kind === 'scalar')) {
    let elem = parts[0];
    for (let i = 1; i < parts.length; i++) elem = joinLeaf(elem, parts[i]);
    return {
      kind: 'catArray',
      elem,
      length: parts.length,
      components: parts.map(() => ({ footprint: 1 })),
    };
  }

  if (parts.every((t: any) => t.kind === 'array' && t.rank === 1)) {
    let elem = parts[0].elem;
    let length = 0;
    let dynamic = false;
    const components: any[] = [];
    for (const p of parts) {
      elem = joinLeaf(elem, p.elem);
      const d = p.shape && p.shape[0];
      if (typeof d === 'number') { length += d; components.push({ footprint: d }); }
      else { dynamic = true; components.push({ footprint: '%dynamic' }); }
    }
    return { kind: 'catArray', elem, length: dynamic ? '%dynamic' : length, components };
  }

  if (parts.every((t: any) => t.kind === 'record')) {
    const fields: any[] = [];
    const seen: Record<string, boolean> = Object.create(null);
    for (const p of parts) {
      for (const k in p.fields) {
        if (seen[k]) return null;   // duplicate field name across components
        seen[k] = true;
        fields.push([k, p.fields[k]]);
      }
    }
    return { kind: 'mergedRecord', fields };
  }

  return null;   // mixed structural kinds / unsupported component
}

/**
 * typeOfShape(spec) — project a ShapeSpec to a structural Type (typeinfer's
 * view). `null` in → `null` out (the caller already knows the shape was
 * rejected and has emitted its diagnostic).
 */
function typeOfShape(spec: any): any {
  if (spec == null) return null;
  switch (spec.kind) {
    case 'deferred':
      return T.deferred();
    case 'catArray':
      return T.array(1, [spec.length], spec.elem);
    case 'mergedRecord': {
      const out: Record<string, any> = {};
      for (const [k, t] of spec.fields) out[k] = t;
      return T.record(out);
    }
    default:
      return null;
  }
}

module.exports = {
  catShape,
  typeOfShape,
};
