'use strict';

// =====================================================================
// axis-stack.ts — consumer-side API for the axisStack IR annotation
// =====================================================================
//
// **Background.** `dissolver.propagateAxisStack` (engine-concepts §18.11
// / §20.10.5 item 4) annotates measure-op IR nodes with `axisStack`
// metadata recording the outer iteration axes their variate carries
// from enclosing axis-introducing constructs (iid / kernel-broadcast /
// aggregate / value broadcast). The annotation is PURE METADATA — it
// doesn't change IR structure or runtime semantics.
//
// The annotation was first populated and read advisorily; it is now
// **authoritative** for the axis-introducing / axis-preserving shapes
// the producer covers (iid / kernel_broadcast / aggregate + the
// preserving wrappers — see `_axisStackForIR` in dissolver.ts). The
// producer's coverage boundary is principled, not a gap: composites
// whose variate shape isn't a single inner's (superpose / select /
// joint / kchain) deliberately carry NO stack.
//
//   - `getAxisStack(ir)` reads `ir.axisStack` and returns it (or null).
//   - `bindingAxisStack(name, ctx)` looks up the binding by name in
//     `ctx.bindings` and returns its IR's axisStack.
//   - Higher-level helpers `outerAxisSize(stack, sourceTag)` extract a
//     specific axis's size for the common consumer patterns.
//
// **Authoritative contract.** When a literal-size axis entry is
// present, consumers TRUST it: a runtime disagreement is producer/
// runtime drift (engine invariant violation), not a user error.
// Consumers:
//   - matIid (materialiser.ts) — the iid axis is the **repeat axis**:
//     iid(M, k) over a composite M holds M's atom-level params constant
//     across the k inner draws (spec §06 within-atom conditional
//     independence). The size comes from the derivation's `dims`, which
//     the producer also records as the `iid` stack entry.
//   - mat-broadcast (mat-broadcast.ts) — trusts the kernel_broadcast
//     axis K; the per-param ladder validates against it.
// Only SYMBOLIC sizes ('%dynamic' / a binding-ref name — dynamic-D,
// ragged) fall back to runtime shape-sniffing; `outerAxisSize` returns
// null for those, signalling the fallback.
//
// **PoissonProcess (ragged arrays) opt out either way.** Ragged
// per-atom-length arrays have no statically-known axis size, so they
// don't carry axisStack and consumers don't consult it for them.
// The ragged-storage path (when implemented per TODO §08) handles
// these cases separately.

interface AxisStackEntry {
  source: 'iid' | 'broadcast' | 'kernel_broadcast' | 'aggregate';
  size: number | string;        // integer literal | binding-ref name | '%dynamic'
  name?: string;                // aggregate output-axis name
}

/**
 * Read the axisStack annotation from a measure-op IR node. Returns
 * the stack (most-outer first) when present and non-empty, else null.
 *
 * Callers should treat null as "no advisory info; fall back to
 * runtime shape inspection".
 */
function getAxisStack(ir: any): AxisStackEntry[] | null {
  if (!ir || !Array.isArray(ir.axisStack) || ir.axisStack.length === 0) {
    return null;
  }
  return ir.axisStack as AxisStackEntry[];
}

/**
 * Look up the axisStack for a binding's RHS IR by name. Reads from
 * `ctx.bindings.get(name).ir.axisStack` (the materialiser ctx shape
 * already exposes this). Returns null if the binding doesn't carry
 * axisStack metadata.
 *
 * `ctx` must expose `bindings.get(name)`; passing an object without
 * `bindings` returns null defensively (advisory mode — consumers
 * gracefully fall back).
 */
function bindingAxisStack(name: string, ctx: any): AxisStackEntry[] | null {
  if (!ctx || !ctx.bindings || typeof ctx.bindings.get !== 'function') {
    return null;
  }
  const b = ctx.bindings.get(name);
  if (!b || !b.ir) return null;
  return getAxisStack(b.ir);
}

/**
 * Extract the size of the FIRST axis-stack entry whose `source`
 * matches `sourceTag`. Returns the size when it's an integer literal;
 * returns null when the size is symbolic ('%dynamic' or a binding-ref
 * name) — consumers in that case must resolve via the env, same as
 * today's runtime path.
 */
function outerAxisSize(
  stack: AxisStackEntry[] | null,
  sourceTag: AxisStackEntry['source'],
): number | null {
  if (!stack) return null;
  for (const e of stack) {
    if (e.source === sourceTag) {
      return typeof e.size === 'number' ? e.size : null;
    }
  }
  return null;
}

/**
 * Whether the axisStack signals an iid axis at the OUTERMOST position.
 * Used by consumers (e.g. matIid composite fallback) to detect the
 * per-atom-shared-prior structure that today's inflated-count fallback
 * collapses (matIid documented semantic gap, materialiser.ts:382-397).
 *
 * Returns the iid axis size (integer) when known, or 'dynamic' when
 * the size is symbolic but the axis is structurally present.
 */
function outermostIidAxis(
  stack: AxisStackEntry[] | null,
): number | 'dynamic' | null {
  if (!stack || stack.length === 0) return null;
  const e = stack[0];
  if (e.source !== 'iid') return null;
  return typeof e.size === 'number' ? e.size : 'dynamic';
}

module.exports = {
  getAxisStack,
  bindingAxisStack,
  outerAxisSize,
  outermostIidAxis,
};
