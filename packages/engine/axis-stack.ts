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
// **Before P4 (this file)** the annotation was populated and NEVER read
// by any engine consumer. The audit (TODO-flatppl-js.md "In-flight
// P1-P9") flagged this as a documented design lever sitting idle.
//
// **P4 contract (advisory mode, user-locked 2026-05-30):**
// Consumers PREFER axisStack when present; when absent they fall back
// to today's runtime shape-sniffing. This lets P4 land incrementally
// without flag-day on the dissolver's coverage:
//
//   - `getAxisStack(ir)` reads `ir.axisStack` and returns it (or null).
//   - `bindingAxisStack(name, ctx)` looks up the binding by name in
//     `ctx.bindings` and returns its IR's axisStack.
//   - Higher-level helpers `outerAxisSize(stack, sourceTag)` extract a
//     specific axis's size for the common consumer patterns.
//
// Once propagateAxisStack coverage includes every axis-introducing
// position (the open follow-ups in dissolver.ts:1855-1880), the
// advisory contract flips to authoritative: gaps become bugs, and
// the runtime shape-sniff fallback retires.
//
// **PoissonProcess (ragged arrays) opt out either way.** Ragged
// per-atom-length arrays have no statically-known axis size, so they
// don't carry axisStack and consumers don't consult it for them.
// The ragged-storage path (when implemented per TODO §08) handles
// these cases separately.

export interface AxisStackEntry {
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
