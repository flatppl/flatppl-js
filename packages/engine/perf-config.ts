'use strict';

// Optimization registry — toggles for accelerated / specialised
// dispatch paths that the engine can take instead of a general
// (reference) interpreter. Every entry's default is `true` (use the
// specialised path); flipping it to `false` forces the general path
// for testing equivalence and isolating regressions.
//
// =====================================================================
// Why
// =====================================================================
//
// Several engine subsystems compose a "general / reference"
// implementation with one or more "specialised / fast" paths:
//
//   - aggregate         : nested-loop interpreter + AGGREGATE_PATTERNS
//                         specialisers (matmul → BLAS-style mul, …)
//   - disintegrate      : `delegate` plan (reuse existing bindings) +
//                         `synthesize` plan (emit fresh AST)
//   - select / mixture  : closed-form `logsumexp` + MC-estimated
//                         weights (resolveRuntimeWeights)
//   - density walkers   : closed-form per-distribution + density.ts
//                         MC density estimator
//
// The specialised paths are correctness-equivalent to the general
// ones by design, but specialisers are subtle and divergences are
// real bugs (LLVM, V8, JAX, Stan all maintain "reference mode" flags
// for exactly this reason).
//
// Per-op toggles let us:
//
//   1. Run every equivalence test under BOTH settings. If a future
//      specialiser quietly diverges from the general path, the
//      doubled test catches it.
//
//   2. Bisect regressions in production: a divergence in `densityof`
//      output can be quickly localised by toggling one optimisation
//      at a time.
//
//   3. Profile honestly: the general path's cost is observable by
//      flipping a flag, not by deleting code.
//
// =====================================================================
// Adding a new toggle
// =====================================================================
//
// When you introduce a specialised path:
//
//   1. Add a key here with default `true`.
//   2. Wrap the specialised dispatch in a `getOptimization('<key>')`
//      check; fall through to the general path when disabled.
//   3. Use the `inBothModes(name, optKey, fn)` helper from
//      `test/_perf-helpers.ts` to register tests that should run
//      under both settings.
//   4. Mention the toggle in `flatppl-engine-concepts.md` §15
//      (Optimization toggles & dual-mode testing) so future
//      contributors know the pattern.
//
// Globally-scoped registry, mutated by setOptimization. Per-context
// scoping would be cleaner but the engine doesn't thread context
// through every walker today; a single global registry keeps the
// surface minimal. If user code ever needs per-instance control,
// lift to a context option.

const OPTIMIZATIONS: Record<string, boolean> = {
  // aggregate(...) pattern dispatch — matmul, dot-product, outer-
  // product, etc. specialisers in sampler.AGGREGATE_PATTERNS. When
  // disabled, every aggregate falls through to the nested-loop
  // interpreter (`_evalAggregateGeneral`).
  'aggregate': true,

  // disintegrate's `delegate` plan — reuses existing bindings when
  // both sides of a structural disintegration are already named
  // bindings. When disabled, the `synthesize` plan emits fresh AST
  // instead. Both are semantically equivalent per spec §06
  // (`jointchain(prior, kernel) ≡ joint`); they differ in the
  // rendered surface form.
  'disintegrate.delegate': true,
};

/**
 * Read the current value of an optimization toggle. Unknown keys
 * default to `true` (specialiser enabled) so a typo doesn't
 * silently disable a path; the keys catalog above is the source of
 * truth.
 */
function getOptimization(name: string): boolean {
  return OPTIMIZATIONS[name] !== false;
}

/**
 * Set a single optimization toggle. Returns the previous value, so
 * callers (e.g. the test helper) can restore state cleanly.
 *
 * Unknown keys are accepted and stored — this lets tests register
 * new toggles before the corresponding engine code lands, without
 * the engine module having to be edited first.
 */
function setOptimization(name: string, enabled: boolean): boolean {
  const prev = getOptimization(name);
  OPTIMIZATIONS[name] = !!enabled;
  return prev;
}

/**
 * Snapshot all toggles → a plain object. Useful in `before`/`after`
 * hooks that want to restore the entire registry.
 */
function snapshotOptimizations(): Record<string, boolean> {
  return { ...OPTIMIZATIONS };
}

/**
 * Restore from a snapshot. Keys not in the snapshot keep their
 * current value (additive merge). Use after a `before`-style
 * snapshot to put everything back in `after`.
 */
function restoreOptimizations(snapshot: Record<string, boolean>) {
  for (const k of Object.keys(snapshot)) OPTIMIZATIONS[k] = snapshot[k];
}

/**
 * Convenience: set multiple toggles at once. Useful in tests that
 * want to disable a set of accelerations together.
 */
function setOptimizations(updates: Record<string, boolean>) {
  for (const k of Object.keys(updates)) OPTIMIZATIONS[k] = !!updates[k];
}

/**
 * The list of registered keys. Adding new ones here keeps the
 * catalog discoverable; the engine code that consults each key is
 * the source of truth for behavior.
 */
function listOptimizations(): string[] {
  return Object.keys(OPTIMIZATIONS);
}

module.exports = {
  getOptimization,
  setOptimization,
  setOptimizations,
  snapshotOptimizations,
  restoreOptimizations,
  listOptimizations,
};
