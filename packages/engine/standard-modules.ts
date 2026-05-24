'use strict';

// =====================================================================
// Standard-module registry (spec §09)
// =====================================================================
//
// `standard_module(name, compat)` in surface FlatPPL resolves to a
// registry of engine-provided synthetic modules. Unlike `load_module`,
// which compiles user-authored `.flatppl` text, a standard module is
// implemented in JS — its bindings are engine functions exposed through
// a synthetic LoweredModule-shaped descriptor.
//
// Architectural decisions (TODO-flatppl-js.md "Multi-file models",
// locked 2026-05-10):
//   - Standard modules are engine-provided. The standard-module
//     category exists for capabilities that need engine support
//     (performance, primitives not expressible in base FlatPPL).
//   - Each entry is indexed by (name, compat). compat follows the same
//     two-part grammar as `flatppl_compat` (e.g. "0.1"). When multiple
//     versions are registered, lookup matches by major.minor compat.
//
// This file lands the REGISTRY SKELETON. Initial registry is empty;
// each standard module is added in its own follow-up commit (see TODO
// §09 — particle-physics, generalized-linear-models, ext-linear-algebra,
// special-functions, polynomials, distances).

// One entry per registered (name, compat) pair.
//
//   name        — module identifier (e.g. 'particle-physics')
//   compat      — version compat string (e.g. '0.1')
//   bindings    — Map<binding-name, BindingDescriptor>
//                 where BindingDescriptor is:
//                   { kind: 'function', impl: (...args) => any, sig?: Signature }
//                   { kind: 'distribution', /* TBD: spec for distribution refs */ }
//                   { kind: 'value', value: any }
//                 The classifier exposes the binding name through the
//                 standard-module's `get_field` access; runtime
//                 dispatch reads the descriptor's `impl`.
interface StandardModuleEntry {
  name: string;
  compat: string;
  bindings: Map<string, BindingDescriptor>;
}

interface BindingDescriptor {
  kind: 'function' | 'distribution' | 'value';
  /** For 'function': the JS callable. Surface form `mod.foo(args)`
   *  dispatches through this. */
  impl?: (...args: any[]) => any;
  /** For 'value': a precomputed JS value (number / Value / record). */
  value?: any;
  /** Optional type signature in the engine's types.ts SIGNATURE_FACTORIES
   *  shape. When omitted, typeinfer falls back to `deferred`. */
  sig?: any;
}

// The registry. Indexed by `${name}@${compat}` — exact-match for now;
// loose semver matching is deferred.
const _REGISTRY = new Map<string, StandardModuleEntry>();

/**
 * Look up a standard module by name + compat. Returns null when no
 * registered module matches.
 */
function lookupStandardModule(name: string, compat: string): StandardModuleEntry | null {
  const key = `${name}@${compat}`;
  return _REGISTRY.get(key) || null;
}

/**
 * Register a standard module. Called once at engine boot per module;
 * subsequent calls for the same (name, compat) replace the previous
 * entry (useful for tests).
 */
function registerStandardModule(entry: StandardModuleEntry): void {
  const key = `${entry.name}@${entry.compat}`;
  _REGISTRY.set(key, entry);
}

/**
 * List all registered standard modules. Used by the viewer's module
 * navigator and by introspection tests.
 */
function listStandardModules(): Array<{ name: string; compat: string }> {
  const out: Array<{ name: string; compat: string }> = [];
  for (const entry of _REGISTRY.values()) {
    out.push({ name: entry.name, compat: entry.compat });
  }
  return out;
}

/**
 * Test-only: clear all registered modules. Used by test setup to
 * avoid cross-test pollution.
 */
function _clearStandardModules(): void {
  _REGISTRY.clear();
}

module.exports = {
  lookupStandardModule,
  registerStandardModule,
  listStandardModules,
  _clearStandardModules,
};
