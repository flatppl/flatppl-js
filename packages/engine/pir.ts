'use strict';

// FlatPIR-aligned in-memory representation. Mirrors the spec at
// flatppl-design/docs/11-flatpir.md, with engine-internal extensions
// (back-refs to source AST, per-call meta slots) attached as side data
// rather than baked into the spec-defined nodes.
//
// Why this module exists
// ======================
// `lower.js` already produces FlatPIR-JSON expressions per binding RHS
// — the right shape for individual expressions. What it doesn't give
// us is a *module-level* container that holds all the bindings and
// their relationships in one place. Without that, every consumer
// (analyzer, orchestrator, visualizer) has to re-lower per binding
// and reconstruct the module structure ad-hoc.
//
// LoweredModule fixes that: lowering happens once, the result is
// a self-contained FlatPIR module, and every subsequent pass
// (type inference, phase inference, scope materialisation,
// derivation building) operates on it as the single source of truth
// for the program's executable structure.
//
// What's spec-defined here
// ========================
//   * %module structure with %public and ordered %bind list
//   * %bind's RHS as a LoweredExpr (lit / ref / call / hole / const)
//   * %ref namespaces: 'self' / '%local' / <module-alias>
//   * %meta slot on every call (placeholder %deferred until inferred)
//   * Type categories from FlatPIR (scalar/array/record/tuple/measure/…)
//
// What's engine-internal
// ======================
//   * `originLoc` and `originName` back-refs from lowered nodes to the
//     source AST they came from — used for diagnostics and DAG display.
//   * The `LoweredModule.source` reference pointing back at the
//     ParsedModule (analyzer's bindings + AST) so consumers that need
//     surface-level info (truncated rhs strings, original kernelof
//     keyword for display) can fetch it without duplicating data.
//   * Scope materialisation metadata stored on functionof/kernelof
//     bindings (synthesised scoped binding map for the boundary-
//     substitution case). See `materializeScopes`.
//
// What's NOT here yet
// ===================
//   * The actual `lowerToModule(ast)` orchestration is in this file
//     but it's a thin wrapper today — it just walks each binding,
//     calls lower.js, and assembles. Type/phase inference and scope
//     materialisation are separate passes that run *after* lowering.
//   * S-expression printer / reader. Future work; would let us dump
//     a `.flatpir` file for inspection.

const lower = require('./lower.ts');

// =====================================================================
// Constructors
// =====================================================================

/**
 * Construct an empty LoweredModule. Bindings are added in source order;
 * `publicSet` is the export list (names not starting with underscore by
 * default; explicit `%public` declaration overrides this in future).
 *
 * @param {object} [opts]
 * @param {object} [opts.source] - the ParsedModule (analyzer output) this
 *                                 was lowered from. Optional only for
 *                                 unit tests; production code always sets it.
 */
function loweredModule(opts?: any) {
  opts = opts || {};
  return {
    bindings: new Map(),       // name → LoweredBinding (insertion-ordered)
    publicSet: new Set(),      // export list; populated during lowering
    source: opts.source || null,
  };
}

/**
 * Construct a LoweredBinding. `rhs` is a LoweredExpr (the FlatPIR-JSON
 * shape produced by lower.js). `originLoc` is the AST source location
 * for diagnostics; `synthetic: true` marks bindings the engine itself
 * produced (lifted inline subexpressions, scope-materialised copies)
 * so DAG display can render them differently from user bindings.
 */
function loweredBinding(name: string, rhs: any, opts?: any) {
  opts = opts || {};
  return {
    name,
    rhs,
    originLoc: opts.originLoc || (rhs && rhs.loc) || null,
    originName: opts.originName || null,    // for scope-copied bindings
    synthetic: !!opts.synthetic,
    // Per-binding type/phase. The canonical place is `rhs.meta` for call
    // RHSes; this duplicates them at the binding level for fast lookup.
    inferredType: opts.inferredType || null,
    phase: opts.phase || null,
  };
}

// =====================================================================
// Module-level lowering
// =====================================================================

/**
 * Build a LoweredModule from the analyzer's bindings map.
 *
 * Pure: doesn't mutate `parsedBindings`. The result is a standalone
 * structure that consumers can mutate freely (e.g. type inference
 * writes meta back).
 *
 * @param {Map} parsedBindings - analyzer output, Map<name, BindingInfo>.
 * @returns {object} a LoweredModule.
 */
function lowerToModule(parsedBindings: Map<string, any>) {
  const m = loweredModule({ source: parsedBindings });
  for (const [name, binding] of parsedBindings) {
    if (!binding.node || !binding.node.value) continue;
    // Multi-LHS bindings (`a, b = rand(...)`) and disintegrate-rewritten
    // bindings expose an `effectiveValue` AST that's the per-name
    // projection — the analyzer attaches it during the multi-LHS pass.
    // Lowering uses effectiveValue when present so each name's IR is
    // its own projection, not a shared tuple call. Falls back to
    // node.value for ordinary single-LHS bindings.
    const sourceAst = binding.effectiveValue || binding.node.value;
    let rhs;
    try {
      rhs = lower.lowerExpr(sourceAst);
    } catch (err) {
      // Lowering failure (malformed AST) — record a placeholder
      // synthetic literal so downstream passes don't crash. The
      // analyzer will already have flagged the underlying parse error.
      rhs = { kind: 'lit', value: null, loc: binding.node.loc, lowerError: String(err) };
    }
    m.bindings.set(name, loweredBinding(name, rhs, {
      originLoc: binding.node.loc,
      synthetic: !!binding.synthetic,
    }));
    // Public-by-default: any name not starting with underscore. The
    // analyzer marks engine-internal multi-LHS shared bindings with
    // a `%mlhs:` prefix to keep them out of the public set.
    if (!name.startsWith('_') && !name.startsWith('%')) m.publicSet.add(name);
  }
  return m;
}

// =====================================================================
// Read helpers
// =====================================================================

/** Whether `expr` is a call to a built-in op named `op`. */
function isBuiltinCall(expr: any, op: string) {
  return expr && expr.kind === 'call' && expr.op === op;
}

/** Whether `expr` is a reference. */
function isRef(expr: any) {
  return expr && expr.kind === 'ref';
}

/**
 * Walk every call in `expr` (depth-first, post-order). Visitor receives
 * the call node; return value is ignored. Used by inference passes that
 * need to annotate every call with meta, not just the outermost.
 */
function walkCalls(expr: any, visit: (e: any) => void) {
  if (!expr || typeof expr !== 'object') return;
  if (expr.kind === 'call') {
    if (expr.args)   for (const a of expr.args) walkCalls(a, visit);
    if (expr.kwargs) for (const k in expr.kwargs) walkCalls(expr.kwargs[k], visit);
    if (expr.fields) for (const f of expr.fields) walkCalls(f.value, visit);
    if (expr.body)   walkCalls(expr.body, visit);
    visit(expr);
  }
}

/**
 * Set the meta annotation on a call expression. Allocates `expr.meta`
 * if not already present. Mirrors FlatPIR's (%meta type phase) where
 * each slot may be %deferred, a concrete value, or (%failed reason).
 */
function setMeta(expr: any, type: any, phase: any) {
  if (!expr.meta) expr.meta = {};
  if (type  != null) expr.meta.type  = type;
  if (phase != null) expr.meta.phase = phase;
}

module.exports = {
  loweredModule, loweredBinding,
  lowerToModule,
  isBuiltinCall, isRef, walkCalls, setMeta,
};
