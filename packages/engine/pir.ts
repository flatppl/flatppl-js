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
function loweredModule(opts?: any): {
  bindings: Map<string, any>;
  publicSet: Set<string>;
  source: any;
  moduleRegistry?: Record<string, any>;
} {
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
    // Optional doc-comment attached by the parser (spec §05 / §04
    // §sec:documentation). Shape: `{ markup: 'md'|'typ', lines: string[] }`.
    // Null when the source binding had no doc-comment. Lowered to the
    // optional `(%doc ...)` sub-form of `(%bind ...)` in FlatPIR.
    doc: opts.doc || null,
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
  // Module-typed bindings (`load_module(...)` / `standard_module(...)` —
  // analyzer's classifier sets `binding.type === 'module'`). The lower
  // pass consults this set when it sees `<name>.field`: if `<name>` is
  // module-typed, emit the spec §11 cross-module ref `(%ref name field)`
  // rather than a data-access `get_field(name, "field")`.
  const moduleNames = new Set<string>();
  for (const [name, binding] of parsedBindings) {
    if (binding && binding.type === 'module') moduleNames.add(name);
  }
  // All module binding names: lets `_lowerReification` distinguish an
  // identifier-form boundary kwarg that designates a real node (a cut —
  // body refs stay `self`, spec §11) from one naming nothing (a pure
  // formal — placeholder semantics, the spec §04 lambda rule).
  const bindingNames = new Set<string>(parsedBindings.keys());
  const lowerCtx = { localScope: null, moduleNames, bindingNames };

  // Module registry: alias → resolved descriptor for downstream
  // consumers (typeinfer / materialiser / sampler) that need to
  // dispatch through `(%ref <alias> X)` refs without re-parsing the
  // `standard_module(...)` / `load_module(...)` IR each time. Filled
  // post-lowerExpr so the binding's IR already exists. For now we
  // only populate standard_module entries; load_module support
  // follows the multi-file end-to-end work.
  const moduleRegistry: Record<string, any> = {};
  m.moduleRegistry = moduleRegistry;
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
      rhs = lower.lowerExpr(sourceAst, lowerCtx);
    } catch (err) {
      // Lowering failure (malformed AST) — record a placeholder
      // synthetic literal so downstream passes don't crash. The
      // analyzer will already have flagged the underlying parse error.
      rhs = { kind: 'lit', value: null, loc: binding.node.loc, lowerError: String(err) };
    }
    m.bindings.set(name, loweredBinding(name, rhs, {
      doc: binding.node && binding.node.doc ? binding.node.doc : null,
      originLoc: binding.node.loc,
      synthetic: !!binding.synthetic,
    }));
    // Module-typed binding: extract the (stdName, stdCompat) from its
    // lowered `standard_module(<name>, <compat>)` call so downstream
    // consumers can resolve `(%ref <alias> X)` without re-parsing.
    if (binding.type === 'module' && rhs && rhs.kind === 'call'
        && rhs.op === 'standard_module'
        && Array.isArray(rhs.args) && rhs.args.length === 2
        && rhs.args[0].kind === 'lit' && rhs.args[1].kind === 'lit') {
      moduleRegistry[name] = {
        kind: 'standard',
        stdName: rhs.args[0].value,
        stdCompat: rhs.args[1].value,
      };
    }
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

/**
 * Walk every call in `expr` (depth-first, post-order). Visitor receives
 * the call node; return value is ignored. Used by inference passes that
 * need to annotate every call with meta, not just the outermost.
 *
 * Delegates IR-position enumeration to `ir-walk.walkIR` (single source
 * of truth for which fields carry recursive IR children: args / kwargs
 * / fields / body / branches / selector / logweights / assigns). Filters
 * to call nodes here.
 */
const { walkIR } = require('./ir-walk.ts');

function walkCalls(expr: any, visit: (e: any) => void) {
  walkIR(expr, (n: any) => { if (n && n.kind === 'call') visit(n); });
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
  loweredModule,
  loweredBinding,
  lowerToModule,
  walkCalls, setMeta,
};
