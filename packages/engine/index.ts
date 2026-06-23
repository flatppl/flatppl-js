'use strict';

const variants = require('./variants.ts');
const { tokenize, T } = require('./tokenizer.ts');
const { parse } = require('./parser.ts');
const { analyze, collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges, computePhases } = require('./analyzer.ts');
const { computeSubDAG, computeFullDAG, findBindingAtLine } = require('./dag.ts');
const disintegrate = require('./disintegrate.ts');
const AST = require('./ast.ts');
const builtins = require('./builtins.ts');
const rng = require('./rng.ts');
const lower = require('./lower.ts');
const orchestrator = require('./orchestrator.ts');
const histogram = require('./histogram.ts');
const empirical = require('./empirical.ts');
const materialiser = require('./materialiser.ts');
const density = require('./density.ts');
const mcRecipe = require('./mc-recipe.ts');
const clm = require('./clm.ts');
const value = require('./value.ts');
const types = require('./types.ts');
const typeinfer = require('./typeinfer.ts');
const pir = require('./pir.ts');
const pirSexpr = require('./pir-sexpr.ts');
const standardModules = require('./standard-modules.ts');
const dataload = require('./dataload.ts');
const { resolveModulePath } = require('./module-resolve.ts');
const { linkModules } = require('./module-link.ts');
const perfConfig = require('./perf-config.ts');
const optimizer = require('./optimizer/index.ts');
const generatedQuantities = require('./generated-quantities.ts');
const posteriorPredictive = require('./posterior-predictive.ts');
// NOTE: ./sampler and ./worker are NOT re-exported here. The main-thread
// engine bundle (lib/engine.min.js) is kept small by stubbing @stdlib at
// build time via build-vendor.mjs + build-stdlib-stub.cjs; the sampler-
// worker bundle (lib/sampler-worker.min.js) carries the real numerics.
// Main-thread code that needs sampling must route through the worker's
// postMessage protocol — see engine/worker.ts for the protocol.

/**
 * Parse and analyze a FlatPPL source text in one call.
 *
 * @param {string} source — surface source
 * @param {object} [opts]
 *   - opts.variant — variant id string ('flatppl') or a variant
 *                    object from ./variants. Canonical FlatPPL is
 *                    the only surface form (spec §05); the seam is
 *                    retained for forward-compatibility. Wins over
 *                    path.
 *   - opts.path    — source path; reserved for future per-file
 *                    surface detection. Currently ignored.
 *   - opts.bundle  — module bundle for multi-file models (spec §04
 *                    load_module). Shape:
 *                      { sources: { [path: string]: text: string } }
 *                    The engine consumes pre-resolved .flatppl text;
 *                    async I/O is the host's responsibility (VS Code:
 *                    vscode.workspace.fs; web: fetch). Defaults to an
 *                    empty bundle when omitted. Architectural decisions
 *                    locked 2026-05-10 (see TODO-flatppl-js.md
 *                    "Multi-file models").
 * Returns { ast, bindings, symbols, diagnostics, variant, bundle }.
 */
function processSource(source: string, opts?: any) {
  const variant = variants.resolveVariant(opts);
  // Normalise the bundle to its canonical shape — every downstream
  // consumer can read `bundle.sources` without null-guarding.
  const bundle = _normaliseBundle(opts && opts.bundle);
  // The primary module's own path: the importer path against which its
  // `load_module(...)` dependencies resolve (spec §04 "Path resolution").
  // The host supplies it via `opts.path`; a standalone compile (no path)
  // resolves deps relative to the bundle root.
  const entryPath = (opts && typeof opts.path === 'string' && opts.path)
    ? opts.path : '<entry>';

  const { primary, modules } = _compileModuleGraph(source, entryPath, bundle, variant);

  // `linkedBindings` is the engine-internal FLATTENED binding map (spec
  // §11: tooling may flatten internally) — the primary plus every
  // transitively-loaded module spliced under namespaced names, with refs
  // rewritten and load-time substitutions rewired. It feeds
  // `buildDerivations` so cross-module materialisation works with the
  // unchanged by-name materialiser. For a single-file source (no
  // dependencies) it is exactly the primary's bindings — no relink. The
  // re-analysis diagnostics are dropped: the per-module diagnostics on
  // `primary`/`modules` are authoritative (the canonical, non-flattened
  // view), and the DAG uses `bindings` (the primary module alone).
  let linkedBindings = primary.bindings;
  if (modules.size > 0) {
    const linkedAst = linkModules(primary, modules);
    linkedBindings = analyze(linkedAst, '').bindings;
  }

  // loweredModule is forwarded for downstream consumers that need
  // on-demand type specialization (e.g. typeinfer.inferExprInScope used
  // by the plot dispatcher). `modules` is the registry of compiled
  // dependency modules (resolved-path → compiled module); empty for a
  // single-file source. Modules do NOT flatten in the canonical IR (spec
  // §11) — each is its own LoweredModule, reached by `(%ref <alias> X)`
  // cross-module refs; `linkedBindings` is the internal materialisation view.
  return {
    ast: primary.ast,
    bindings: primary.bindings,
    loweredModule: primary.loweredModule,
    symbols: primary.symbols,
    diagnostics: primary.diagnostics,
    variant, bundle, modules, linkedBindings, path: entryPath,
  };
}

// Compile the primary source plus, transitively, every `.flatppl` module
// it loads via `load_module(...)`. The host has already pre-resolved the
// dependency text into `bundle.sources` (keyed by resolved path); this
// stays synchronous. Dependencies compile BEFORE their importer so that
// cross-module type inference can read each dep's typed bindings.
// Resolution failures (missing source, dependency cycle, non-literal
// path) become diagnostics on the importing module — never crashes or
// infinite loops.
function _compileModuleGraph(
  entrySource: string, entryPath: string,
  bundle: { sources: Record<string, string> }, variant: any,
): { primary: any; modules: Map<string, any> } {
  // Resolved-path → compiled module. Holds the dependencies only; the
  // primary is returned separately as the top-level result.
  const modules = new Map<string, any>();

  // `stack` is the active ancestor chain (importer paths currently being
  // compiled) — used for cycle detection. A path already in `modules` is a
  // diamond dependency (compile once), distinct from a path on `stack`
  // (a cycle).
  function compile(path: string, src: string, stack: Set<string>): any {
    const { tokens, diagnostics: tokenDiags } = tokenize(src, variant);
    const { ast, diagnostics: parseDiags } = parse(tokens, variant);

    const depDiags: any[] = [];
    for (const dep of _collectLoadModuleDeps(ast)) {
      if (dep.relPath == null) {
        depDiags.push({ severity: 'error',
          message: 'load_module(...) requires a string-literal path '
            + '(module paths are resolved at load time, spec §04)',
          loc: dep.loc });
        continue;
      }
      const resolved = resolveModulePath(path, dep.relPath);
      if (stack.has(resolved)) {
        depDiags.push({ severity: 'error',
          message: "Module dependency cycle detected: '" + resolved
            + "' is already being loaded (via '" + path + "')",
          loc: dep.loc });
        continue;
      }
      if (modules.has(resolved)) continue;          // diamond dep — compile once
      const depSource = bundle.sources[resolved];
      if (depSource === undefined) {
        depDiags.push({ severity: 'error',
          message: "Module source not found in bundle: could not resolve '"
            + dep.relPath + "' (looked for '" + resolved + "')",
          loc: dep.loc });
        continue;
      }
      const childStack = new Set(stack); childStack.add(resolved);
      modules.set(resolved, compile(resolved, depSource, childStack));
    }

    const a = analyze(ast, src, { modulePath: path, modules });
    return {
      path, ast,
      bindings: a.bindings,
      loweredModule: a.loweredModule,
      symbols: a.symbols,
      diagnostics: [...tokenDiags, ...parseDiags, ...depDiags, ...a.diagnostics],
    };
  }

  const primary = compile(entryPath, entrySource, new Set([entryPath]));
  return { primary, modules };
}

// Pre-scan an AST for `load_module("path", …)` calls (spec §04). A
// `load_module` result is a module reference, not a value (it cannot be
// passed to functions or stored), so in practice it appears only as a
// binding RHS — but we walk the whole tree defensively. Returns one entry
// per call: `relPath` is the string-literal path, or null when the path
// argument is not a literal (an error, surfaced by the caller).
function _collectLoadModuleDeps(ast: any): Array<{ relPath: string | null; loc: any }> {
  const out: Array<{ relPath: string | null; loc: any }> = [];
  function visit(node: any) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const c of node) visit(c); return; }
    if (node.type === 'CallExpr' && node.callee
        && node.callee.type === 'Identifier' && node.callee.name === 'load_module') {
      const first = node.args && node.args[0];
      out.push({
        relPath: (first && first.type === 'StringLiteral') ? String(first.value) : null,
        loc: node.loc,
      });
    }
    for (const k in node) {
      if (k === 'loc') continue;
      const v = node[k];
      if (v && typeof v === 'object') visit(v);
    }
  }
  visit(ast);
  return out;
}

// Canonicalise the bundle: accept undefined / empty / partial shapes
// and always return { sources: Record<string, string> }. Subsequent
// passes can read bundle.sources directly without conditional checks.
function _normaliseBundle(b: any): { sources: Record<string, string> } {
  if (!b || typeof b !== 'object') return { sources: {} };
  const sources: Record<string, string> = {};
  if (b.sources && typeof b.sources === 'object') {
    for (const k in b.sources) {
      if (Object.prototype.hasOwnProperty.call(b.sources, k)
          && typeof b.sources[k] === 'string') {
        sources[k] = b.sources[k];
      }
    }
  }
  return { sources };
}

module.exports = {
  // High-level
  processSource,
  variants,
  // Components
  tokenize, T,
  parse,
  analyze, collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges, computePhases,
  computeSubDAG, computeFullDAG, findBindingAtLine,
  disintegrate,
  AST, builtins,
  // Lightweight sampling-stack components (no stdlib pull-in)
  rng, lower, orchestrator, histogram, empirical, materialiser, density,
  mcRecipe, clm,
  value,
  types, typeinfer, pir, pirSexpr, standardModules, dataload,
  // Optimization toggles for dual-mode testing (engine-concepts §15).
  perfConfig,
  // Plot-pane "Find maximum" optimizer (CMA-ES + FD polish → ModeFit).
  optimizer,
  // Stan-style generated quantities: samplewise evaluation over a record measure.
  generatedQuantities,
  // Posterior-predictive sampling for plain-distribution likelihoods.
  posteriorPredictive,
};
