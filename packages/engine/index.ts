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
const value = require('./value.ts');
const types = require('./types.ts');
const typeinfer = require('./typeinfer.ts');
const pir = require('./pir.ts');
const pirSexpr = require('./pir-sexpr.ts');
const perfConfig = require('./perf-config.ts');
// NOTE: ./sampler and ./worker are NOT re-exported here. They pull in
// stdlib's distribution packages (~1 MB after bundling) and are intended
// for the sampler-worker bundle only. Main-thread / extension-host code
// that needs to drive sampling should send messages to the worker over
// its postMessage protocol — see engine/worker.js for the protocol and
// vscode-extension/lib/sampler-worker.min.js for the bundled worker.

/**
 * Parse and analyze a FlatPPL source text in one call.
 *
 * @param {string} source — surface source
 * @param {object} [opts]
 *   - opts.variant — variant id string ('flatppl' | 'flatppy' | 'flatppj')
 *                    or a variant object from ./variants. Wins over path.
 *   - opts.path    — source path; the extension picks the variant when
 *                    opts.variant is absent. Defaults to FlatPPL when
 *                    nothing else is supplied.
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
  const { tokens, diagnostics: tokenDiags } = tokenize(source, variant);
  const { ast, diagnostics: parseDiags } = parse(tokens, variant);
  const { bindings, loweredModule, diagnostics: analyzeDiags, symbols }
    = analyze(ast, source);

  const diagnostics = [...tokenDiags, ...parseDiags, ...analyzeDiags];

  // loweredModule is forwarded for downstream consumers that need
  // on-demand type specialization (e.g. typeinfer.inferExprInScope
  // used by the plot dispatcher to compute the output shape of a
  // polymorphic function at a specific call site — module-level
  // inference produces best-effort with `any` inputs, which under-
  // specifies in general).
  return { ast, bindings, loweredModule, symbols, diagnostics, variant, bundle };
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
  value,
  types, typeinfer, pir, pirSexpr,
  // Optimization toggles for dual-mode testing (engine-concepts §15).
  perfConfig,
};
