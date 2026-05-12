'use strict';

const variants = require('./variants');
const { tokenize, T } = require('./tokenizer');
const { parse } = require('./parser');
const { analyze, collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges, computePhases } = require('./analyzer');
const { computeSubDAG, computeFullDAG, findBindingAtLine } = require('./dag');
const disintegrate = require('./disintegrate');
const AST = require('./ast');
const builtins = require('./builtins');
const rng = require('./rng');
const lower = require('./lower');
const orchestrator = require('./orchestrator');
const histogram = require('./histogram');
const empirical = require('./empirical');
const types = require('./types');
const typeinfer = require('./typeinfer');
const pir = require('./pir');
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
 * Returns { ast, bindings, symbols, diagnostics, variant }.
 */
function processSource(source, opts) {
  const variant = variants.resolveVariant(opts);
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
  return { ast, bindings, loweredModule, symbols, diagnostics, variant };
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
  rng, lower, orchestrator, histogram, empirical,
  types, typeinfer, pir,
};
