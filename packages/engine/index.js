'use strict';

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
// NOTE: ./sampler and ./worker are NOT re-exported here. They pull in
// stdlib's distribution packages (~1 MB after bundling) and are intended
// for the sampler-worker bundle only. Main-thread / extension-host code
// that needs to drive sampling should send messages to the worker over
// its postMessage protocol — see engine/worker.js for the protocol and
// vscode-extension/lib/sampler-worker.min.js for the bundled worker.

/**
 * Parse and analyze a FlatPPL source text in one call.
 * Returns { ast, bindings, symbols, diagnostics }.
 */
function processSource(source) {
  const { tokens, diagnostics: tokenDiags } = tokenize(source);
  const { ast, diagnostics: parseDiags } = parse(tokens);
  const { bindings, diagnostics: analyzeDiags, symbols } = analyze(ast, source);

  const diagnostics = [...tokenDiags, ...parseDiags, ...analyzeDiags];

  return { ast, bindings, symbols, diagnostics };
}

module.exports = {
  // High-level
  processSource,
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
  rng, lower, orchestrator, histogram,
};
