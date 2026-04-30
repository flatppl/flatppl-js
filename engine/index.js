'use strict';

const { tokenize, T } = require('./tokenizer');
const { parse } = require('./parser');
const { analyze, collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges } = require('./analyzer');
const { computeSubDAG, findBindingAtLine } = require('./dag');
const AST = require('./ast');
const builtins = require('./builtins');

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
  findEnclosingRanges,
  computeSubDAG, findBindingAtLine,
  AST, builtins,
};
