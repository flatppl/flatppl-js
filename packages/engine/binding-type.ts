'use strict';
// binding-type — the `type` tag a binding carries, read from its RHS.
//
// The analyzer tags every user binding with it, and lift tags the synthetic
// bindings it creates and the bindings a rewrite changes the head of. Both
// read this one table: when lift kept its own copy, the copy lacked the
// likelihood tag, and a lifted `likelihoodof` was reported as a fixed binding
// that "produced no value".

// Heads that produce a first-class object or a variate, by the tag their
// consumers dispatch on. Every other call is 'call'.
const HEAD_BINDING_TYPE: ReadonlyMap<string, string> = new Map([
  ['draw', 'draw'],
  ['elementof', 'input'],
  ['external', 'input'],
  ['lawof', 'lawof'],
  ['functionof', 'functionof'],
  ['kernelof', 'kernelof'],
  ['fn', 'fn'],
  ['bijection', 'bijection'],
  // fchain produces a function-valued binding (spec §04 "Function
  // composition and annotation"). Its own tag keeps the IR op grep-able, but
  // consumers test the function-like predicate, not the tag.
  ['fchain', 'fchain'],
  ['likelihoodof', 'likelihood'],
  ['joint_likelihood', 'likelihood'],
  ['bayesupdate', 'bayesupdate'],
  ['load_module', 'module'],
  ['standard_module', 'module'],
  ['load_data', 'data'],
]);

const LITERAL_NODE_TYPES = new Set([
  'ArrayLiteral', 'NumberLiteral', 'StringLiteral', 'TupleLiteral', 'BoolLiteral',
]);

/** Classify the §04 input, reified-object and module producers and §07 loaded data. */
function bindingTypeOf(valueNode: any): string {
  if (!valueNode) return 'call';
  if (valueNode.type === 'CallExpr' && valueNode.callee?.type === 'Identifier') {
    return HEAD_BINDING_TYPE.get(valueNode.callee.name) ?? 'call';
  }
  return LITERAL_NODE_TYPES.has(valueNode.type) ? 'literal' : 'call';
}

module.exports = { bindingTypeOf };
