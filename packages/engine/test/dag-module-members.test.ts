'use strict';

// DAG-view representation of cross-module references (spec §04 Module
// composition). An INLINE member use (`a = common.f_a(theta2)`) should
// render as an explicit `common.f_a` member node — not collapse into a
// bare edge to the opaque `common` module node, which hides which member
// is used. A member ALIASED to a local binding (`theta1 = common.theta1_dist`)
// keeps its local name (the binding's own node already stands for it).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, computeSubDAG } = require('..');

const DEP = [
  'theta1_dist = Normal(0, 1)',
  'theta2_dist = Exponential(1)',
  'c = elementof(reals)',
  'f_a = par -> c * par',
].join('\n');

function build(model: string) {
  return processSource(model, { bundle: { sources: { 'dep.flatppl': DEP } } });
}

test('an inline cross-module member use emits a member node, not a bare module edge', () => {
  const r = build([
    'c_scaling = 5',
    'common = load_module("dep.flatppl", c = c_scaling)',
    'theta2 = common.theta2_dist',
    'a = common.f_a(theta2)',
  ].join('\n'));
  const { nodes, edges } = computeSubDAG(r.bindings, 'a', { linkedBindings: r.linkedBindings });
  const ids = nodes.map((n: any) => n.id);

  assert.ok(ids.includes('common.f_a'), 'a `common.f_a` member node should exist');
  assert.ok(edges.some((e: any) => e.source === 'common.f_a' && e.target === 'a'),
    'member node should feed its consumer (common.f_a → a)');
  assert.ok(!edges.some((e: any) => e.source === 'common' && e.target === 'a'),
    'the bare `common → a` collapse edge should be gone');
  assert.ok(edges.some((e: any) => e.source === 'common' && e.target === 'common.f_a'),
    'membership edge common → common.f_a so the substitution path stays visible');

  const mem = nodes.find((n: any) => n.id === 'common.f_a');
  assert.deepEqual(mem.moduleMember, { module: 'common', field: 'f_a' },
    'member node carries drill-in metadata');
});

test('a member aliased to a local name stays a local node (no member node)', () => {
  const r = build([
    'c_scaling = 5',
    'common = load_module("dep.flatppl", c = c_scaling)',
    'theta1 = common.theta1_dist',
  ].join('\n'));
  const { nodes, edges } = computeSubDAG(r.bindings, 'theta1', { linkedBindings: r.linkedBindings });
  const ids = nodes.map((n: any) => n.id);

  assert.ok(ids.includes('theta1'), 'theta1 local node present');
  assert.ok(!ids.includes('common.theta1_dist'),
    'no member node for a member that is aliased to a local name');
  assert.ok(edges.some((e: any) => e.source === 'common' && e.target === 'theta1'),
    'the alias keeps a direct common → theta1 edge');
});
