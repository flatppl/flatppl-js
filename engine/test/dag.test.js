'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, computeSubDAG, findBindingAtLine } = require('../index');

function dagOf(src, target) {
  const { bindings } = processSource(src);
  return computeSubDAG(bindings, target);
}

test('dag: simple ancestor trace', () => {
  const dag = dagOf(`
a = elementof(reals)
b = elementof(reals)
c = a + b
d = c * 2
`, 'd');
  const ids = dag.nodes.map(n => n.id).sort();
  assert.deepEqual(ids, ['a', 'b', 'c', 'd']);
});

test('dag: target flag is set on the requested node', () => {
  const dag = dagOf(`
a = elementof(reals)
b = a * 2
`, 'b');
  const target = dag.nodes.find(n => n.isTarget);
  assert.equal(target.id, 'b');
  assert.ok(!dag.nodes.find(n => n.id === 'a').isTarget);
});

test('dag: unknown node returns empty', () => {
  const dag = dagOf('a = 1\n', 'no_such');
  assert.deepEqual(dag.nodes, []);
  assert.deepEqual(dag.edges, []);
});

test('dag: kernelof boundary inputs stop the trace', () => {
  // 'theta' is a boundary input — its ancestors should NOT appear in the sub-DAG.
  const dag = dagOf(`
src = elementof(reals)
theta = src * 2
x = draw(Normal(mu = theta, sigma = 1))
m = kernelof(x, theta = theta)
`, 'm');
  const ids = dag.nodes.map(n => n.id).sort();
  assert.deepEqual(ids, ['m', 'theta', 'x']);
  // theta should be marked as boundary
  const thetaNode = dag.nodes.find(n => n.id === 'theta');
  assert.equal(thetaNode.isBoundary, true);
});

test('dag: kernelof boundary uses argName as label', () => {
  const dag = dagOf(`
v = elementof(reals)
y = v * 2
m = kernelof(y, alpha = v)
`, 'm');
  const vNode = dag.nodes.find(n => n.id === 'v');
  assert.equal(vNode.isBoundary, true);
  assert.equal(vNode.label, 'alpha');
});

test('dag: edge edgeType distinguishes call vs data', () => {
  const dag = dagOf(`
f = fn(_ * 2)
y = elementof(reals)
z = f(y)
`, 'z');
  const fEdge = dag.edges.find(e => e.source === 'f' && e.target === 'z');
  const yEdge = dag.edges.find(e => e.source === 'y' && e.target === 'z');
  assert.equal(fEdge.edgeType, 'call');
  assert.equal(yEdge.edgeType, 'data');
});

test('dag: synthetic boundary nodes for placeholder boundaries', () => {
  // _par_ is a placeholder; argName 'par' becomes a synthetic boundary,
  // labeled with the placeholder syntax (not the kwarg name) so the
  // original identifier in the body remains visible. The body uses a
  // non-fixed-phase binding so the reification isn't fn-like.
  const dag = dagOf(`
x = elementof(reals)
f = functionof(x * _par_, par = _par_)
`, 'f');
  const synth = dag.nodes.find(n => n.id === 'f:par');
  assert.ok(synth);
  assert.equal(synth.label, '_par_');
  assert.equal(synth.isBoundary, true);
});

test('dag: fn renders as a bare hexagon — no synthetic hole nodes', () => {
  const dag = dagOf(`
g = fn(_ + _)
`, 'g');
  const holes = dag.nodes.filter(n => n.id.startsWith('g:_'));
  assert.equal(holes.length, 0);
});

test('dag: bayesian_inference_1 model node sub-DAG', () => {
  const src = `
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Exponential(rate = 1.0))
f_a = functionof(5.0 * _par_, par = _par_)
f_b = fn(abs(_) * _)
a = f_a(theta1)
b = f_b(theta1, theta2)
obs = draw(iid(Normal(mu = a, sigma = b), 10))
model = kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)
`;
  const dag = dagOf(src, 'model');
  const ids = new Set(dag.nodes.map(n => n.id));
  // Should include obs, a, b, f_a, f_b, theta1, theta2 (boundaries)
  for (const id of ['model', 'obs', 'a', 'b', 'f_a', 'f_b', 'theta1', 'theta2']) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
  // theta1, theta2 must be boundaries
  for (const name of ['theta1', 'theta2']) {
    assert.equal(dag.nodes.find(n => n.id === name).isBoundary, true);
  }
});

test('dag: findBindingAtLine returns binding for the right line', () => {
  const src = 'a = 1\nb = 2\nc = 3\n';
  const { bindings } = processSource(src);
  const b = findBindingAtLine(bindings, 1);
  assert.equal(b.name, 'b');
});

test('dag: findBindingAtLine returns null for non-binding lines', () => {
  const src = 'a = 1\n# comment\nc = 3\n';
  const { bindings } = processSource(src);
  assert.equal(findBindingAtLine(bindings, 1), null);
});

test('dag: findBindingAtLine returns specific name when col is given (decomposition)', () => {
  // forward_kernel, prior = ...
  // columns:        ^0           ^16  (forward_kernel ends at 14, prior at ~21)
  const src = 'forward_kernel, prior = (1, 2)\n';
  const { bindings } = processSource(src);
  // Cursor on 'forward_kernel' (col 5) -> forward_kernel
  assert.equal(findBindingAtLine(bindings, 0, 5).name, 'forward_kernel');
  // Cursor on 'prior' (col 18) -> prior
  assert.equal(findBindingAtLine(bindings, 0, 18).name, 'prior');
});

test('dag: findBindingAtLine without col returns first binding (legacy behaviour)', () => {
  const src = 'forward_kernel, prior = (1, 2)\n';
  const { bindings } = processSource(src);
  assert.equal(findBindingAtLine(bindings, 0).name, 'forward_kernel');
});

test('dag: findBindingAtLine col falls back to first binding when col is off-name', () => {
  // Cursor on the RHS — no name match, fall back to first binding on the line
  const src = 'a, b = (1, 2)\n';
  const { bindings } = processSource(src);
  assert.equal(findBindingAtLine(bindings, 0, 10).name, 'a');
});
