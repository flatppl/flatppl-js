'use strict';

// =====================================================================
// keyword-distinguished-inputs.test.ts — spec §04 "Calling conventions"
// =====================================================================
//
// A special operation's *distinguished inputs* are "unnamed, ordered inputs of
// fixed arity". §04: "A distinguished input has no name and so cannot be passed
// by keyword. […] Where this specification refers to a distinguished input by a
// name, as in `aggregate(f_reduction, output_axes, expr)`, the name identifies
// the input in prose only. A call binds the input by position, never by keyword
// argument."
//
// So every head below refuses the keyword spelling of a distinguished input and
// keeps the positional one. What FOLLOWS the distinguished inputs is governed by
// each head's own rule and stays legal by keyword: the boundary inputs of
// `functionof` / `kernelof`, the collection arguments of `broadcast`, and the
// substitutions of `load_module`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

const RULE = /never by keyword argument/;

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

/** Assert the keyword spelling of a distinguished input is refused by name. */
function refuses(head: string, src: string) {
  const ds = errors(src);
  const hit = ds.filter((d: any) => RULE.test(d.message) && d.message.includes(head + '()'));
  assert.ok(hit.length > 0,
    `expected ${head} to refuse the keyword spelling of a distinguished input; `
    + `got: ${ds.map((d: any) => d.message).join('; ')}`);
  assert.ok(hit.every((d: any) => d.loc && d.loc.start),
    'the refusal carries a source location');
}

/** Assert the positional spelling raises no keyword-rule diagnostic. */
function accepts(src: string, mustBeClean = true) {
  const ds = errors(src);
  assert.equal(ds.filter((d: any) => RULE.test(d.message)).length, 0,
    `positional spelling must not trip the keyword rule; got: `
    + ds.map((d: any) => d.message).join('; '));
  if (mustBeClean) {
    assert.equal(ds.length, 0,
      `positional spelling must analyse cleanly; got: `
      + ds.map((d: any) => d.message).join('; '));
  }
}

// ---------------------------------------------------------------------
// One distinguished input, and nothing else
// ---------------------------------------------------------------------

test('elementof: the keyword spelling of its distinguished input is refused', () => {
  refuses('elementof', 'mu = elementof(set = reals)\n');
  accepts('mu = elementof(reals)\ny = exp(mu)\n');
});

test('external: the keyword spelling of its distinguished input is refused', () => {
  refuses('external', 'x = external(valueset = reals)\n');
  accepts('x = external(reals)\ny = exp(x)\n');
});

test('draw: the keyword spelling of its distinguished input is refused', () => {
  refuses('draw', 'x = draw(measure = Normal(0.0, 1.0))\n');
  accepts('x = draw(Normal(0.0, 1.0))\n');
});

test('lawof: the keyword spelling of its distinguished input is refused', () => {
  refuses('lawof', 'x = draw(Normal(0.0, 1.0))\nL = lawof(variate = x)\n');
  accepts('x = draw(Normal(0.0, 1.0))\nL = lawof(x)\n');
});

test('fixed: the keyword spelling of its distinguished input is refused', () => {
  // `fixed(...)` marks a held-constant preset entry (spec §03), so it appears
  // INSIDE a record rather than as a statement's outermost call — the walk has
  // to reach it there.
  refuses('fixed', 'p = record(a = fixed(value = 2.0), b = 1.0)\n');
  accepts('p = record(a = fixed(2.0), b = 1.0)\n');
});

test('broadcasted: the keyword spelling of its distinguished input is refused', () => {
  refuses('broadcasted', 'g = broadcasted(f = sin)\n');
  accepts('v = [1.0, 2.0]\ng = broadcasted(sin)\nr = g(v)\n');
});

// ---------------------------------------------------------------------
// One distinguished input, then further inputs of the head's own kind
// ---------------------------------------------------------------------

test('functionof: the first input is positional, the boundary inputs stay keyword', () => {
  refuses('functionof', 'mu = elementof(reals)\nf = functionof(expr = mu + 1.0, mu = mu)\n');
  accepts('mu = elementof(reals)\nf = functionof(mu + 1.0, mu = mu)\n');
});

test('kernelof: the first input is positional, the boundary inputs stay keyword', () => {
  refuses('kernelof', 'k = kernelof(expr = prev + 1.0, prev = prev)\n');
  accepts('k = kernelof(prev + 1.0, prev = prev)\n');
});

test('broadcast: the head is positional, the collection arguments stay keyword', () => {
  refuses('broadcast', 'v = [1.0, 2.0]\nr = broadcast(f = sin, x = v)\n');
  accepts('v = [1.0, 2.0]\nr = broadcast(sin, v)\n');
  // §04 gives `broadcast` "named or unnamed inputs that match the inputs of
  // that function" after the head, so the keyword form of those is legal —
  // including over a distribution head.
  accepts('v = [1.0, 2.0]\nr = broadcast(Poisson, rate = v)\n');
});

test('get: the collection is positional', () => {
  refuses('get', 'v = [1.0, 2.0]\nx = get(collection = v, index = 1)\n');
  accepts('v = [1.0, 2.0]\nx = get(v, 1)\n');
});

test('load_module: the path is positional, the substitutions stay keyword', () => {
  refuses('load_module', 'm = load_module(path = "mod.flatppl")\n');
  accepts('m = load_module("mod.flatppl")\n');
});

// ---------------------------------------------------------------------
// Two and three distinguished inputs
// ---------------------------------------------------------------------

test('standard_module: both distinguished inputs are positional', () => {
  refuses('standard_module',
    'pp = standard_module(name = "particle-physics", version = "0.1")\n');
  accepts('pp = standard_module("particle-physics", "0.1")\n'
    + 'y = pp.kallen(1.0, 2.0, 3.0)\n');
});

test('ksuperpose: both distinguished inputs are positional', () => {
  // The idiomatic form APPLIES the resulting kernel, so the `ksuperpose` call
  // is the callee of an enclosing call — another position the walk must reach.
  refuses('ksuperpose',
    'w = [0.5, 0.5]\n'
    + 'mix = normalize(ksuperpose(kernel = Normal, weights = w)(mu = [0.0, 1.0], sigma = 0.5))\n');
  accepts('w = [0.5, 0.5]\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = [0.0, 1.0], sigma = 0.5))\n');
});

test('markovchain: all three distinguished inputs are positional', () => {
  refuses('markovchain',
    'f = fn(Normal(mu = _, sigma = 0.5))\n'
    + 'traj = markovchain(kernel = f, init = 0.0, n = 4)\n');
  accepts('f = fn(Normal(mu = _, sigma = 0.5))\ntraj = markovchain(f, 0.0, 4)\n');
});

test('kscan: all three distinguished inputs are positional', () => {
  refuses('kscan',
    'f = fn(Normal(mu = _, sigma = 0.5))\n'
    + 'traj = kscan(kernel = f, init = 0.0, xs = [0.5, 1.0])\n');
  accepts('f = fn(Normal(mu = _, sigma = 0.5))\ntraj = kscan(f, 0.0, [0.5, 1.0])\n');
});

test('aggregate: all three distinguished inputs are positional', () => {
  refuses('aggregate',
    'A = [[1.0, 2.0]]\nR = aggregate(f = sum, axes = [.i], e = A[.i, .j])\n');
  accepts('A = [[1.0, 2.0]]\nR = aggregate(sum, [.i], A[.i, .j])\n');
});

test('metricsum: all three distinguished inputs are positional', () => {
  refuses('metricsum',
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\np = [3.0, 2.0]\n'
    + 'r = metricsum(metric = g, output_axes = [.mu^], expr = p[.mu^])\n');
  accepts('g = rowstack([[1.0, 0.0], [0.0, -1.0]])\np = [3.0, 2.0]\n'
    + 'r = metricsum(g, [.mu^], p[.mu^])\n');
});

// ---------------------------------------------------------------------
// A keyword at a trailing distinguished input, and the mixed spelling
// ---------------------------------------------------------------------

test('a keyword at any distinguished position is refused, not just the first', () => {
  // §05 forbids a positional argument after a keyword one, so a keyword at
  // position i means positions 1..i-1 were spelled positionally.
  refuses('markovchain',
    'f = fn(Normal(mu = _, sigma = 0.5))\n'
    + 'traj = markovchain(f, 0.0, n = 4)\n');
  refuses('standard_module', 'pp = standard_module("particle-physics", version = "0.1")\n');
});

test('a head with no distinguished input keeps its keyword form', () => {
  // §04 gives `joint` "variadic unnamed or named inputs" and `cartprod` the
  // same, so neither has a distinguished input and the keyword spelling is the
  // canonical relabel-as form. The refusal must not reach them.
  accepts('M1 = Normal(0.0, 1.0)\nM2 = Normal(1.0, 1.0)\nJ = joint(a = M1, b = M2)\n');
  accepts('S = cartprod(a = reals, b = reals)\n');
});

test('one refusal per offending distinguished input', () => {
  const ds = errors(
    'f = fn(Normal(mu = _, sigma = 0.5))\n'
    + 'traj = markovchain(kernel = f, init = 0.0, n = 4)\n');
  assert.equal(ds.filter((d: any) => RULE.test(d.message)).length, 3,
    'three keyword arguments at three distinguished inputs give three refusals');
});
