'use strict';

// Tests for engine/ir-walk.ts — the centralised IR-traversal
// primitives. Pins forEachIRChild's coverage of every recursive
// FlatPIR sub-position (args / kwargs / fields / body / branches /
// selector / logweights / assigns) and mapIR's identity-preserving
// rebuild contract.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { forEachIRChild, walkIR, mapIR } = require('../ir-walk.ts');

// =====================================================================
// forEachIRChild — every sub-position must be visited
// =====================================================================

test('forEachIRChild: args', () => {
  const a = { kind: 'ref', ns: 'self', name: 'a' };
  const b = { kind: 'ref', ns: 'self', name: 'b' };
  const node = { kind: 'call', op: 'add', args: [a, b] };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, [a, b]);
});

test('forEachIRChild: kwargs', () => {
  const mu = { kind: 'lit', value: 0, numType: 'real' };
  const sigma = { kind: 'lit', value: 1, numType: 'real' };
  const node = { kind: 'call', op: 'Normal', kwargs: { mu, sigma } };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.equal(seen.length, 2);
  assert.ok(seen.includes(mu));
  assert.ok(seen.includes(sigma));
});

test('forEachIRChild: fields (record/joint shape)', () => {
  const v1 = { kind: 'ref', ns: 'self', name: 'a' };
  const v2 = { kind: 'ref', ns: 'self', name: 'b' };
  const node = {
    kind: 'call', op: 'record',
    fields: [{ name: 'a', value: v1 }, { name: 'b', value: v2 }],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, [v1, v2]);
});

test('forEachIRChild: body (functionof)', () => {
  const body = { kind: 'call', op: 'mul', args: [] };
  const node = {
    kind: 'call', op: 'functionof',
    params: ['x'], paramKwargs: ['x'], paramSources: [],
    body,
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, [body]);
});

test('forEachIRChild: branches (select with inline IR)', () => {
  const ir1 = { kind: 'call', op: 'Normal', kwargs: {} };
  const ir2 = { kind: 'call', op: 'Dirac', kwargs: {} };
  const node = {
    kind: 'call', op: 'select',
    branches: [{ ir: ir1 }, { ir: ir2 }],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, [ir1, ir2]);
});

test('forEachIRChild: branches by-name skip recursion (no inline IR)', () => {
  const node = {
    kind: 'call', op: 'select',
    branches: [{ ref: 'M1' }, { ref: 'M2' }],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, []);
});

test('forEachIRChild: selector and logweights', () => {
  const selector = { kind: 'call', op: 'Bernoulli', kwargs: {} };
  const lw0 = { kind: 'lit', value: -0.6931, numType: 'real' };
  const lw1 = { kind: 'lit', value: -0.6931, numType: 'real' };
  const node = {
    kind: 'call', op: 'select',
    branches: [], selector, logweights: [lw0, lw1],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.equal(seen.length, 3);
  assert.ok(seen.includes(selector));
  assert.ok(seen.includes(lw0));
  assert.ok(seen.includes(lw1));
});

test('forEachIRChild: assigns (load_module)', () => {
  const r = { kind: 'ref', ns: 'self', name: 'sig_strength' };
  const node = {
    kind: 'call', op: 'load_module',
    args: [{ kind: 'lit', value: 'signal.flatppl' }],
    assigns: [{ name: 'mu', value: r }],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  // Both the path literal (positional arg) AND the assign value.
  assert.equal(seen.length, 2);
  assert.ok(seen.includes(r));
});

test('forEachIRChild: ignores non-IR fields (loc, meta, params)', () => {
  const node = {
    kind: 'call', op: 'foo',
    params: ['a', 'b'],
    paramKwargs: ['x', 'y'],
    paramSources: [{ kind: 'binding', name: 'a' }],
    loc: { start: { line: 1, col: 0 }, end: { line: 1, col: 5 } },
    meta: { type: { kind: 'scalar', prim: 'real' } },
    axisStack: [{ source: 'iid', size: 10 }],
  };
  const seen: any[] = [];
  forEachIRChild(node, (c: any) => seen.push(c));
  assert.deepEqual(seen, []);
});

test('forEachIRChild: non-objects are no-ops', () => {
  const seen: any[] = [];
  forEachIRChild(null, (c: any) => seen.push(c));
  forEachIRChild(undefined, (c: any) => seen.push(c));
  forEachIRChild('string', (c: any) => seen.push(c));
  forEachIRChild(42, (c: any) => seen.push(c));
  assert.deepEqual(seen, []);
});

// =====================================================================
// walkIR — post-order
// =====================================================================

test('walkIR: post-order across nested call', () => {
  const a = { kind: 'ref', ns: 'self', name: 'a' };
  const b = { kind: 'ref', ns: 'self', name: 'b' };
  const inner = { kind: 'call', op: 'mul', args: [a, b] };
  const c = { kind: 'lit', value: 1 };
  const outer = { kind: 'call', op: 'add', args: [inner, c] };
  const order: any[] = [];
  walkIR(outer, (n: any) => order.push(n));
  // post-order: a, b, inner, c, outer.
  assert.deepEqual(order, [a, b, inner, c, outer]);
});

test('walkIR: walks branches + selector + logweights', () => {
  const Normal_kw = { kind: 'lit', value: 0 };
  const Normal_call = { kind: 'call', op: 'Normal', kwargs: { mu: Normal_kw } };
  const Dirac_kw = { kind: 'lit', value: 1 };
  const Dirac_call = { kind: 'call', op: 'Dirac', kwargs: { value: Dirac_kw } };
  const selector = { kind: 'call', op: 'Bernoulli', kwargs: {} };
  const lw0 = { kind: 'lit', value: -0.6931 };
  const lw1 = { kind: 'lit', value: -0.6931 };
  const sel = {
    kind: 'call', op: 'select',
    branches: [{ ir: Normal_call }, { ir: Dirac_call }],
    selector, logweights: [lw0, lw1],
  };
  const seen = new Set();
  walkIR(sel, (n: any) => seen.add(n));
  // All inline nodes must be visited.
  assert.ok(seen.has(Normal_kw));
  assert.ok(seen.has(Normal_call));
  assert.ok(seen.has(Dirac_kw));
  assert.ok(seen.has(Dirac_call));
  assert.ok(seen.has(selector));
  assert.ok(seen.has(lw0));
  assert.ok(seen.has(lw1));
  assert.ok(seen.has(sel));
});

// =====================================================================
// mapIR — identity preservation + rebuild
// =====================================================================

test('mapIR: identity-preserving when transform is identity', () => {
  const a = { kind: 'ref', ns: 'self', name: 'a' };
  const b = { kind: 'ref', ns: 'self', name: 'b' };
  const inner = { kind: 'call', op: 'mul', args: [a, b] };
  const outer = { kind: 'call', op: 'add', args: [inner], kwargs: { extra: a } };
  const out = mapIR(outer, (n: any) => n);
  // No descendant changed, so the root identity must be preserved.
  assert.equal(out, outer);
});

test('mapIR: rewrites a ref inside args', () => {
  const a = { kind: 'ref', ns: 'self', name: 'a' };
  const b = { kind: 'ref', ns: 'self', name: 'b' };
  const node = { kind: 'call', op: 'add', args: [a, b] };
  const renamed = mapIR(node, (n: any) =>
    n && n.kind === 'ref' && n.name === 'a'
      ? { kind: 'ref', ns: '%local', name: 'a' }
      : n);
  assert.notEqual(renamed, node);          // root rebuilt
  assert.equal(renamed.args[1], b);        // untouched leaf preserved
  assert.equal(renamed.args[0].ns, '%local');
});

test('mapIR: rewrites a ref inside branches (regression for select-walker drift)', () => {
  const psi_ref = { kind: 'ref', ns: 'self', name: 'psi' };
  const weighted = { kind: 'call', op: 'weighted', args: [psi_ref, {}] };
  const dirac = { kind: 'call', op: 'Dirac' };
  const sel = {
    kind: 'call', op: 'select',
    branches: [{ ir: weighted }, { ir: dirac }],
  };
  const out = mapIR(sel, (n: any) =>
    n && n.kind === 'ref' && n.ns === 'self' && n.name === 'psi'
      ? { kind: 'ref', ns: '%local', name: 'psi' }
      : n);
  assert.notEqual(out, sel);
  assert.equal(out.branches[0].ir.args[0].ns, '%local');
  assert.equal(out.branches[1].ir, dirac); // untouched branch preserved
});

test('mapIR: rewrites a ref inside logweights array', () => {
  const r = { kind: 'ref', ns: 'self', name: 'logw' };
  const node = {
    kind: 'call', op: 'select',
    branches: [], logweights: [r, { kind: 'lit', value: 0 }],
  };
  let touched = 0;
  const out = mapIR(node, (n: any) => {
    if (n && n.kind === 'ref' && n.name === 'logw') {
      touched++;
      return { kind: 'lit', value: -42 };
    }
    return n;
  });
  assert.equal(touched, 1);
  assert.equal(out.logweights[0].value, -42);
  assert.equal(out.logweights[1].value, 0);
});
