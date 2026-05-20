'use strict';

// Tests for engine/pir.js — the LoweredModule structure and the
// AST → FlatPIR-JSON-aligned module-level lowering.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const pir = require('../pir.ts');

function lowered(src) {
  const { bindings } = processSource(src);
  return pir.lowerToModule(bindings);
}

// =====================================================================
// Module structure
// =====================================================================

test('lowerToModule: produces one binding per analyzer binding, in source order', () => {
  const m = lowered(`
    a = 1.0
    b = 2.0
    c = a + b
  `);
  assert.deepEqual(Array.from(m.bindings.keys()), ['a', 'b', 'c']);
});

test('lowerToModule: source backref is the analyzer bindings map', () => {
  const { bindings } = processSource(`x = 42`);
  const m = pir.lowerToModule(bindings);
  assert.equal(m.source, bindings);
});

test('lowerToModule: public set excludes underscore-prefixed names', () => {
  const m = lowered(`
    public_a = 1
    _private_b = 2
    __anon = 3
  `);
  assert.ok(m.publicSet.has('public_a'));
  assert.ok(!m.publicSet.has('_private_b'));
  assert.ok(!m.publicSet.has('__anon'));
});

// =====================================================================
// Per-binding lowering — sanity checks against lower.js shapes
// =====================================================================

test('binding RHS: literal becomes a lit expression', () => {
  const m = lowered(`x = 42`);
  const b = m.bindings.get('x');
  assert.equal(b.rhs.kind, 'lit');
  assert.equal(b.rhs.value, 42);
});

test('binding RHS: identifier becomes a self ref', () => {
  const m = lowered(`
    a = 1
    b = a
  `);
  const b = m.bindings.get('b');
  assert.equal(b.rhs.kind, 'ref');
  assert.equal(b.rhs.ns, 'self');
  assert.equal(b.rhs.name, 'a');
});

test('binding RHS: distribution call has op + kwargs', () => {
  const m = lowered(`d = Normal(mu = 0, sigma = 1)`);
  const r = m.bindings.get('d').rhs;
  assert.equal(r.kind, 'call');
  assert.equal(r.op, 'Normal');
  assert.equal(r.kwargs.mu.kind, 'lit');
  assert.equal(r.kwargs.sigma.kind, 'lit');
});

test('binding RHS: arithmetic operator desugars to add/mul/etc.', () => {
  const m = lowered(`
    a = 1
    b = 2
    c = a + b * 3
  `);
  const r = m.bindings.get('c').rhs;
  assert.equal(r.kind, 'call');
  assert.equal(r.op, 'add');
  assert.equal(r.args[1].op, 'mul');
});

test('binding RHS: functionof body lives in %local scope', () => {
  // Per FlatPIR: parameter refs inside a functionof body use
  // (%ref %local <name>). lower.js already implements this scope
  // tracking; this test pins the contract at the module level.
  const m = lowered(`
    f = functionof(_par_ + 1, par = _par_)
  `);
  const r = m.bindings.get('f').rhs;
  assert.equal(r.kind, 'call');
  assert.equal(r.op, 'functionof');
  // Body's left-hand side is a ref into %local for the parameter,
  // not into self.
  const localRef = r.body.args[0];
  assert.equal(localRef.ns, '%local');
  assert.equal(localRef.name, '_par_');
});

test('binding RHS: closed-over fixed ancestors stay as self refs', () => {
  // Inside a functionof body, refs to outer fixed names (like c here)
  // are NOT in %local — they're closures into self. Boundary
  // substitution doesn't touch them (they're not declared as kwargs).
  const m = lowered(`
    c = 2.5
    f = functionof(c * _par_, par = _par_)
  `);
  const r = m.bindings.get('f').rhs;
  // Body is `c * _par_` → (mul (ref self c) (ref %local _par_))
  const mulCall = r.body;
  assert.equal(mulCall.args[0].ns, 'self');
  assert.equal(mulCall.args[0].name, 'c');
  assert.equal(mulCall.args[1].ns, '%local');
});

// =====================================================================
// Synthetic / origin tracking
// =====================================================================

test('originLoc: every binding carries the source AST loc for diagnostics', () => {
  const m = lowered(`x = 42\ny = 3.14`);
  const x = m.bindings.get('x'), y = m.bindings.get('y');
  assert.equal(x.originLoc.start.line, 0);
  assert.equal(y.originLoc.start.line, 1);
});

test('synthetic flag: regular user bindings are not synthetic', () => {
  const m = lowered(`x = 1`);
  assert.equal(m.bindings.get('x').synthetic, false);
});

// =====================================================================
// Helpers
// =====================================================================

test('walkCalls: visits every call depth-first, post-order', () => {
  const m = lowered(`
    x = 1
    y = Normal(mu = x + 2, sigma = 1)
  `);
  const r = m.bindings.get('y').rhs;
  const visited = [];
  pir.walkCalls(r, c => visited.push(c.op));
  // add (inner) before Normal (outer)
  assert.deepEqual(visited, ['add', 'Normal']);
});

test('setMeta: writes type/phase into the call meta slot', () => {
  const m = lowered(`x = Normal(mu = 0, sigma = 1)`);
  const r = m.bindings.get('x').rhs;
  pir.setMeta(r, 'measure<real>', 'fixed');
  assert.equal(r.meta.type, 'measure<real>');
  assert.equal(r.meta.phase, 'fixed');
});
