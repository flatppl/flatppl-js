'use strict';

// Standard-module registry skeleton (spec §09). The registry surface
// — lookup / register / list — is testable in isolation; downstream
// integration with the analyzer (resolving `standard_module(name,
// compat)` calls to a registry entry) is a follow-up.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const stdmod = require('../standard-modules.ts');

test('standard-modules: empty registry by default', () => {
  stdmod._clearStandardModules();
  assert.deepEqual(stdmod.listStandardModules(), []);
});

test('standard-modules: lookup returns null for unknown (name, compat)', () => {
  stdmod._clearStandardModules();
  assert.equal(stdmod.lookupStandardModule('particle-physics', '0.1'), null);
});

test('standard-modules: register + lookup round-trip', () => {
  stdmod._clearStandardModules();
  const bindings = new Map();
  bindings.set('square', { kind: 'function',
    impl: (x: any) => x * x });
  stdmod.registerStandardModule({
    name: 'test-mod',
    compat: '0.1',
    bindings,
  });
  const got = stdmod.lookupStandardModule('test-mod', '0.1');
  assert.ok(got, 'lookup returns the entry');
  assert.equal(got.name, 'test-mod');
  assert.equal(got.compat, '0.1');
  assert.ok(got.bindings.has('square'));
  // The impl is callable end-to-end.
  const desc = got.bindings.get('square');
  assert.equal(desc.kind, 'function');
  assert.equal(desc.impl(5), 25);
});

test('standard-modules: compat mismatch yields null', () => {
  stdmod._clearStandardModules();
  stdmod.registerStandardModule({
    name: 'test-mod', compat: '0.1', bindings: new Map(),
  });
  assert.equal(stdmod.lookupStandardModule('test-mod', '0.2'), null,
    'exact-match registry; semver matching is a follow-up');
});

test('standard-modules: re-registering same (name, compat) replaces', () => {
  stdmod._clearStandardModules();
  const b1 = new Map(); b1.set('x', { kind: 'value', value: 1 });
  const b2 = new Map(); b2.set('x', { kind: 'value', value: 2 });
  stdmod.registerStandardModule({ name: 'm', compat: '0.1', bindings: b1 });
  stdmod.registerStandardModule({ name: 'm', compat: '0.1', bindings: b2 });
  const got = stdmod.lookupStandardModule('m', '0.1');
  assert.equal(got.bindings.get('x').value, 2);
});

test('standard-modules: listStandardModules surfaces every registered entry', () => {
  stdmod._clearStandardModules();
  stdmod.registerStandardModule({ name: 'a', compat: '0.1', bindings: new Map() });
  stdmod.registerStandardModule({ name: 'b', compat: '0.1', bindings: new Map() });
  const list = stdmod.listStandardModules();
  assert.equal(list.length, 2);
  assert.ok(list.some((e: any) => e.name === 'a'));
  assert.ok(list.some((e: any) => e.name === 'b'));
});

test('standard-modules: binding descriptors support function / value kinds', () => {
  stdmod._clearStandardModules();
  const bindings = new Map();
  bindings.set('PI', { kind: 'value', value: 3.14159 });
  bindings.set('double', { kind: 'function', impl: (x: any) => x * 2 });
  stdmod.registerStandardModule({
    name: 'kinds-demo', compat: '0.1', bindings,
  });
  const got = stdmod.lookupStandardModule('kinds-demo', '0.1');
  assert.equal(got.bindings.get('PI').value, 3.14159);
  assert.equal(got.bindings.get('double').impl(7), 14);
});

test('standard-modules: exposed on engine index as standardModules', () => {
  const e = require('..');
  assert.ok(e.standardModules, 'engine exports standardModules');
  assert.equal(typeof e.standardModules.lookupStandardModule, 'function');
  assert.equal(typeof e.standardModules.registerStandardModule, 'function');
  assert.equal(typeof e.standardModules.listStandardModules, 'function');
});

test('particle-physics: kallen / breakup_momentum / blatt_weisskopf match spec §09 formulas', () => {
  stdmod._clearStandardModules();
  stdmod._registerBuiltinStandardModules();
  const pp = stdmod.lookupStandardModule('particle-physics', '0.1').bindings;
  // Independent spec closed-form oracles (NOT the engine impls):
  const kallen = (x: any, y: any, z: any) => x*x + y*y + z*z - 2*(x*y + y*z + z*x);
  const chi = (l: any, z: any) =>
    [1, 1+z, 9+3*z+z*z, 225+45*z+6*z*z+z**3, 11025+1575*z+135*z*z+10*z**3+z**4][l];
  const breakup = (m: any, a: any, b: any) => Math.sqrt(kallen(m*m, a*a, b*b)) / (2*m);
  const blatt = (l: any, p: any, d: any) => { const z = (d*p)**2; return Math.sqrt(z**l / chi(l, z)); };
  const close = (a: any, b: any) => Math.abs(a - b) < 1e-12;

  const K = pp.get('kallen').impl, BM = pp.get('breakup_momentum').impl, BW = pp.get('blatt_weisskopf').impl;
  assert.ok(close(K(2, 3, 4), -23), 'kallen(2,3,4) = -23');
  assert.ok(close(K(2, 3, 4), kallen(2, 3, 4)), 'kallen matches formula');
  assert.ok(close(BM(5, 1, 2), breakup(5, 1, 2)), 'breakup_momentum matches √λ(m²,ma²,mb²)/(2m)');
  for (const [l, p, d] of [[0,2,3],[1,2,3],[2,1,1],[3,1.5,2],[4,0.8,1.2]]) {
    assert.ok(close(BW(l, p, d), blatt(l, p, d)), `blatt_weisskopf l=${l} = √(zˡ/χ_ℓ(z))`);
  }
  assert.ok(close(BW(0, 2, 3), 1), 'F_0 = 1');
  // ℓ=5..7 are spec'd but not yet implemented (barrier polynomials 0..4 only).
  assert.throws(() => BW(5, 1, 1), 'blatt_weisskopf l=5 not implemented');
});
