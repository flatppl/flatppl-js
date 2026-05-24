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
