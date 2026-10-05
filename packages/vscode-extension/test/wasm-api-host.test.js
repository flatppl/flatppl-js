'use strict';

// Unit tests for the extension host's wasm-API loader (src/wasmApiHost.ts).
// The glue load, the wasm byte read and the existence check are injected,
// so the real ~2 MB artifact is not needed — these tests pin the
// contract the math-export and convert commands depend on: one
// instantiation, bytes handed to init, a named missing export reported,
// and a failed init retried. Needs a prior build:vendor
// (src/wasmApiHost.js).

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {
  createWasmApiHost, wasmApiPaths, WASM_API_GLUE, WASM_API_BINARY,
} = require('../src/wasmApiHost.js');

function harness(overrides = {}) {
  const calls = { init: [], loads: [], reads: [] };
  const mod = {
    default: async (opts) => { calls.init.push(opts); },
    convert: (input, from, to) => from + '->' + to + ':' + input,
    export_math: () => '<!doctype html>',
  };
  const paths = wasmApiPaths('/ext');
  const deps = Object.assign({
    paths,
    exists: () => true,
    loadGlue: (p) => { calls.loads.push(p); return mod; },
    readBinary: (p) => { calls.reads.push(p); return Uint8Array.from([0, 97, 115, 109]); },
  }, overrides);
  return { host: createWasmApiHost(deps), calls, mod, paths };
}

test('wasmApiPaths points at the build artifacts in lib/', () => {
  const p = wasmApiPaths('/ext');
  assert.equal(p.glue, path.join('/ext', 'lib', WASM_API_GLUE));
  assert.equal(p.binary, path.join('/ext', 'lib', WASM_API_BINARY));
  assert.equal(WASM_API_GLUE, 'flatppl_wasm_api.cjs');
  assert.equal(WASM_API_BINARY, 'flatppl_wasm_api_bg.wasm');
});

test('available() needs both the glue and the wasm binary', () => {
  assert.equal(harness().host.available(), true);
  const { paths } = harness();
  assert.equal(harness({ exists: (p) => p !== paths.glue }).host.available(), false);
  assert.equal(harness({ exists: (p) => p !== paths.binary }).host.available(), false);
});

test('init gets the wasm bytes, not a path the glue would fetch', async () => {
  const { host, calls, paths } = harness();
  await host.api();
  assert.deepEqual(calls.loads, [paths.glue]);
  assert.deepEqual(calls.reads, [paths.binary]);
  assert.equal(calls.init.length, 1);
  assert.deepEqual(calls.init[0].module_or_path, Uint8Array.from([0, 97, 115, 109]));
});

test('overlapping first calls share one instantiation', async () => {
  const { host, calls } = harness();
  const [a, b] = await Promise.all([host.api(), host.api()]);
  assert.equal(a, b);
  await host.api();
  assert.equal(calls.init.length, 1);
  assert.equal(calls.loads.length, 1);
});

test('entry returns the named export and calls through to the wasm', async () => {
  const { host } = harness();
  const convert = await host.entry('convert');
  assert.equal(convert('{}', 'pyhf', 'flatppl'), 'pyhf->flatppl:{}');
});

test('a missing export names itself and the rebuild', async () => {
  const { host } = harness({
    loadGlue: () => ({ default: async () => {} }),
  });
  await assert.rejects(() => host.entry('convert'),
    /the bundled wasm API has no convert \(rebuild lib\/ from a current flatppl-rust\)/);
});

test('a glue with no init entry is reported, not called', async () => {
  const { host } = harness({ loadGlue: () => ({}) });
  await assert.rejects(() => host.api(), /no init entry/);
});

test('a failed init does not poison the cache', async () => {
  let attempt = 0;
  const { host } = harness({
    loadGlue: () => ({
      default: async () => { attempt += 1; if (attempt === 1) throw new Error('bad magic'); },
      convert: () => 'ok',
    }),
  });
  await assert.rejects(() => host.api(), /bad magic/);
  const convert = await host.entry('convert');
  assert.equal(convert(), 'ok');
  assert.equal(attempt, 2);
});
