'use strict';

// Unit tests for the "FlatPPL: Convert HS3 / pyhf to FlatPPL" core.
// Lives in src/convert.ts (no vscode import; the wasm entry is injected),
// so it is required directly with no VS Code host stubbing — same pattern
// as the math-export tests. Needs a prior build:vendor (src/convert.js).

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  runConvert, sourceFormatForPath, convertedFlatpplName, SOURCE_FORMATS,
} = require('../src/convert.js');

test('sourceFormatForPath reads the gallery extensions and nothing else', () => {
  assert.equal(sourceFormatForPath('/w/model.hs3.json'), 'hs3');
  assert.equal(sourceFormatForPath('/w/model.hs3'), 'hs3');
  assert.equal(sourceFormatForPath('/w/model.pyhf.json'), 'pyhf');
  assert.equal(sourceFormatForPath('/w/model.pyhf'), 'pyhf');
  assert.equal(sourceFormatForPath('/w/MODEL.PyHF.JSON'), 'pyhf');
  // A bare .json says nothing — guessing the schema from the name would
  // misread half the HS3TestSuite corpus.
  assert.equal(sourceFormatForPath('/w/2bin_1channel.json'), null);
  assert.equal(sourceFormatForPath('/w/model.flatppl'), null);
  assert.equal(sourceFormatForPath(''), null);
});

test('convertedFlatpplName drops the import extension', () => {
  assert.equal(convertedFlatpplName('/w/model.pyhf.json'), 'model.flatppl');
  assert.equal(convertedFlatpplName('/w/model.hs3.json'), 'model.flatppl');
  assert.equal(convertedFlatpplName('/w/model.hs3'), 'model.flatppl');
  assert.equal(convertedFlatpplName('/w/2bin_1channel.json'), '2bin_1channel.flatppl');
  assert.equal(convertedFlatpplName('a.b.json'), 'a.b.flatppl');
  assert.equal(convertedFlatpplName(''), 'model.flatppl');
});

test('runConvert calls the wasm entry with the inferred importer and flatppl target', () => {
  const calls = [];
  const r = runConvert({
    path: '/w/ws.pyhf.json',
    source: '{"channels": []}',
    convert: (input, from, to) => { calls.push([input, from, to]); return 'x = 1\n'; },
  });
  assert.deepEqual(calls, [['{"channels": []}', 'pyhf', 'flatppl']]);
  assert.deepEqual(r, { format: 'pyhf', name: 'ws.flatppl', source: 'x = 1\n' });
});

test('an explicit format overrides the name', () => {
  const calls = [];
  const r = runConvert({
    path: '/w/ws.pyhf.json',
    source: '{}',
    from: 'hs3',
    convert: (input, from, to) => { calls.push(from); return ''; },
  });
  assert.deepEqual(calls, ['hs3']);
  assert.equal(r.format, 'hs3');
});

test('an unnameable source is refused before the importer runs', () => {
  let ran = false;
  assert.throws(() => runConvert({
    path: '/w/workspace.json',
    source: '{}',
    convert: () => { ran = true; return ''; },
  }), /cannot tell whether .* is HS3 or pyhf/);
  assert.equal(ran, false);
});

test('the importer diagnostic propagates verbatim', () => {
  assert.throws(() => runConvert({
    path: '/w/ws.hs3.json',
    source: '{}',
    convert: () => { throw new Error('hs3 import: unknown modifier "shapesys2"'); },
  }), /hs3 import: unknown modifier "shapesys2"/);
});

test('SOURCE_FORMATS is exactly what the wasm convert accepts', () => {
  assert.deepEqual(SOURCE_FORMATS, ['hs3', 'pyhf']);
});
