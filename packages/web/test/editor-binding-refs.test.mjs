import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');

function loadEditor() {
  const file = new URL('../src/editor.ts', import.meta.url);
  const scope = { FlatPPLEngine: engine };
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), { globalThis: scope, window: scope },
    { filename: file.pathname });
  return scope.FlatPPLWebEditor;
}

/** Each binding-link range as `line:col name`, 1-based, for readable diffs. */
function refsOf(src) {
  const lines = src.split('\n');
  return Array.from(loadEditor().bindingRefs(src), ({ from, to, name }) => {
    let line = 0, start = 0;
    while (start + lines[line].length < from) { start += lines[line].length + 1; line++; }
    assert.equal(src.slice(from, to), name);
    return `${line + 1}:${from - start + 1} ${name}`;
  });
}

test('binding links cover binding targets and references only', () => {
  const src = [
    'x = 1',
    'a = 2',
    'f(x) = x^2 + a',
    'g = x -> x + a',
    'r = record(a = a, x = x)',
    'y = r.a',
    'k = f.(x)',
    'm ~ Normal(mu = a, sigma = 1)',
  ].join('\n');
  assert.deepEqual(refsOf(src), [
    '1:1 x', '2:1 a',
    // `x` is the parameter of f and of the lambda, not the binding x.
    '3:1 f', '3:14 a',
    '4:1 g', '4:14 a',
    // Keyword-argument names are not references.
    '5:1 r', '5:16 a', '5:23 x',
    // The member name `a` is local to r.
    '6:1 y', '6:5 r',
    '7:1 k', '7:5 f', '7:8 x',
    '8:1 m', '8:17 a',
  ]);
});

test('a binding name in a comment or a string gets no link', () => {
  assert.deepEqual(refsOf('x = 1\n# x here\ns = "x"'), ['1:1 x', '3:1 s']);
});
