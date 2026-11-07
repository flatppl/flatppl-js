'use strict';

// The engine's passes recurse over expressions. A statement nested deeper
// than they can recurse is refused with a located engine-limitation error,
// instead of a RangeError from some later pass (found on 2000-term
// left-nested sums from an older determinizer).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, moduleDeps } = require('../index.ts');

const chain = (n: number) => Array(n).fill('t').join(' + ');

test('a sum nested past the limit is a located engine-limitation error', () => {
  const r = processSource(`t = 0.5\nscore = ${chain(2000)}\nother = t + 1.0\n`);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
  assert.match(errs[0].message, /^engine limitation: this expression nests 2001 levels deep; this engine handles up to 256 levels/);
  assert.equal(errs[0].loc.start.line, 1);
  assert.ok(r.bindings.has('other'), 'the other statements still analyse');
  assert.ok(!r.bindings.has('score'));
});

test('a parse that would overflow is one error, not a crash', () => {
  const r = processSource(`t = 0.5\nscore = ${chain(30000)}\n`);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /^engine limitation: this expression nests too deeply to parse; this engine handles up to 256 levels/);
  assert.deepEqual(moduleDeps(`t = 0.5\nscore = ${chain(30000)}\n`), []);
});

test('a sum within the limit still analyses and evaluates', () => {
  const { orchestrator } = require('../index.ts');
  const r = processSource(`t = 0.5\nscore = ${chain(200)}\n`);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(orchestrator.buildDerivations(r.bindings).fixedValues.get('score'), 100);
});
