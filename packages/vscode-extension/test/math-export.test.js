'use strict';

// Unit tests for runMathExport — the host-agnostic core behind the
// "FlatPPL: Export math as HTML / LaTeX / Typst" commands. Lives in
// src/mathExport.ts (no vscode import; every host dependency is injected),
// so it is required directly with no VS Code host stubbing — same pattern
// as the dependencyPrefetch tests. Needs a prior build:vendor (the
// transpiled src/*.js and lib/math-view.cjs).

const test = require('node:test');
const assert = require('node:assert/strict');
const { runMathExport, mathExportTargetPath } = require('../src/mathExport.js');

// A fake engine walk that resolves the deps it is given through readSource.
function fakeWalk(deps) {
  return async (_primaryPath, _source, readSource) => {
    const sources = {};
    for (const d of deps) {
      const t = await readSource(d);
      if (t != null) sources[d] = t;
    }
    return { sources };
  };
}

function harness(overrides = {}) {
  const calls = { requests: [], writes: [], confirms: [] };
  const deps = Object.assign({
    document: 'html',
    source: 'mu ~ Normal(0, 1)',
    path: '/work/models/eight.flatppl',
    resolveBundle: fakeWalk(['/work/models/dep.flatppl']),
    readSource: async (p) => (p.endsWith('dep.flatppl') ? 'x = 1' : null),
    exportMath: (json) => { calls.requests.push(JSON.parse(json)); return '<!doctype html>'; },
    exists: async () => false,
    confirmOverwrite: async (t) => { calls.confirms.push(t); return true; },
    write: async (t, text) => { calls.writes.push([t, text]); },
  }, overrides);
  return { deps, calls };
}

test('mathExportTargetPath puts <stem>.<ext> beside the source', () => {
  assert.equal(mathExportTargetPath('/work/models/eight.flatppl', 'tex'), '/work/models/eight.tex');
  assert.equal(mathExportTargetPath('eight.flatppl', 'typ'), 'eight.typ');
  assert.equal(mathExportTargetPath('/w/a.b.flatppl', 'html'), '/w/a.b.html');
});

test('runMathExport sends the buffer, path and bundle per the contract and writes beside the source', async () => {
  const { deps, calls } = harness();
  const r = await runMathExport(deps);
  assert.deepEqual(r, { status: 'written', target: '/work/models/eight.html' });
  assert.deepEqual(calls.requests, [{
    source: 'mu ~ Normal(0, 1)',
    path: '/work/models/eight.flatppl',
    bundle: { '/work/models/dep.flatppl': 'x = 1' },
    document: 'html',
  }]);
  assert.deepEqual(calls.writes, [['/work/models/eight.html', '<!doctype html>']]);
  assert.deepEqual(calls.confirms, []);   // nothing to overwrite, no prompt
});

test('an existing target asks first; a refusal cancels without writing', async () => {
  const { deps, calls } = harness({ document: 'typ', exists: async () => true, confirmOverwrite: async () => false });
  const r = await runMathExport(deps);
  assert.deepEqual(r, { status: 'cancelled', target: '/work/models/eight.typ' });
  assert.deepEqual(calls.writes, []);
  assert.equal(calls.requests.length, 1);   // rendered before asking
});

test('an explicit target (chosen in a dialog) replaces the default and is not asked about again', async () => {
  const { deps, calls } = harness({ target: '/elsewhere/out.html', exists: async () => true });
  const r = await runMathExport(deps);
  assert.equal(r.target, '/elsewhere/out.html');
  assert.deepEqual(calls.confirms, []);   // the dialog already confirmed the replacement
  assert.equal(calls.writes[0][0], '/elsewhere/out.html');
});

test('a renderer error propagates before any prompt or write', async () => {
  const { deps, calls } = harness({
    exists: async () => true,
    exportMath: () => { throw new Error('model.flatppl: FlatPPL parse error (line 1)'); },
  });
  await assert.rejects(runMathExport(deps), /parse error/);
  assert.deepEqual(calls.confirms, []);
  assert.deepEqual(calls.writes, []);
});

test('a failed bundle walk still renders the primary with an empty bundle', async () => {
  const { deps, calls } = harness({ resolveBundle: async () => { throw new Error('fs down'); } });
  await runMathExport(deps);
  assert.deepEqual(calls.requests[0].bundle, {});
});
