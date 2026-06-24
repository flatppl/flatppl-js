'use strict';

// The host-agnostic transitive `load_module` bundle walker (spec §04). Both
// the web gallery and the VS Code extension drive THIS one function with
// their own async `readSource` primitive, so the walk can't drift between
// hosts. Sources are keyed by RESOLVED path (the engine's bundle key).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { resolveBundle } = require('..');

test('resolveBundle walks transitive deps, keyed by resolved path, deduped', async () => {
  const files: Record<string, string> = {
    'examples/a.flatppl': 'x = load_module("b.flatppl")\ny = load_module("sub/c.flatppl")',
    'examples/b.flatppl': 'z = load_module("sub/c.flatppl")', // shared dep
    'examples/sub/c.flatppl': 'w = 1',
  };
  const reads: string[] = [];
  const readSource = async (p: string) => { reads.push(p); return files[p] ?? null; };

  const r = await resolveBundle('examples/a.flatppl', files['examples/a.flatppl'], readSource);

  assert.equal(r.primaryPath, 'examples/a.flatppl');
  assert.deepEqual(Object.keys(r.sources).sort(),
    ['examples/b.flatppl', 'examples/sub/c.flatppl']);
  // c.flatppl is reached from BOTH a and b but read exactly once (dedupe by
  // resolved path) — the canonical key both importers resolve to.
  assert.equal(reads.filter((p) => p === 'examples/sub/c.flatppl').length, 1);
});

test('resolveBundle tolerates a missing dep (omitted, no throw)', async () => {
  const primary = 'x = load_module("missing.flatppl")';
  const r = await resolveBundle('a.flatppl', primary, async () => null);
  assert.deepEqual(Object.keys(r.sources), []);
});

test('resolveBundle is cycle-safe', async () => {
  const files: Record<string, string> = {
    'a.flatppl': 'x = load_module("b.flatppl")',
    'b.flatppl': 'y = load_module("a.flatppl")',
  };
  const r = await resolveBundle('a.flatppl', files['a.flatppl'], async (p: string) => files[p] ?? null);
  assert.deepEqual(Object.keys(r.sources), ['b.flatppl']); // a is the primary; no loop
});
