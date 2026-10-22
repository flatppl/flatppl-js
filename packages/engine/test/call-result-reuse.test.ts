'use strict';

// checkUserCall reuses a callee's inferred result when every argument has
// exactly the type its module-binding boundary had at the definition. These
// tests run inference with the reuse on and off and require identical binding
// types, per-call annotations and diagnostics, over every model in this
// repository and in the sibling examples and testsuite checkouts.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const { _callResultReuse: reuse } = require('../typeinfer.ts');

function findSibling(name: string, env: string): string | null {
  if (process.env[env]) return process.env[env]!;
  let dir = __dirname;
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, name);
    if (fs.existsSync(cand)) return cand;
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}

function models(root: string | null): string[] {
  if (!root) return [];
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.flatppl')) out.push(full);
    }
  };
  walk(root);
  return out.sort();
}

// Load data and module dependencies the way a host bundle supplies them.
function inputFor(file: string) {
  const source = fs.readFileSync(file, 'utf8');
  const sources: Record<string, string> = {};
  const files: Record<string, any> = {};
  for (const m of source.matchAll(/load_(data|module)\("([^"]+)"/g)) {
    const dep = path.resolve(path.dirname(file), m[2]);
    if (!fs.existsSync(dep)) continue;
    if (m[1] === 'module') sources[dep] = fs.readFileSync(dep, 'utf8');
    else files[dep] = fs.readFileSync(dep);
  }
  return { source, opts: { path: file, bundle: { sources, files } } };
}

function snapshot(source: string, opts: any): string {
  let proc: any;
  try {
    proc = processSource(source, opts);
  } catch (e: any) {
    return 'threw: ' + e.message;
  }
  const calls: any[] = [];
  const walk = (n: any) => {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n.meta && n.meta.type) calls.push(n.meta.type);
    for (const k in n) if (k !== 'meta' && k !== 'loc') walk(n[k]);
  };
  const bindings: any = {};
  for (const [name, b] of proc.loweredModule?.bindings || []) {
    calls.length = 0;
    walk(b.rhs);
    bindings[name] = { type: b.inferredType, calls: calls.slice() };
  }
  // A skipped re-inference draws fewer fresh type variables, and synthetic
  // binding names count up across processSource calls. Compare both up to a
  // consistent renaming, numbered by first appearance.
  const ids = new Map<string, string>();
  return JSON.stringify({
    diagnostics: proc.diagnostics.map((d: any) => [d.severity, d.message, d.loc]),
    bindings,
  }).replace(/\bT_\d+|__[A-Za-z_]+?\d+/g, (id: string) => {
    if (!ids.has(id)) ids.set(id, '#' + ids.size);
    return ids.get(id)!;
  });
}

function compare(source: string, opts: any): string | null {
  reuse.enabled = true;
  const on = snapshot(source, opts);
  reuse.enabled = false;
  try {
    const off = snapshot(source, opts);
    return on === off ? null : 'types or diagnostics differ';
  } finally {
    reuse.enabled = true;
  }
}

const CORPORA: Array<[string, string[]]> = [
  ['engine fixtures', models(path.join(__dirname, 'fixtures'))],
  ['web demos', models(path.join(__dirname, '..', '..', 'web', 'demo'))],
  ['flatppl-examples', models(findSibling('flatppl-examples', 'FLATPPL_EXAMPLES_DIR'))],
  ['flatppl-testsuite corpora', models(
    (() => { const d = findSibling('flatppl-testsuite', 'FLATPPL_TESTSUITE_DIR'); return d && path.join(d, 'corpora'); })())],
];

for (const [label, files] of CORPORA) {
  test(`call-result reuse leaves types and diagnostics unchanged: ${label}`,
    { skip: files.length ? false : 'checkout absent' }, () => {
      const before = reuse.hits;
      const differ = files.filter(file => {
        const { source, opts } = inputFor(file);
        return compare(source, opts) != null;
      });
      assert.deepEqual(differ, []);
      // The comparison is vacuous unless the reuse applied somewhere.
      if (label !== 'web demos') assert.ok(reuse.hits > before, `${label}: reuse never applied`);
    });
}

test('an ill-typed call re-infers, so its diagnostics and locations stay the same', () => {
  // `a` is well typed and reuses K's result. `b` reports an unknown argument
  // first, so it and every later call re-infer as before.
  const source = `
theta ~ Normal(0, 1)
y ~ Normal(theta, 1)
K = kernelof(record(y = y), theta = theta)
a = K(theta = theta)
b = K(theta = theta, scale = 2)
c = K(theta)
`;
  const before = reuse.hits;
  assert.equal(compare(source, {}), null);
  assert.equal(reuse.hits - before, 1);
  const errors = processSource(source).diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors.map((d: any) => [d.message, d.loc.start.line]),
    [['call to "K" has unknown argument "scale"', 5]]);
});
