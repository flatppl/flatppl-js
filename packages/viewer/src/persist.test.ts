// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const require_ = createRequire(import.meta.url);
const { processSource } = require_('@flatppl/engine');
const { persistNamedDomain, persistNamedPreset } = await import('./persist.ts');

// The web host's editSource (packages/web/src/app.ts): 0-based lines, the
// column added to the line start without clamping.
function applyEdit(src, args) {
  const lineStarts = [0];
  for (let i = 0; i < src.length; i++) if (src.charCodeAt(i) === 10) lineStarts.push(i + 1);
  const off = (loc) => (lineStarts[loc.line] ?? 0) + (loc.col || 0);
  return src.slice(0, off(args.range.start)) + args.newText + src.slice(off(args.range.end));
}

function ctxFor(src) {
  const edits = [];
  const ctx = {
    currentBindings: processSource(src).bindings,
    domainOverrides: new Map(),
    presets: new Map(),
    host: { editSource: (args) => { edits.push(args); } },
  };
  return { ctx, edits };
}

test('persistNamedDomain replaces exactly the cartprod binding', () => {
  const src = [
    'mu ~ Normal(0.0, 1.0)',
    'sigma ~ Exponential(1.0)',
    'box = cartprod(mu = interval(-1.5, 1.5), sigma = interval(0.25, 2.5))',
    'z = 1.0',
    '',
  ].join('\n');
  const { ctx, edits } = ctxFor(src);
  ctx.domainOverrides.set('box', { ranges: { mu: { lo: -2.5, hi: 2.5 } } });
  persistNamedDomain(ctx, { domainName: 'box' });
  assert.equal(edits.length, 1);
  const out = applyEdit(src, edits[0]);
  assert.equal(out, [
    'mu ~ Normal(0.0, 1.0)',
    'sigma ~ Exponential(1.0)',
    'box = cartprod(mu = interval(-2.5, 2.5), sigma = interval(0.25, 2.5))',
    'z = 1.0',
    '',
  ].join('\n'));
  // Round trip: the edited source analyses cleanly and keeps every binding.
  const after = processSource(out);
  assert.deepEqual(after.diagnostics.filter((d) => d.severity === 'error'), []);
  assert.deepEqual([...after.bindings.keys()].sort(), ['box', 'mu', 'sigma', 'z']);
});

test('persistNamedDomain on the first line leaves the rest intact', () => {
  const src = 'box = cartprod(a = interval(0.0, 1.0))\nx = 2.0\n';
  const { ctx, edits } = ctxFor(src);
  ctx.domainOverrides.set('box', { ranges: { a: { lo: 0.5, hi: 0.75 } } });
  persistNamedDomain(ctx, { domainName: 'box' });
  assert.equal(applyEdit(src, edits[0]), 'box = cartprod(a = interval(0.5, 0.75))\nx = 2.0\n');
});
