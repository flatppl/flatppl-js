// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { resolveDroppedChain, rewriteEdgesForCollapse } = await import('./dag.ts');

// Regression for #176: the graph-view compactor collapsed a reification
// bubble whose anchor was ITSELF a member of an outer collapsed bubble.
// dminus-to-3pi-amplitude.flatppl nests three deep —
// phase_weight ⊃ intensity_fn ⊃ angular_tensors, each anchor directly
// listed as a member of the one enclosing it. A single-hop lookup left an
// edge pointing at `intensity_fn` even though intensity_fn itself was
// dropped under `phase_weight`, and cytoscape threw on the dangling
// endpoint, killing the whole DAG render.

test('resolveDroppedChain follows a multi-level nested-anchor chain to the surviving anchor', () => {
  // angular_tensors dropped under intensity_fn, itself dropped under
  // phase_weight — phase_weight is never a value, so it's the chain's end.
  const dropped = new Map([
    ['angular_tensors', 'intensity_fn'],
    ['intensity_fn', 'phase_weight'],
    ['p1', 'angular_tensors'],
  ]);
  assert.equal(resolveDroppedChain(dropped, 'angular_tensors'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'p1'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'phase_weight'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'unrelated'), 'unrelated');
});

test('rewriteEdgesForCollapse re-routes every endpoint to a surviving anchor, never leaving a dangling edge', () => {
  const dropped = new Map([
    ['angular_tensors', 'intensity_fn'],
    ['intensity_fn', 'phase_weight'],
    ['p1', 'angular_tensors'],
  ]);
  const survivingNodeIds = new Set(['phase_weight', 'p2', 'q_of']);
  const edges = [
    // p1 -> angular_tensors: both endpoints nested under phase_weight,
    // collapses to a self-edge and is dropped.
    { source: 'p1', target: 'angular_tensors', edgeType: 'data' },
    // Straddles the collapsed bubble: p2 (surviving) into intensity_fn
    // (dropped, chain-resolves to phase_weight, which survives).
    { source: 'p2', target: 'intensity_fn', edgeType: 'data' },
    // Duplicate of the above once both re-route to the same pair —
    // deduped.
    { source: 'p2', target: 'angular_tensors', edgeType: 'data' },
    // Untouched by any drop.
    { source: 'q_of', target: 'p2', edgeType: 'call' },
  ];
  const result = rewriteEdgesForCollapse(edges, dropped, survivingNodeIds);
  assert.deepEqual(
    result.map((e) => [e.source, e.target, e.edgeType]).sort(),
    [
      ['p2', 'phase_weight', 'data'],
      ['q_of', 'p2', 'call'],
    ],
  );
  // Every surviving edge's endpoints are, by construction, in
  // survivingNodeIds — this is the property whose absence crashed
  // cytoscape in #176.
  for (const e of result) {
    assert.ok(survivingNodeIds.has(e.source), `dangling source ${e.source}`);
    assert.ok(survivingNodeIds.has(e.target), `dangling target ${e.target}`);
  }
});

test('rewriteEdgesForCollapse drops an edge whose resolved endpoint still is not a surviving node (defensive net)', () => {
  // No entry for 'ghost' at all — simulates a future gap in the drop
  // accounting rather than the nested-chain case above.
  const dropped = new Map();
  const survivingNodeIds = new Set(['a']);
  const edges = [{ source: 'a', target: 'ghost', edgeType: 'data' }];
  assert.deepEqual(rewriteEdgesForCollapse(edges, dropped, survivingNodeIds), []);
});
