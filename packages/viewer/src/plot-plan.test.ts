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

const { buildPlotPlan, fixedTupleColumnMode, tupleAxisLabelsFromIR } = await import('./plot-plan.ts');

// fixedTupleColumnMode narrows a fixed-phase `tuple` type to the
// paired-columns plan: every element must be a rank-1 array, all of
// the same literal length.

test('fixedTupleColumnMode: true for a tuple of two equal-length rank-1 arrays', () => {
  const t = { kind: 'tuple', elems: [
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
  ] };
  assert.equal(fixedTupleColumnMode(t), true);
});

test('fixedTupleColumnMode: false for a 2-tuple of scalars (single point, stays text)', () => {
  const t = { kind: 'tuple', elems: [{ kind: 'real' }, { kind: 'real' }] };
  assert.equal(fixedTupleColumnMode(t), false);
});

test('fixedTupleColumnMode: false for a mixed tuple (one scalar, one array)', () => {
  const t = { kind: 'tuple', elems: [
    { kind: 'real' },
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
  ] };
  assert.equal(fixedTupleColumnMode(t), false);
});

test('fixedTupleColumnMode: false when array element lengths differ', () => {
  const t = { kind: 'tuple', elems: [
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
    { kind: 'array', rank: 1, shape: [7], elem: { kind: 'real' } },
  ] };
  assert.equal(fixedTupleColumnMode(t), false);
});

test('fixedTupleColumnMode: false for a rank-2 array element', () => {
  const t = { kind: 'tuple', elems: [
    { kind: 'array', rank: 2, shape: [3, 3], elem: { kind: 'real' } },
    { kind: 'array', rank: 1, shape: [3], elem: { kind: 'real' } },
  ] };
  assert.equal(fixedTupleColumnMode(t), false);
});

test('fixedTupleColumnMode: false for a dynamic-length array element', () => {
  const t = { kind: 'tuple', elems: [
    { kind: 'array', rank: 1, shape: ['%dynamic'], elem: { kind: 'real' } },
    { kind: 'array', rank: 1, shape: ['%dynamic'], elem: { kind: 'real' } },
  ] };
  assert.equal(fixedTupleColumnMode(t), false);
});

// tupleAxisLabelsFromIR reads the RHS IR of a literal tuple
// construction (`{op:'tuple', args}`): a bare self-ref labels by
// binding name, anything else falls back to `name[i+1]`.

test('tupleAxisLabelsFromIR: refs label by binding name', () => {
  const ir = { kind: 'call', op: 'tuple', args: [
    { kind: 'ref', ns: 'self', name: 'xs' },
    { kind: 'ref', ns: 'self', name: 'ys' },
  ] };
  assert.deepEqual(tupleAxisLabelsFromIR('paired', ir), ['xs', 'ys']);
});

test('tupleAxisLabelsFromIR: a non-ref element falls back to name[i+1]', () => {
  const ir = { kind: 'call', op: 'tuple', args: [
    { kind: 'ref', ns: 'self', name: 'xs' },
    { kind: 'call', op: 'add', args: [] },
  ] };
  assert.deepEqual(tupleAxisLabelsFromIR('paired', ir), ['xs', 'paired[2]']);
});

test('tupleAxisLabelsFromIR: null when the RHS is not a literal tuple construction', () => {
  assert.equal(tupleAxisLabelsFromIR('paired', { kind: 'ref', ns: 'self', name: 'other' }), null);
  assert.equal(tupleAxisLabelsFromIR('paired', null), null);
});

// buildPlotPlan routing: a fixed-phase `tuple` binding whose type is
// paired columns takes mode 'tuple' (with provenance axisLabels when
// the RHS is a literal tuple construction); everything else — a
// 2-tuple of scalars, a mixed tuple — keeps the 'fixed-record' text
// path unchanged.

function baseCtx(name, liftedIr) {
  const liftedBindings = new Map();
  if (liftedIr !== undefined) liftedBindings.set(name, { ir: liftedIr });
  return {
    derivationsState: {
      derivations: {},
      fixedValues: new Set([name]),
      discrete: {},
      bindings: liftedBindings,
    },
    currentBindings: new Map(),
  };
}

test('buildPlotPlan: fixed tuple of paired columns routes to mode "tuple" with provenance labels', () => {
  const name = 'paired';
  const inferredType = { kind: 'tuple', elems: [
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
    { kind: 'array', rank: 1, shape: [5], elem: { kind: 'real' } },
  ] };
  const ir = { kind: 'call', op: 'tuple', args: [
    { kind: 'ref', ns: 'self', name: 'xs' },
    { kind: 'ref', ns: 'self', name: 'ys' },
  ] };
  const binding = { name, type: 'const', phase: 'fixed', inferredType };
  const ctx = baseCtx(name, ir);
  const plan = buildPlotPlan(ctx, binding);
  assert.equal(plan.mode, 'tuple');
  assert.deepEqual(plan.axisLabels, ['xs', 'ys']);
});

test('buildPlotPlan: fixed 2-tuple of scalars stays mode "fixed-record" (single point, no heatmap)', () => {
  const name = 'point';
  const inferredType = { kind: 'tuple', elems: [{ kind: 'real' }, { kind: 'real' }] };
  const binding = { name, type: 'const', phase: 'fixed', inferredType };
  const ctx = baseCtx(name, null);
  const plan = buildPlotPlan(ctx, binding);
  assert.equal(plan.mode, 'fixed-record');
  assert.equal(plan.axisLabels, undefined);
});

test('buildPlotPlan: a fixed record keeps mode "fixed-record" (unaffected by the tuple narrowing)', () => {
  const name = 'rec';
  const inferredType = { kind: 'record', fields: { a: { kind: 'real' } } };
  const binding = { name, type: 'const', phase: 'fixed', inferredType };
  const ctx = baseCtx(name, null);
  const plan = buildPlotPlan(ctx, binding);
  assert.equal(plan.mode, 'fixed-record');
});
