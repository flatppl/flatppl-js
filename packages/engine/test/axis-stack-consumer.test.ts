'use strict';

// =====================================================================
// axis-stack-consumer.test.ts — P4 advisory consumption of axisStack
// =====================================================================
//
// Pins the contract added by P4 of the broadcast / aggregate /
// batching consolidation (TODO-flatppl-js.md "In-flight P1-P9"):
//
//   - `axis-stack.bindingAxisStack(name, ctx)` reads the axisStack
//     annotation from a binding's IR (set by dissolver.propagateAxis-
//     Stack); returns null when absent.
//   - `axis-stack.outerAxisSize(stack, sourceTag)` extracts a specific
//     axis's size from the stack (integer literal only; symbolic
//     sizes return null so consumers fall back to runtime resolution).
//   - The advisory contract: consumers PREFER axisStack but fall back
//     to runtime shape-sniffing when absent. This pins that the
//     module behaves gracefully in both modes.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const axisStackMod = require('../axis-stack.ts');

// =====================================================================
// 1. getAxisStack — null-safe reader
// =====================================================================

test('axis-stack: getAxisStack returns null for IR without annotation', () => {
  assert.equal(axisStackMod.getAxisStack(null), null);
  assert.equal(axisStackMod.getAxisStack({}), null);
  assert.equal(axisStackMod.getAxisStack({ axisStack: [] }), null);
  assert.equal(axisStackMod.getAxisStack({ axisStack: 'wrong-type' }), null);
});

test('axis-stack: getAxisStack returns the array verbatim when present', () => {
  const stack = [{ source: 'iid', size: 10 }];
  const ir = { kind: 'call', op: 'iid', axisStack: stack };
  assert.equal(axisStackMod.getAxisStack(ir), stack);
});

// =====================================================================
// 2. bindingAxisStack — reads from ctx.bindings.get(name).ir
// =====================================================================

test('axis-stack: bindingAxisStack returns null when ctx lacks bindings', () => {
  assert.equal(axisStackMod.bindingAxisStack('foo', null), null);
  assert.equal(axisStackMod.bindingAxisStack('foo', {}), null);
  assert.equal(axisStackMod.bindingAxisStack('foo', { bindings: {} }), null);
});

test('axis-stack: bindingAxisStack reads from binding.ir.axisStack', () => {
  const stack = [{ source: 'kernel_broadcast', size: 5 }];
  const ctx = {
    bindings: new Map([['foo', { ir: { axisStack: stack } }]]),
  };
  assert.equal(axisStackMod.bindingAxisStack('foo', ctx), stack);
  // Missing binding → null.
  assert.equal(axisStackMod.bindingAxisStack('bar', ctx), null);
});

// =====================================================================
// 3. outerAxisSize — extract by source tag
// =====================================================================

test('axis-stack: outerAxisSize returns integer literal size', () => {
  const stack = [
    { source: 'iid', size: 10 },
    { source: 'kernel_broadcast', size: 5 },
  ];
  assert.equal(axisStackMod.outerAxisSize(stack, 'iid'), 10);
  assert.equal(axisStackMod.outerAxisSize(stack, 'kernel_broadcast'), 5);
});

test('axis-stack: outerAxisSize returns null for symbolic size', () => {
  // Symbolic sizes (binding-ref name or '%dynamic') return null so
  // consumers fall back to runtime resolution.
  const stack = [
    { source: 'iid', size: 'n_samples' },
    { source: 'kernel_broadcast', size: '%dynamic' },
  ];
  assert.equal(axisStackMod.outerAxisSize(stack, 'iid'), null);
  assert.equal(axisStackMod.outerAxisSize(stack, 'kernel_broadcast'), null);
});

test('axis-stack: outerAxisSize returns null when source tag not present', () => {
  const stack = [{ source: 'iid', size: 10 }];
  assert.equal(axisStackMod.outerAxisSize(stack, 'kernel_broadcast'), null);
  assert.equal(axisStackMod.outerAxisSize(stack, 'aggregate'), null);
});

test('axis-stack: outerAxisSize returns null for null stack', () => {
  assert.equal(axisStackMod.outerAxisSize(null, 'iid'), null);
});

// =====================================================================
// 4. outermostIidAxis — convenience for matIid composite diagnostic
// =====================================================================

test('axis-stack: outermostIidAxis returns integer size for iid-top stack', () => {
  const stack = [{ source: 'iid', size: 10 }];
  assert.equal(axisStackMod.outermostIidAxis(stack), 10);
});

test('axis-stack: outermostIidAxis returns "dynamic" for symbolic iid size', () => {
  const stack = [{ source: 'iid', size: 'n_samples' }];
  assert.equal(axisStackMod.outermostIidAxis(stack), 'dynamic');
});

test('axis-stack: outermostIidAxis returns null when outer entry is not iid', () => {
  const stack = [{ source: 'kernel_broadcast', size: 5 }];
  assert.equal(axisStackMod.outermostIidAxis(stack), null);
});

test('axis-stack: outermostIidAxis returns null for empty / missing stack', () => {
  assert.equal(axisStackMod.outermostIidAxis(null), null);
  assert.equal(axisStackMod.outermostIidAxis([]), null);
});
