'use strict';

// =====================================================================
// axis-stack.test.ts — P3a IR-axis-context annotation
// =====================================================================
//
// Pins `propagateAxisStack` (dissolver.ts, engine-concepts §18.2 /
// §20.10): the static outer-axis metadata attached to
// measure-op IR nodes after dissolution settles. The annotation is
// pure IR metadata; no runtime semantics depend on it yet, but the
// schema and the propagation are infrastructure for fusion thread (b)
// (kernel-broadcast fusion).
//
// Three classes of test:
//   1. Direct propagateAxisStack on synthetic IR — exercises the
//      _axisStackForIR shape recognition in isolation.
//   2. End-to-end via processSource — pins that annotated bindings
//      survive the dissolveBindings call in the normal pipeline.
//   3. Idempotency + non-annotation — re-running the pass on already-
//      annotated bindings is stable; non-axis-introducing IRs are
//      left untouched.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const dissolver = require('../dissolver.ts');
const { processSource } = require('../index.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function mkBindings(entries: Record<string, any>): Map<string, any> {
  const m = new Map<string, any>();
  for (const k in entries) m.set(k, entries[k]);
  return m;
}

// =====================================================================
// 1. Direct shape recognition (no bindings context — defaults)
// =====================================================================

test('axisStack: iid(M, literal n) → one iid entry with statically-known size', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'ref', ns: 'self', name: 'M' },
      { kind: 'lit', value: 10 },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 10 }]);
});

test('axisStack: iid(M, ref) → size carries the binding name when no inferred type', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'ref', ns: 'self', name: 'M' },
      { kind: 'ref', ns: 'self', name: 'n_count' },
    ],
  };
  const bindings = mkBindings({ n_count: { phase: 'fixed' } });
  const stack = dissolver._axisStackForIR(ir, bindings);
  assert.deepEqual(stack, [{ source: 'iid', size: 'n_count' }]);
});

test('axisStack: broadcast with function head → source=broadcast', () => {
  const ir = {
    kind: 'call', op: 'broadcast',
    args: [
      { kind: 'ref', ns: 'self', name: 'f' },
      { kind: 'ref', ns: 'self', name: 'xs' },
    ],
  };
  const bindings = mkBindings({
    f:  { inferredType: { kind: 'function' } },
    xs: { inferredType: { kind: 'array', rank: 1, shape: [7], elem: { kind: 'scalar', prim: 'real' } } },
  });
  const stack = dissolver._axisStackForIR(ir, bindings);
  assert.deepEqual(stack, [{ source: 'broadcast', size: 7, name: 'xs' }]);
});

test('axisStack: broadcast with kernel head → source=kernel_broadcast', () => {
  const ir = {
    kind: 'call', op: 'broadcast',
    args: [
      { kind: 'ref', ns: 'self', name: 'K' },
      { kind: 'ref', ns: 'self', name: 'data' },
    ],
  };
  const bindings = mkBindings({
    K:    { inferredType: { kind: 'kernel' } },
    data: { inferredType: { kind: 'array', rank: 1, shape: [50], elem: { kind: 'scalar', prim: 'real' } } },
  });
  const stack = dissolver._axisStackForIR(ir, bindings);
  assert.deepEqual(stack, [{ source: 'kernel_broadcast', size: 50, name: 'data' }]);
});

test('axisStack: broadcast with kwargs → uses first kwarg arg', () => {
  const ir = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: 'f' }],
    kwargs: { x: { kind: 'ref', ns: 'self', name: 'mu_vec' } },
  };
  const bindings = mkBindings({
    f:      { inferredType: { kind: 'function' } },
    mu_vec: { inferredType: { kind: 'array', rank: 1, shape: [4], elem: { kind: 'scalar', prim: 'real' } } },
  });
  const stack = dissolver._axisStackForIR(ir, bindings);
  assert.deepEqual(stack, [{ source: 'broadcast', size: 4, name: 'mu_vec' }]);
});

test('axisStack: aggregate with 2 output axes → 2 entries with axis names', () => {
  const ir = {
    kind: 'call', op: 'aggregate',
    args: [
      { kind: 'ref', ns: 'self', name: 'sum' },
      {
        kind: 'call', op: 'vector',
        args: [
          { kind: 'axis', name: 'i' },
          { kind: 'axis', name: 'k' },
        ],
      },
      { kind: 'call', op: 'mul', args: [/* doesn't matter for axisStack */] },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [
    { source: 'aggregate', size: '%dynamic', name: 'i' },
    { source: 'aggregate', size: '%dynamic', name: 'k' },
  ]);
});

test('axisStack: non-axis-introducing IR returns null', () => {
  const ir = { kind: 'call', op: 'add', args: [] };
  assert.equal(dissolver._axisStackForIR(ir, null), null);
});

test('axisStack: malformed iid (missing n) returns null', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [{ kind: 'ref', ns: 'self', name: 'M' }],
  };
  assert.equal(dissolver._axisStackForIR(ir, null), null);
});

// =====================================================================
// 1b. Axis-structure-preserving wrappers — produce the SAME stack as
//     the inner measure (P4 producer-recursion follow-up).
// =====================================================================

test('axisStack: normalize(iid(M, n)) → inner iid stack passes through', () => {
  const ir = {
    kind: 'call', op: 'normalize',
    args: [{
      kind: 'call', op: 'iid',
      args: [
        { kind: 'ref', ns: 'self', name: 'M' },
        { kind: 'lit', value: 8 },
      ],
    }],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 8 }]);
});

test('axisStack: truncate(iid(M, n), S) → inner iid stack passes through', () => {
  const ir = {
    kind: 'call', op: 'truncate',
    args: [
      {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 12 },
        ],
      },
      { kind: 'ref', ns: 'self', name: 'S' },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 12 }]);
});

test('axisStack: weighted(w, iid(M, n)) → inner iid stack passes through', () => {
  const ir = {
    kind: 'call', op: 'weighted',
    args: [
      { kind: 'lit', value: 2.5 },
      {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 9 },
        ],
      },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 9 }]);
});

test('axisStack: pushfwd(f, iid(M, n)) → inner iid stack passes through', () => {
  const ir = {
    kind: 'call', op: 'pushfwd',
    args: [
      { kind: 'ref', ns: 'self', name: 'f' },
      {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 16 },
        ],
      },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 16 }]);
});

test('axisStack: bayesupdate(L, prior) → prior stack passes through', () => {
  const ir = {
    kind: 'call', op: 'bayesupdate',
    args: [
      { kind: 'ref', ns: 'self', name: 'L' },
      {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 7 },
        ],
      },
    ],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 7 }]);
});

test('axisStack: nested wrappers — normalize(truncate(iid(M, n), S)) → still iid stack', () => {
  const ir = {
    kind: 'call', op: 'normalize',
    args: [{
      kind: 'call', op: 'truncate',
      args: [
        {
          kind: 'call', op: 'iid',
          args: [
            { kind: 'ref', ns: 'self', name: 'M' },
            { kind: 'lit', value: 20 },
          ],
        },
        { kind: 'ref', ns: 'self', name: 'S' },
      ],
    }],
  };
  const stack = dissolver._axisStackForIR(ir, null);
  assert.deepEqual(stack, [{ source: 'iid', size: 20 }]);
});

test('axisStack: wrapper around non-axis-introducing inner → null', () => {
  // normalize(Normal(0,1)) has no axis — should NOT fabricate one.
  const ir = {
    kind: 'call', op: 'normalize',
    args: [{
      kind: 'call', op: 'Normal',
      kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
    }],
  };
  assert.equal(dissolver._axisStackForIR(ir, null), null);
});

// =====================================================================
// 2. Through dissolveBindings (the full dissolver entry point)
// =====================================================================

test('dissolveBindings annotates iid IR with axisStack', () => {
  const bindings = mkBindings({
    X: {
      ir: {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 5 },
        ],
      },
      phase: 'stochastic',
    },
  });
  dissolver.dissolveBindings(bindings);
  const ir = bindings.get('X').ir;
  assert.deepEqual(ir.axisStack, [{ source: 'iid', size: 5 }]);
});

test('dissolveBindings annotates aggregate IR with output axes', () => {
  const bindings = mkBindings({
    C: {
      ir: {
        kind: 'call', op: 'aggregate',
        args: [
          { kind: 'ref', ns: 'self', name: 'sum' },
          {
            kind: 'call', op: 'vector',
            args: [
              { kind: 'axis', name: 'i' },
              { kind: 'axis', name: 'k' },
            ],
          },
          { kind: 'call', op: 'mul', args: [] },
        ],
      },
      phase: 'parameterised',
    },
  });
  dissolver.dissolveBindings(bindings);
  const ir = bindings.get('C').ir;
  assert.ok(Array.isArray(ir.axisStack));
  assert.equal(ir.axisStack.length, 2);
  assert.deepEqual(ir.axisStack.map((e: any) => e.name), ['i', 'k']);
});

// =====================================================================
// 3. Idempotency + non-annotation
// =====================================================================

test('axisStack: re-running propagateAxisStack is idempotent', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'ref', ns: 'self', name: 'M' },
      { kind: 'lit', value: 7 },
    ],
  };
  const bindings = mkBindings({ X: { ir, phase: 'stochastic' } });
  dissolver.propagateAxisStack(bindings);
  const firstIR = bindings.get('X').ir;
  assert.ok(firstIR.axisStack);
  dissolver.propagateAxisStack(bindings);
  // Same axisStack content, but the IR object may stay identical
  // (the pass short-circuits when the annotation is already equal).
  const secondIR = bindings.get('X').ir;
  assert.deepEqual(secondIR.axisStack, firstIR.axisStack);
  assert.strictEqual(secondIR, firstIR,
    'idempotent: IR identity preserved on second pass');
});

test('axisStack: bindings without axis-introducing IR are untouched', () => {
  const ir = { kind: 'call', op: 'add', args: [] };
  const bindings = mkBindings({ Y: { ir, phase: 'fixed' } });
  dissolver.propagateAxisStack(bindings);
  assert.equal(bindings.get('Y').ir.axisStack, undefined);
  assert.strictEqual(bindings.get('Y').ir, ir,
    'non-axis IR object identity preserved');
});

// =====================================================================
// 4. Nested annotation
// =====================================================================

test('axisStack nested: iid(iid(M, n_inner), n_outer) inline produces 2-entry stack', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [
      {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'ref', ns: 'self', name: 'M' },
          { kind: 'lit', value: 7 },
        ],
      },
      { kind: 'lit', value: 4 },
    ],
  };
  const bindings = mkBindings({ X: { ir, phase: 'stochastic' } });
  dissolver.propagateAxisStack(bindings);
  const stack = bindings.get('X').ir.axisStack;
  assert.deepEqual(stack, [
    { source: 'iid', size: 4 },
    { source: 'iid', size: 7 },
  ]);
});

test('axisStack nested: iid(M_ref, n) inherits via cross-binding ref (fixed-point)', () => {
  // M_inner is itself an annotated iid; M_outer wraps it via ref.
  // The fixed-point iteration of propagateAxisStack ensures M_outer
  // picks up M_inner's stack after M_inner is annotated.
  const irInner = {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'ref', ns: 'self', name: 'M' },
      { kind: 'lit', value: 7 },
    ],
  };
  const irOuter = {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'ref', ns: 'self', name: 'M_inner' },
      { kind: 'lit', value: 4 },
    ],
  };
  const bindings = mkBindings({
    M_inner: { ir: irInner, phase: 'stochastic' },
    M_outer: { ir: irOuter, phase: 'stochastic' },
  });
  dissolver.propagateAxisStack(bindings);
  assert.deepEqual(bindings.get('M_inner').ir.axisStack, [
    { source: 'iid', size: 7 },
  ]);
  assert.deepEqual(bindings.get('M_outer').ir.axisStack, [
    { source: 'iid', size: 4 },
    { source: 'iid', size: 7 },
  ]);
});

test('axisStack nested: iid(broadcast(K, args), n) produces iid + kernel_broadcast entries', () => {
  const ir = {
    kind: 'call', op: 'iid',
    args: [
      {
        kind: 'call', op: 'broadcast',
        args: [
          { kind: 'ref', ns: 'self', name: 'K' },
          { kind: 'ref', ns: 'self', name: 'data' },
        ],
      },
      { kind: 'lit', value: 3 },
    ],
  };
  const bindings = mkBindings({
    K:    { inferredType: { kind: 'kernel' } },
    data: { inferredType: { kind: 'array', rank: 1, shape: [50], elem: { kind: 'scalar', prim: 'real' } } },
    Y:    { ir, phase: 'stochastic' },
  });
  dissolver.propagateAxisStack(bindings);
  assert.deepEqual(bindings.get('Y').ir.axisStack, [
    { source: 'iid', size: 3 },
    { source: 'kernel_broadcast', size: 50, name: 'data' },
  ]);
});

// ---------------------------------------------------------------------
// Enabler (§22.4): a kernel-broadcast records the INNER axis its kernel
// body introduces — _axisStackForIR recurses into the kernel head's
// reified body. This is the static K_inner the (deferred) nested /
// joint / jointchain batch-flatten folds consume (TODO Phase 8).
// ---------------------------------------------------------------------

test('axisStack: nested kernel-broadcast records [kernel_broadcast K_outer, broadcast K_inner]', () => {
  // y = broadcast(patient_kernel, sigma_g = sigmas_per_patient)
  // patient_kernel = kernelof(broadcast(Normal, mu = visit_means,
  //                                     sigma = sigma_g), sigma_g = sigma_g)
  const patientKernelIR = {
    kind: 'call', op: 'functionof', params: ['sigma_g'],
    body: {
      kind: 'call', op: 'lawof',
      args: [{
        kind: 'call', op: 'broadcast',
        args: [{ kind: 'ref', ns: 'self', name: 'Normal' }],
        kwargs: {
          mu: { kind: 'ref', ns: 'self', name: 'visit_means' },
          sigma: { kind: 'ref', ns: 'self', name: 'sigma_g' },
        },
      }],
    },
  };
  const yIR = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: 'patient_kernel' }],
    kwargs: { sigma_g: { kind: 'ref', ns: 'self', name: 'sigmas_per_patient' } },
  };
  const bindings = mkBindings({
    patient_kernel:     { ir: patientKernelIR, inferredType: { kind: 'kernel' } },
    visit_means:        { inferredType: { kind: 'array', rank: 1, shape: [4], elem: { kind: 'scalar', prim: 'real' } } },
    sigmas_per_patient: { inferredType: { kind: 'array', rank: 1, shape: [3], elem: { kind: 'scalar', prim: 'real' } } },
    y:                  { ir: yIR, phase: 'stochastic' },
  });
  dissolver.propagateAxisStack(bindings);
  assert.deepEqual(bindings.get('y').ir.axisStack, [
    { source: 'kernel_broadcast', size: 3, name: 'sigmas_per_patient' },
    { source: 'broadcast', size: 4, name: 'visit_means' },
  ]);
});

test('axisStack: iid-composite kernel-broadcast records [kernel_broadcast K, iid D]', () => {
  // y = broadcast(obs_kernel, mu = means);  obs_kernel = kernelof(iid(Normal(mu), D), mu = mu)
  const obsKernelIR = {
    kind: 'call', op: 'functionof', params: ['mu'],
    body: {
      kind: 'call', op: 'lawof',
      args: [{
        kind: 'call', op: 'iid',
        args: [
          { kind: 'call', op: 'Normal', kwargs: { mu: { kind: 'ref', ns: 'self', name: 'mu' }, sigma: { kind: 'lit', value: 1 } } },
          { kind: 'lit', value: 5 },
        ],
      }],
    },
  };
  const yIR = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: 'obs_kernel' }],
    kwargs: { mu: { kind: 'ref', ns: 'self', name: 'means' } },
  };
  const bindings = mkBindings({
    obs_kernel: { ir: obsKernelIR, inferredType: { kind: 'kernel' } },
    means:      { inferredType: { kind: 'array', rank: 1, shape: [6], elem: { kind: 'scalar', prim: 'real' } } },
    y:          { ir: yIR, phase: 'stochastic' },
  });
  dissolver.propagateAxisStack(bindings);
  assert.deepEqual(bindings.get('y').ir.axisStack, [
    { source: 'kernel_broadcast', size: 6, name: 'means' },
    { source: 'iid', size: 5 },
  ]);
});

// =====================================================================
// 5. resolveOuterAxisSize — materialise-time symbolic-size resolution
// =====================================================================
//
// The symbolic-ladder fix: literal sizes pass through; a SYMBOLIC
// binding-name size (recorded when a collection arg's length isn't a
// compile-time literal but the binding is fixed-phase) const-folds
// against ctx.fixedValues; a genuinely '%dynamic' / unresolvable size
// returns null (the nested-broadcast fold then refuses rather than
// folding a wrong size). Unlike `outerAxisSize`, which returns null for
// ALL symbolic sizes.

const { resolveOuterAxisSize, outerAxisSize: _outerAxisSizeLit } = require('../axis-stack.ts');

const mkFixed = (entries: Record<string, any>) => new Map(Object.entries(entries));

test('resolveOuterAxisSize: literal sizes pass through', () => {
  const stack = [{ source: 'kernel_broadcast', size: 4 }, { source: 'broadcast', size: 3 }];
  const ctx = { bindings: mkBindings({}) };
  assert.equal(resolveOuterAxisSize(stack, 'kernel_broadcast', ctx), 4);
  assert.equal(resolveOuterAxisSize(stack, 'broadcast', ctx), 3);
});

test('resolveOuterAxisSize: symbolic binding-name resolves from fixedValues', () => {
  const stack = [{ source: 'broadcast', size: 'k' }];
  const ctx = { bindings: mkBindings({ k: {} }), fixedValues: mkFixed({ k: 7 }) };
  assert.equal(resolveOuterAxisSize(stack, 'broadcast', ctx), 7);
  // outerAxisSize (literal-only) gives null for the same symbolic stack.
  assert.equal(_outerAxisSizeLit(stack, 'broadcast'), null);
});

test('resolveOuterAxisSize: %dynamic is unresolvable → null', () => {
  const stack = [{ source: 'broadcast', size: '%dynamic' }];
  const ctx = { bindings: mkBindings({}), fixedValues: mkFixed({}) };
  assert.equal(resolveOuterAxisSize(stack, 'broadcast', ctx), null);
});

test('resolveOuterAxisSize: symbolic name with no fixed value → null', () => {
  const stack = [{ source: 'broadcast', size: 'k' }];
  const ctx = { bindings: mkBindings({ k: {} }), fixedValues: mkFixed({}) };
  assert.equal(resolveOuterAxisSize(stack, 'broadcast', ctx), null);
});

test('resolveOuterAxisSize: absent axis / null stack → null', () => {
  assert.equal(resolveOuterAxisSize([{ source: 'iid', size: 5 }], 'kernel_broadcast', { bindings: mkBindings({}) }), null);
  assert.equal(resolveOuterAxisSize(null, 'broadcast', { bindings: mkBindings({}) }), null);
});
