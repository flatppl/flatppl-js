'use strict';

// A fixed-phase binding whose evaluation throws at run time has no value. A
// read of it, or of anything that depends on it, must report the binding and
// the cause. It once reported "no derivation for '<name>'", a TypeError, or a
// false "engine gap" for each dependent.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { FixedValues } = require('../fixed-values.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const NOT_PD = "Fixed-phase binding 'lp' failed to evaluate: lower_cholesky: matrix is not positive definite";
const OUT_OF_BOUNDS = "Fixed-phase binding 'b' failed to evaluate: evaluateExpr: get index 5 out of bounds for length 2";

async function rejection(ctx: any, name: string): Promise<string> {
  try {
    await ctx.getMeasure(name);
  } catch (e: any) {
    return e.message;
  }
  throw new Error(`'${name}' materialised, expected a rejection`);
}

function errors(built: any): string[] {
  return built.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

test('a primitive with no derivation that throws reports the binding and the cause', async () => {
  const { ctx, built } = makeMatCtx(`flatppl_compat = "0.1"
lp = builtin_logdensityof(MvNormal, record(mu = [0.0, 0.0], cov = rowstack([[1.0, 2.0], [2.0, 1.0]])), [0.3, 0.7])
y = lp + 1.0
`);
  assert.ok((await rejection(ctx, 'lp')).startsWith(NOT_PD));
  assert.ok((await rejection(ctx, 'y')).startsWith(NOT_PD));
  // One located error at the root, no "engine gap" for either binding.
  const errs = errors(built);
  assert.equal(errs.length, 1, errs.join('\n'));
  assert.ok(errs[0].startsWith(NOT_PD));
});

test('an evaluated fixed value that throws reports the binding to every reader', async () => {
  const { ctx, built } = makeMatCtx(`flatppl_compat = "0.1"
a = [1.0, 2.0]
b = a[5]
y = b + 1.0
m = Normal(mu = b, sigma = 1.0)
z ~ Normal(mu = b, sigma = 1.0)
s, st = rand(Normal(mu = b, sigma = 1.0), rnginit([1, 2, 3, 4]))
`);
  for (const name of ['b', 'y', 'm', 'z', 's']) {
    assert.equal(await rejection(ctx, name), OUT_OF_BOUNDS, name);
  }
  assert.deepEqual(errors(built), [OUT_OF_BOUNDS]);
});

test('a failure read through the measure-context resolver keeps the root binding', () => {
  const bindings = new Map<string, any>([
    ['bad', { phase: 'fixed', ir: { name: 'bad', refs: [] } }],
    ['dep', { phase: 'fixed', ir: { name: 'dep', refs: [] } }],
  ]);
  const diagnostics: any[] = [];
  const fv = new FixedValues({
    bindings, derivations: {},
    resolveMeasureRef: () => null, isMeasureBinding: () => false,
    samplerLib: {
      evaluateExpr: (ir: any, env: any) => {
        if (ir.name === 'bad') throw new Error('boom');
        return env.__resolveValueRef('bad', {})[0];
      },
      walk: () => { throw new Error('not reached'); },
    },
    expandMeasureIR: () => null, collectSelfRefs: (ir: any) => new Set(ir.refs), lowerExpr: (x: any) => x,
    diagnostics, bindingLoc: () => undefined,
  });
  assert.equal(fv.get('dep'), undefined);
  assert.equal(fv.failure('dep'), "Fixed-phase binding 'bad' failed to evaluate: boom");
  assert.deepEqual(diagnostics.map((d: any) => d.message), ["Fixed-phase binding 'bad' failed to evaluate: boom"]);
});

test('a later success clears the failure', () => {
  let fail = true;
  const fv = new FixedValues({
    bindings: new Map<string, any>([['t', { phase: 'fixed', ir: { refs: [] } }]]), derivations: {},
    resolveMeasureRef: () => null, isMeasureBinding: () => false,
    samplerLib: { evaluateExpr: () => { if (fail) throw new Error('boom'); return 1; }, walk: () => null },
    expandMeasureIR: () => null, collectSelfRefs: () => new Set(), lowerExpr: (x: any) => x,
  });
  assert.equal(fv.failure('t'), "Fixed-phase binding 't' failed to evaluate: boom");
  fail = false;
  assert.equal(fv.failure('t'), undefined);
  assert.equal(fv.get('t'), 1);
});

// The reverse error: a binding with no fixed value by construction is not a
// failure. Both cases below drew a false "failed to evaluate" error.
test('a kernel broadcast is a measure, not a failed fixed value', async () => {
  const { ctx, built } = makeMatCtx(`flatppl_compat = "0.1"
means = [0.0, 10.0]
x ~ broadcast(Normal, means, [0.001, 0.001])
`, { sampleCount: 4 });
  assert.deepEqual((await ctx.getMeasure('x')).dims, [2]);
  assert.deepEqual(errors(built), []);
});

test('a value that reads a draw is not a failed fixed value', async () => {
  const { ctx, built } = makeMatCtx(`flatppl_compat = "0.1"
x ~ Dirac([1, 2])
sx = sum(x)
`, { sampleCount: 3 });
  assert.deepEqual(Array.from((await ctx.getMeasure('sx')).samples), [3, 3, 3]);
  assert.deepEqual(errors(built), []);
});

// A classifier that needs a failed fixed value declines, so the draw has no
// derivation. The draw reports the failed binding, not "no derivation".
test('a draw whose iid size failed reports the failed size', async () => {
  const { ctx, built } = makeMatCtx(`flatppl_compat = "0.1"
a = [1, 2]
n = a[5]
y ~ iid(Normal(mu = 0.0, sigma = 1.0), n)
`, { sampleCount: 4 });
  const want = "Fixed-phase binding 'n' failed to evaluate: evaluateExpr: get index 5 out of bounds for length 2";
  assert.equal(await rejection(ctx, 'y'), want);
  assert.deepEqual(errors(built), [want]);
});
