'use strict';

// A fixed-phase binding whose evaluation throws at run time has no value. A
// read of it, or of anything that depends on it, must report the binding and
// the cause. It once reported "no derivation for '<name>'", a TypeError, or a
// false "engine gap" for each dependent.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const { FixedValues } = require('../fixed-values.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const NOT_PD = "Fixed-phase binding 'lp' failed to evaluate: MvNormal: parameter 'cov' = [1, 2, 2, 1] "
  + 'is outside its declared value set positive definite matrices (spec §08)';
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
  assert.throws(() => fv.get('dep'), { message: "Fixed-phase binding 'bad' failed to evaluate: boom" });
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

// A direct `fixedValues.get` of a failed binding once returned `undefined`.
// The analyzer now refuses this vector-of-vectors `cov` (#532). The build runs
// anyway here, so the run-time failure still reaches the reader. Oracle for
// the matrix case: scipy.stats.multivariate_normal([0, 1], [[2, .5], [.5, 1]])
// .logpdf([0.3, 0.7]) = -2.2205421032341994.
test('fixedValues.get of a failed binding throws the named cause', () => {
  const build = (cov: string) => {
    const r = engine.processSource(`flatppl_compat = "0.1"
lp = builtin_logdensityof(MvNormal, record(mu = [0.0, 1.0], cov = ${cov}), [0.3, 0.7])
y = lp + 1.0
`);
    return orchestrator.buildDerivations(r.bindings, r.loweredModule).fixedValues;
  };
  const failed = build('[[2.0, 0.5], [0.5, 1.0]]');
  const cause = /^Error: Fixed-phase binding 'lp' failed to evaluate: builtin_logdensityof\(MvNormal\): param 'cov': argument is a vector-of-vectors/;
  for (const name of ['lp', 'y']) {
    assert.equal(failed.has(name), false, name);
    assert.throws(() => failed.get(name), cause, name);
  }
  const valid = build('rowstack([[2.0, 0.5], [0.5, 1.0]])');
  assert.ok(Math.abs(valid.get('lp') - -2.2205421032341994) < 1e-12, String(valid.get('lp')));
  assert.ok(Math.abs(valid.get('y') - -1.2205421032341994) < 1e-12, String(valid.get('y')));
});

test('resolveIRToValue of a failed binding reports its recorded cause', () => {
  const r = engine.processSource(`flatppl_compat = "0.1"
a = [1.0, 2.0]
b = a[5]
c = a[2]
`);
  const built = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
  assert.throws(() => orchestrator.resolveIRToValue(ref('b'), built.bindings, built.fixedValues),
    { message: OUT_OF_BOUNDS });
  assert.equal(orchestrator.resolveIRToValue(ref('c'), built.bindings, built.fixedValues), 2);
});

test('a latent whose law fails to lower is refused, not dropped', () => {
  const { collectLatents } = require('../model-spec.ts');
  const bindings = new Map<string, any>([
    ['x', { type: 'draw', deps: [], node: { value: { type: 'NoSuchNode' } } }],
  ]);
  assert.throws(() => collectLatents('x', { bindings }),
    { message: "latent 'x' could not be lowered: lower: unsupported AST node type 'NoSuchNode'" });
});
