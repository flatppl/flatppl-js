'use strict';

// Regression: a measure-bodied ARROW lambda `(args) -> <measure>` is a
// kernel (spec §04: a lambda is `functionof` sugar; the engine treats
// "functionof on a measure" as a kernel, analyzer.ts). Broadcasting it
// must classify as a kernel-broadcast, NOT a deterministic value
// broadcast.
//
// Before the fix the arrow form lowered to `functionof(<measure>, …)`
// WITHOUT the `lawof` wrapper that `kernelof` inserts
// (`kernelof(x) ≡ functionof(lawof(x))`), and every composite
// kernel-broadcast detector in kernel-broadcast-shape.ts hard-required
// `body.op === 'lawof'`. So `p ~ beta_row_K.(a, b)` classified as
// kind=evaluate, the measure body was inlined into the value evaluator,
// and the worker threw "call op 'iid' not evaluable in sampler
// context". Three sub-gaps, all fixed:
//   1. detectors peel an optional `lawof` so the arrow form is
//      recognised (kernel-broadcast-shape.peelKernelBody).
//   2. positional inner-dist args (`Beta(a_g, b_g)`) normalise to named
//      kwargs for the executor (distKwargsWithPositional).
//   3. positional OUTER broadcast args (`K.(a, b)`) bind to the
//      kernel's surface params by position
//      (mat-broadcast._normalizeCompositeBroadcastArgs).

const { test } = require('node:test');
const assert = require('node:assert');
const { makeMatCtx, expectPlottable } = require('./_materialise-helpers.ts');

test('arrow-form iid-composite kernel broadcast (positional dist args) materialises', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  await expectPlottable(ctx, 'p');
});

test('arrow-form iid-composite kernel broadcast (kwarg dist args) materialises', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(alpha = a_g, beta = b_g), N)
p ~ beta_row_K.(a, b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  await expectPlottable(ctx, 'p');
});

// The litter hierarchical Beta-Binomial (the reported repro). The arrow
// kernel `beta_row_K` now samples and the prior (a record over it)
// materialises — the originally-reported "iid not evaluable" crash is
// gone. The full posterior additionally needs broadcast over 2D [G, N]
// collection args (the binomial likelihood) on BOTH the sample and
// density sides, which is a separate v1 broadcast-shape limitation
// (kernel/dist broadcast "supports scalar / [K] / [N] / [N, K]"), not
// the arrow-kernel recognition bug fixed here.
test('litter Beta-Binomial: arrow kernel + prior materialise (reported crash fixed)', async () => {
  const src = `
G = 2
N = 16
n_data = [[13,12,9,9,8,8,13,12,10,10,9,13,5,7,10,10],[12,11,10,9,11,10,10,9,9,5,9,7,10,6,10,7]]
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, G)
mu ~ iid(Beta(1, 1), G)
a = mu .* a_plus_b
b = (1 .- mu) .* a_plus_b
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 128 });
  // The exact thing that used to throw "iid not evaluable in sampler
  // context" — both now succeed.
  await expectPlottable(ctx, 'p');
  const prior = await ctx.getMeasure('prior');
  assert.ok(prior, 'prior should materialise');
});
