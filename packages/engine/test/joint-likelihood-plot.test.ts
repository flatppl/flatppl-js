'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const SRC = `
n = elementof(cartpow(posreals, 2))
m = functionof(Poisson.(n .* 50.0))
L1 = likelihoodof(m, [48.0, 52.0])
L2 = likelihoodof(m, [55.0, 49.0])
J = joint_likelihood(L1, L2)
J_score = logdensityof(J, record(n = [1.0, 1.0]))
`;

test('joint_likelihood: plottable signature with unioned axes', () => {
  const proc = processSource(SRC);
  const built = orchestrator.buildDerivations(proc.bindings);
  const sig = orchestrator.signatureOf('J', built.bindings);
  assert.ok(sig && sig.kind === 'likelihood' && sig.terms && sig.terms.length === 2,
    `expected likelihood sig with 2 terms; got ${JSON.stringify(sig)}`);
  assert.deepEqual(orchestrator.distributeAxes(sig).map((a: any) => a.key), ['n[1]', 'n[2]']);
});

test('joint_likelihood: signature satisfies buildPlotPlan\'s callable-branch gate', () => {
  // buildPlotPlan (viewer plot-plan.ts) rejects a likelihood binding unless its
  // signature has a body OR terms AND ≥1 axis. A joint signature has NO `body`
  // (it carries `terms`), so the gate MUST accept `terms` — regressing the
  // signature to drop `terms`, or the gate to require `body`, makes the joint
  // unplottable ("Not plottable"). This pins the engine-side half of that gate.
  const proc = processSource(SRC);
  const built = orchestrator.buildDerivations(proc.bindings);
  const sig = orchestrator.signatureOf('J', built.bindings);
  assert.ok(sig && (sig.body || sig.terms), 'joint sig must satisfy the (body || terms) gate');
  assert.ok(orchestrator.distributeAxes(sig).length > 0, 'joint sig must have ≥1 sweep axis');
});

test('joint_likelihood: term-sum equals the joint logdensity (oracle)', async () => {
  // The engine's logdensityof(J, θ) is the independent oracle for the per-term-sum design.
  // Both must agree to 1e-9.
  const { ctx } = ctxFor(SRC, 1);
  const mJ = await ctx.getMeasure('J_score');
  const joint: number = (mJ.samples ?? (mJ.value && mJ.value.data))[0];

  // Score the two individual terms at the same point.
  const SRC_L1 = SRC + '\nL1_score = logdensityof(L1, record(n = [1.0, 1.0]))\n';
  const SRC_L2 = SRC + '\nL2_score = logdensityof(L2, record(n = [1.0, 1.0]))\n';
  const { ctx: ctx1 } = ctxFor(SRC_L1, 1);
  const { ctx: ctx2 } = ctxFor(SRC_L2, 1);
  const mL1 = await ctx1.getMeasure('L1_score');
  const mL2 = await ctx2.getMeasure('L2_score');
  const v1: number = (mL1.samples ?? (mL1.value && mL1.value.data))[0];
  const v2: number = (mL2.samples ?? (mL2.value && mL2.value.data))[0];
  const termSum = v1 + v2;
  const delta = Math.abs(joint - termSum);

  assert.ok(delta < 1e-9,
    `joint logdensity ${joint} vs term-sum ${termSum} (Δ ${delta})`);
});
