'use strict';

// ════════════════════════════════════════════════════════════════════════
// CLM Phase 1 — clm node + lowerMeasure, additive (assert-only).
// flatppl-dev/measure-lowering-unification-plan.md Phase 1.
//
// Gates the additive pass before any consumer reads it:
//  - SNAPSHOT: lowerMeasure(name).body === expandMeasure(name) on every
//    GREEN agreement fixture (lowering must not perturb a body that already
//    samples ≡ density today);
//  - ⊆ INVARIANT: every fed self-ref in body (callables excluded — they
//    resolve by name) is a declared input — now MEANINGFUL because prereq D
//    made collectSelfRefs descend .bijection;
//  - STRUCTURE: kchain ⇒ reduce={marginal, over:<prior>}; a plain product
//    measure ⇒ reduce=null; record-base prior + fields are declared inputs
//    with shape descriptors (critique A + C); bayesupdate kernel params are
//    boundary inputs fed from the prior (the separate-prior idiom).
// ════════════════════════════════════════════════════════════════════════

const test = require('node:test');
const assert = require('node:assert');
const eng = require('../index.ts');
const der = require('../derivations.ts');
const { collectSelfRefs } = require('../ir-shared.ts');
const { lowerMeasure } = require('../clm.ts');

function ctxOf(src: string) {
  const b = der.buildDerivations(eng.processSource(src).bindings);
  return { derivations: b.derivations, bindings: b.bindings, fixedValues: b.fixedValues || new Map() };
}

// The fed-ref half of body: self-refs minus callable bindings (which the
// walker resolves by name, never feeds) — the exact set ⊆ must cover.
function fedSelfRefs(body: any, ctx: any): any[] {
  const isCallable = (n: string) => {
    const bb = ctx.bindings.get(n);
    return !!(bb && (bb.type === 'functionof' || bb.type === 'kernelof'
      || bb.type === 'fn' || bb.type === 'bijection'));
  };
  return Array.from(collectSelfRefs(body)).filter((n: any) => !isCallable(n));
}

function assertSubset(node: any, ctx: any, label: string) {
  const names = new Set(node.inputs.map((i: any) => i.name));
  for (const r of fedSelfRefs(node.body, ctx)) {
    assert.ok(names.has(r),
      `${label}: body self-ref '${r}' is not a declared clm input ` +
      `(⊆ invariant) — inputs=${JSON.stringify([...names])}`);
  }
}

const GREEN: Array<[string, string, string]> = [
  ['kchain hole-param', 'ch', `
a ~ Normal(0.0, 0.01)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
M = Normal(0.0, 10.0)
ch = kchain(M, K)`],
  ['M1 relabelled-prior kchain', 'ch', `
theta ~ Normal(0.0, 1.0)
shifted = theta + 20.0
M = lawof(shifted)
K = functionof(Normal(mu = shifted, sigma = 1.0), shifted = shifted)
ch = kchain(M, K)`],
  ['kchain marginalises the prior', 'ch', `
theta ~ Normal(0.0, 1.0)
K = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
prior = lawof(theta)
ch = kchain(prior, K)`],
  ['hole-kernel over a RECORD base prior', 'ch', `
mu_p = 0.0
joint_indep = joint(t1 = Normal(mu = mu_p, sigma = 1.0), t2 = Exponential(rate = 1.0))
K = functionof(Normal(mu = t1, sigma = t2), t1 = t1, t2 = t2)
ch = kchain(joint_indep, K)`],
];

for (const [id, name, src] of GREEN) {
  test(`[clm Phase 1] snapshot + ⊆ — ${id}`, () => {
    const ctx = ctxOf(src);
    const expanded = der.expandMeasure(name, ctx);
    const node = lowerMeasure(name, ctx);
    assert.ok(node && node.op === 'clm', 'lowerMeasure returns a clm node');
    // SNAPSHOT: body must not be perturbed on a green fixture.
    assert.strictEqual(JSON.stringify(node.body), JSON.stringify(expanded),
      `${id}: clm.body diverged from expandMeasure`);
    // STRUCTURE: every green fixture here is a kchain ⇒ marginal reduce.
    assert.ok(node.reduce && node.reduce.kind === 'marginal',
      `${id}: expected reduce={marginal}, got ${JSON.stringify(node.reduce)}`);
    assert.ok(node.reduce.over != null, `${id}: marginal reduce must record 'over'`);
    // Every input carries a shape descriptor (critique C).
    for (const i of node.inputs) {
      assert.ok(i.shape && typeof i.shape.kind === 'string' && i.shape.repeatTile,
        `${id}: input '${i.name}' missing a shape descriptor`);
    }
    assertSubset(node, ctx, id);
  });
}

test('[clm Phase 1] record-base prior + fields are declared boundary inputs (A+C)', () => {
  const ctx = ctxOf(GREEN[3][2]);
  const node = lowerMeasure('ch', ctx);
  const byName: Record<string, any> = {};
  for (const i of node.inputs) byName[i.name] = i;
  // The prior measure and BOTH record fields are inputs, fed from the prior.
  assert.ok(byName.joint_indep && byName.joint_indep.shape.kind === 'record');
  for (const f of ['t1', 't2']) {
    assert.ok(byName[f], `field '${f}' should be a declared input`);
    assert.strictEqual(byName[f].source.kind, 'boundary');
    assert.strictEqual(byName[f].source.from, 'joint_indep');
    assert.strictEqual(byName[f].source.field, f);
    assert.strictEqual(byName[f].shape.kind, 'scalar');
  }
});

test('[clm Phase 1] a plain product measure has reduce=null', () => {
  const ctx = ctxOf('m = Normal(0.0, 1.0)\n');
  const node = lowerMeasure('m', ctx);
  assert.ok(node && node.op === 'clm');
  assert.strictEqual(node.reduce, null);
  assertSubset(node, ctx, 'plain-normal');
});

test('[clm Phase 1] bayesupdate — kernel params are boundary inputs fed from the prior (separate-prior idiom)', () => {
  // The motivating bi1 shape: prior and likelihood defined separately.
  const ctx = ctxOf(`
theta ~ Normal(0.0, 2.0)
obs ~ Normal(mu = theta, sigma = 1.0)
fwd = kernelof(record(obs = obs), theta = theta)
prior = lawof(record(theta = theta))
L = likelihoodof(fwd, record(obs = 2.0))
post = bayesupdate(L, prior)`);
  const node = lowerMeasure('post', ctx);
  assert.ok(node && node.op === 'clm', 'post lowers to a clm node');
  const theta = node.inputs.find((i: any) => i.name === 'theta');
  assert.ok(theta, 'kernel param theta is a declared input');
  assert.strictEqual(theta.source.kind, 'boundary');
  assert.strictEqual(theta.source.from, 'prior');
  assertSubset(node, ctx, 'bayesupdate');
});
