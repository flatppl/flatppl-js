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
const { lowerMeasure, setClmEnabled } = require('../clm.ts');
const { buildCtx } = require('./_agreement-harness.ts');

// Mean / std of a measure's scalar atoms (uniform weights — these fixtures
// have none) for the sample-side matClm equivalence check.
function meanStd(m: any): { mean: number; std: number; n: number } {
  const s = m.samples || (m.value && m.value.data);
  let mean = 0; for (let i = 0; i < s.length; i++) mean += s[i]; mean /= s.length;
  let v = 0; for (let i = 0; i < s.length; i++) v += (s[i] - mean) ** 2; v /= s.length;
  return { mean, std: Math.sqrt(v), n: s.length };
}

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

test('[clm Phase 4] matClm samples a kchain through the canonical body ≡ the direct path', async () => {
  // Sample-side CLM consumer: lowerMeasure(ch) → clm; materialiseMeasureIR(clm)
  // routes through matClm (binds the prior M as a boundary measure, walks the
  // body). Must match the direct matJointchain marginal sample distribution —
  // kchain(Normal(0,10), Normal(mu=a,sigma=1)) marginalises to ~Normal(0,√101).
  // (Statistical, not bit-identical: independent seeding of the two paths.)
  const src = `
a ~ Normal(0.0, 0.01)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
M = Normal(0.0, 10.0)
ch = kchain(M, K)`;
  const direct = meanStd(await buildCtx(src, 40000, 1234).ctx.getMeasure('ch'));
  const { ctx } = buildCtx(src, 40000, 1234);
  const node = lowerMeasure('ch', ctx);
  const viaClm = meanStd(await eng.materialiser.materialiseMeasureIR(node, ctx));
  assert.ok(Math.abs(direct.mean - viaClm.mean) < 0.4,
    `matClm mean ${viaClm.mean.toFixed(3)} vs direct ${direct.mean.toFixed(3)}`);
  assert.ok(Math.abs(direct.std - viaClm.std) / direct.std < 0.1,
    `matClm std ${viaClm.std.toFixed(3)} vs direct ${direct.std.toFixed(3)} (>10% apart)`);
  // And it really is the marginal spread (√101 ≈ 10.05), not the kernel's σ=1.
  assert.ok(viaClm.std > 5, `matClm should sample the marginal (std≈10), got ${viaClm.std.toFixed(2)}`);
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

// ════════════════════════════════════════════════════════════════════════
// Phase 4 LIVE REROUTE — §15 dual-mode validation (engine-concepts §15/§18).
//
// The clmRerouteStage routes a `jointchain`-kind binding through lowerMeasure
// + matClm when `isClmEnabled()`. The legacy matJointchain path stays the
// equivalence ORACLE: materialising the SAME chain with the flag OFF and ON
// must agree (statistically — the two paths seed independently, so moments
// within MC tolerance, not bit-identical). This is the gate the reroute must
// hold before matJointchain.bindLeaf is retired (scaffolding removed last).
// ════════════════════════════════════════════════════════════════════════

function momentsOf(name: string, src: string, N: number, seed: number) {
  const { ctx } = buildCtx(src, N, seed);
  return Promise.resolve(ctx.getMeasure(name));
}

// Materialise `name` with the live reroute forced OFF then ON; return both.
async function bothModes(name: string, src: string, N: number, seed: number) {
  setClmEnabled(false);
  const off = await momentsOf(name, src, N, seed);
  setClmEnabled(true);
  try {
    const on = await momentsOf(name, src, N, seed);
    return { off, on };
  } finally {
    setClmEnabled(false);          // never leak the flag into other tests
  }
}

const CHAIN_FIXTURES: Array<[string, string, string]> = [
  ['2-step kchain marginal', 'ch', `
M = Normal(mu = 0.0, sigma = 10.0)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
ch = kchain(M, K)`],
  ['N-ary (3-step) kchain marginal', 'd', `
M  = Normal(mu = 0.0, sigma = 1.0)
K1 = fn(Normal(mu = _, sigma = 1.0))
K2 = fn(Normal(mu = get(_, 2), sigma = 1.0))
d  = kchain(M, K1, K2)`],
  ['record-base kchain (cat-arity A)', 'ch', `
joint_indep = joint(t1 = Normal(mu = 0.0, sigma = 1.0), t2 = Exponential(rate = 1.0))
K = functionof(Normal(mu = t1, sigma = t2), t1 = t1, t2 = t2)
ch = kchain(joint_indep, K)`],
];

for (const [id, name, src] of CHAIN_FIXTURES) {
  test(`[clm Phase 4 dual-mode] ${id} — reroute ≡ legacy (moments)`, async () => {
    const { off, on } = await bothModes(name, src, 40000, 99);
    assert.ok(off.samples && on.samples, `${id}: both modes scalar marginal`);
    const a = meanStd(off), b = meanStd(on);
    assert.ok(Math.abs(a.mean - b.mean) < 0.15,
      `${id}: mean off=${a.mean.toFixed(3)} on=${b.mean.toFixed(3)}`);
    assert.ok(Math.abs(a.std - b.std) / Math.max(a.std, 1e-9) < 0.08,
      `${id}: std off=${a.std.toFixed(3)} on=${b.std.toFixed(3)} (>8% apart)`);
  });
}

test('[clm Phase 4 dual-mode] jointchain RETAIN — reroute ≡ legacy (per-variate + cov)', async () => {
  const src = `
M = Normal(mu = 0.0, sigma = 1.0)
K = functionof(Normal(mu = x, sigma = 1.0), x = x)
jc = jointchain(M, K)`;
  const { off, on } = await bothModes('jc', src, 40000, 99);
  assert.ok(off.elems && on.elems && off.elems.length === 2 && on.elems.length === 2,
    'both modes retain a 2-tuple');
  const covOf = (m: any) => {
    const a = m.elems[0].samples, b = m.elems[1].samples;
    let ma = 0, mb = 0; const n = a.length;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    let c = 0; for (let i = 0; i < n; i++) c += (a[i] - ma) * (b[i] - mb);
    return c / n;
  };
  // Cov(a,b) = Var(a) = 1 for b = a + N(0,1). Both paths must recover it.
  assert.ok(Math.abs(covOf(off) - covOf(on)) < 0.12,
    `retain cov off=${covOf(off).toFixed(3)} on=${covOf(on).toFixed(3)}`);
});
