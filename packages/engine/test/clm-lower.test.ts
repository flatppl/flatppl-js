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
const { lowerMeasure, feedInputs } = require('../clm.ts');
const { buildCtx } = require('./_agreement-harness.ts');
const shared = require('../materialiser-shared.ts');

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
// Phase 4 LIVE REROUTE — chain sampling through the CLM path (matJointchain
// retired). jointchain/kchain bindings sample SOLELY via clmRerouteStage →
// lowerMeasure → matClm. These assert the chain marginals' moments against
// ANALYTICAL truth (a stronger check than the former reroute-≡-legacy
// dual-mode comparison, now that there is no legacy path to compare to).
// ════════════════════════════════════════════════════════════════════════

const CHAIN_FIXTURES: Array<[string, string, string, number]> = [
  // 2-step kchain marginal: M=N(0,10), K=N(a,1) ⇒ N(0, √101), std≈10.05.
  ['2-step kchain marginal', 'ch', `
M = Normal(mu = 0.0, sigma = 10.0)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
ch = kchain(M, K)`, Math.sqrt(101)],
  // N-ary 3-step: a0~N(0,1); a1~N(a0,1); a2~N(a1,1) ⇒ N(0, √3), std≈1.73.
  ['N-ary (3-step) kchain marginal', 'd', `
M  = Normal(mu = 0.0, sigma = 1.0)
K1 = fn(Normal(mu = _, sigma = 1.0))
K2 = fn(Normal(mu = get(_, 2), sigma = 1.0))
d  = kchain(M, K1, K2)`, Math.sqrt(3)],
  // Record-base (cat-arity A): obs~N(t1,t2), t1~N(0,1), t2~Exp(1).
  // Var(obs) = Var(t1) + E[t2²] = 1 + 2 = 3 ⇒ std≈√3.
  ['record-base kchain (cat-arity A)', 'ch', `
joint_indep = joint(t1 = Normal(mu = 0.0, sigma = 1.0), t2 = Exponential(rate = 1.0))
K = functionof(Normal(mu = t1, sigma = t2), t1 = t1, t2 = t2)
ch = kchain(joint_indep, K)`, Math.sqrt(3)],
];

for (const [id, name, src, wantStd] of CHAIN_FIXTURES) {
  test(`[clm Phase 4] ${id} — CLM marginal moments ≈ analytic`, async () => {
    const { ctx } = buildCtx(src, 40000, 99);
    const m = await ctx.getMeasure(name);
    assert.ok(m.samples, `${id}: scalar marginal`);
    const s = meanStd(m);
    assert.ok(Math.abs(s.mean) < 0.2, `${id}: mean ${s.mean.toFixed(3)} ≉ 0`);
    assert.ok(Math.abs(s.std - wantStd) / wantStd < 0.08,
      `${id}: std ${s.std.toFixed(3)} vs analytic ${wantStd.toFixed(3)} (>8% apart)`);
  });
}

test('[clm Phase 4] jointchain RETAIN — CLM keeps (a,b) with Cov(a,b)=Var(a)=1', async () => {
  const src = `
M = Normal(mu = 0.0, sigma = 1.0)
K = functionof(Normal(mu = x, sigma = 1.0), x = x)
jc = jointchain(M, K)`;
  const { ctx } = buildCtx(src, 40000, 99);
  const m = await ctx.getMeasure('jc');
  assert.ok(m.elems && m.elems.length === 2, 'retains a 2-tuple (a, b)');
  const a = m.elems[0].samples, b = m.elems[1].samples;
  let ma = 0, mb = 0; const n = a.length;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let cov = 0; for (let i = 0; i < n; i++) cov += (a[i] - ma) * (b[i] - mb);
  cov /= n;
  // b = a + N(0,1) ⇒ Cov(a,b) = Var(a) = 1.
  assert.ok(Math.abs(cov - 1.0) < 0.1, `retain cov ${cov.toFixed(3)} ≉ 1`);
});

// ════════════════════════════════════════════════════════════════════════
// Phase 5 — explicit-boundary lowering (the viewer kernel/profile plot path).
// lowerMeasure(<kernel body IR>, ctx, {boundaries, freeInputs}) feeds the
// caller-supplied input VALUES through the ONE feedInputs contract, replacing
// the viewer's inlineForProfile + substituteLocals bake. matClm routes the
// `node.fed` lowering through its feed path even with reduce=null.
// ════════════════════════════════════════════════════════════════════════

test('[clm Phase 5] explicit scalar boundary — kernel fed at a=5, samples ~ Normal(5,1)', async () => {
  const src = `
a = elementof(reals)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)`;
  const { ctx } = buildCtx(src, 40000, 7);
  const body = ctx.bindings.get('K').ir.body;            // the reified kernel body (Normal(mu=%local a, …))
  const node = lowerMeasure(body, ctx, { boundaries: { a: 5.0 } });
  assert.ok(node && node.op === 'clm', 'lowers to a clm node');
  assert.ok(node.fed, 'explicit-boundary lowering sets node.fed → matClm feed path');
  const aIn = node.inputs.find((i: any) => i.name === 'a');
  assert.ok(aIn, 'a is a declared input');
  assert.strictEqual(aIn.source.kind, 'explicit');
  assert.strictEqual(aIn.source.value, 5.0);
  const m = await eng.materialiser.materialiseMeasureIR(node, ctx);
  const s = meanStd(m);
  assert.ok(Math.abs(s.mean - 5.0) < 0.1, `mean ${s.mean.toFixed(3)} ≉ 5 (boundary not fed?)`);
  assert.ok(Math.abs(s.std - 1.0) < 0.1, `std ${s.std.toFixed(3)} ≉ 1`);
});

test('[clm Phase 5] explicit RECORD boundary — pars fed per-atom, samples ~ Normal(5,2)', async () => {
  // The transport `pars` shape: a record-typed kernel input field-accessed in
  // the body. feedInputs binds it as the per-atom-record array get_field
  // consumes (matching measureToPerAtomRecords) — the `_pars_`-class case.
  const src = `
pars = elementof(cartprod(a = reals, b = posreals))
K = functionof(Normal(mu = pars.a, sigma = pars.b), pars = pars)`;
  const { ctx } = buildCtx(src, 40000, 7);
  const body = ctx.bindings.get('K').ir.body;
  const node = lowerMeasure(body, ctx, { boundaries: { pars: { a: 5.0, b: 2.0 } } });
  const parsIn = node.inputs.find((i: any) => i.name === 'pars');
  assert.ok(parsIn && parsIn.source.kind === 'explicit', 'pars is an explicit input');
  assert.strictEqual(parsIn.shape.kind, 'record', 'record-shaped boundary');
  const m = await eng.materialiser.materialiseMeasureIR(node, ctx);
  const s = meanStd(m);
  assert.ok(Math.abs(s.mean - 5.0) < 0.15, `mean ${s.mean.toFixed(3)} ≉ 5 (record field not fed?)`);
  assert.ok(Math.abs(s.std - 2.0) < 0.15, `std ${s.std.toFixed(3)} ≉ 2`);
});

test('[clm Phase 5] free input (the profile sweep axis) is declared but UNFED', async () => {
  const src = `
a = elementof(reals)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)`;
  const { ctx } = buildCtx(src, 100, 7);
  const body = ctx.bindings.get('K').ir.body;
  const node = lowerMeasure(body, ctx, { freeInputs: ['a'] });
  const aIn = node.inputs.find((i: any) => i.name === 'a');
  assert.ok(aIn && aIn.source.kind === 'free', 'a is a free input');
  const fed = await feedInputs(node, ctx);
  assert.ok(!Object.prototype.hasOwnProperty.call(fed.refArrays, 'a'),
    'a free input is not bound into refArrays (the worker varies it per sweep point)');
});

// ════════════════════════════════════════════════════════════════════════
// Smell A, Stage 0 — prepareDensityRefs honours ctx._extraRefArrays.
//
// The CLM boundary feed threads fed columns through `ctx._extraRefArrays`,
// the overlay `collectRefArrays` already honours. `prepareDensityRefs` (the
// density / evaluate ref-prep) must honour it too, so a matEvaluate-class
// handler reached from a CLM-fed child ctx (the Smell A materialiser merge)
// conditions on the FED prior rather than re-materialising a like-named
// binding via getMeasure (the boundary-conflation bug, audit §3). No
// production caller hits prepareDensityRefs inside a CLM-fed ctx today, so
// this is a no-op for the current suite; the test pins the contract directly.
// ════════════════════════════════════════════════════════════════════════
test('Stage 0: prepareDensityRefs honours ctx._extraRefArrays (CLM feed overlay)', async () => {
  const N = 4;
  const col = { shape: [N], data: Float64Array.from([1, 2, 3, 4]) };  // a fed per-atom column
  let getMeasureCalled = false;
  const ir = { kind: 'call', op: 'add', args: [
    { kind: 'ref', ns: 'self', name: 'theta' },
    { kind: 'lit', value: 1, numType: 'real' },
  ] };
  const ctx = {
    bindings: new Map(),
    fixedValues: new Map(),
    sampleCount: N,
    getMeasure: (n: any) => {
      getMeasureCalled = true;
      return Promise.reject(new Error('getMeasure must not run for fed ref ' + n));
    },
    _extraRefArrays: { theta: col },
  };
  const prep = await shared.prepareDensityRefs(ir, ctx, 'stage0-test');
  assert.strictEqual(prep.refArrays.theta, col,
    'fed column overlaid into refArrays by reference (not cloned)');
  assert.ok(!prep.perAtomNames.includes('theta'),
    'fed ref must not be queued for getMeasure');
  assert.strictEqual(getMeasureCalled, false,
    'getMeasure must not run for a CLM-fed boundary ref');
});
