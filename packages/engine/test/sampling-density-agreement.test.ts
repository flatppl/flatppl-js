'use strict';

// ════════════════════════════════════════════════════════════════════════
// STANDING INVARIANT for the measure-lowering unification
// (flatppl-dev/measure-lowering-unification-plan.md, Phase 0).
//
// The engine walks the SAME measure IR along two paths — SAMPLE and DENSITY —
// that drift. This harness pins the property the unification must make
// structural: for a CLOSED scalar measure M, `logdensityof(M, x)` agrees with
// the empirical log-density of the SAMPLE histogram of the same M (compared as
// a RATIO between probe points spread across the support, cancelling the
// normalisation constant + bin width). Harness in test/_agreement-harness.ts;
// it is self-checked on a plain Normal + a truncate below.
//
// Each fixture is tagged by audit ID + DISPOSITION:
//   GREEN       — already correct (sweep fix or always-correct); guards against
//                 regression. assert(ok).
//   WILL-FLIP   — still divergent; the CLM unification (or an earlier targeted
//                 fix) will fix it. assert(!ok) TIED TO THE FAILURE MODE (so it
//                 is red for the RIGHT reason); flips loudly to a test FAILURE
//                 when fixed → re-tag to GREEN.
//   OUT-OF-SCOPE — a divergence CLM does NOT address (M2 selector,
//                 MvNormal-kchain); test.skip with a note, so the eventual
//                 "all green" gate is not falsely blocked.
//
// The scalar-ratio harness is BLIND to pure normalisation-constant bugs (M3
// totalmass) and cannot see correlation (H7) or density-curve shape (H10);
// those use fieldCorrelation here / dedicated checks at their phase.
// ════════════════════════════════════════════════════════════════════════

const test = require('node:test');
const assert = require('node:assert');
const { agreement, fieldCorrelation } = require('./_agreement-harness.ts');

// ---- harness self-checks (MUST pass — they validate the harness itself) ----

test('agreement self-check: closed Normal(0,1) sample ≡ density', async () => {
  const r = await agreement('m = Normal(0.0, 1.0)\n', 'm', { N: 60000, tol: 0.3 });
  assert.ok(r.ok, `self-check failed (maxErr=${r.maxErr}, ${r.reason || ''}): ` + JSON.stringify(r.probes));
});

test('agreement self-check: closed truncate(Normal) sample ≡ density (C1)', async () => {
  const r = await agreement('m = truncate(Normal(0.0, 1.0), interval(-2.0, 2.0))\n', 'm', { N: 60000, tol: 0.3 });
  assert.ok(r.ok, `truncate self-check failed (maxErr=${r.maxErr}, ${r.reason || ''}): ` + JSON.stringify(r.probes));
});

// ---- GREEN regression guards (sweep fixes / always-correct) --------------

const GREEN: Array<[string, string, string]> = [
  ['kchain hole-param over a wide prior (M1-family)', 'ch', `
a ~ Normal(0.0, 0.01)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
M = Normal(0.0, 10.0)
ch = kchain(M, K)`],
  ['M1: kchain over a RELABELLED prior (density ≡ histogram)', 'ch', `
theta ~ Normal(0.0, 1.0)
shifted = theta + 20.0
M = lawof(shifted)
K = functionof(Normal(mu = shifted, sigma = 1.0), shifted = shifted)
ch = kchain(M, K)`],
  ['kchain marginalises the prior (the correct contrast to H8)', 'ch', `
theta ~ Normal(0.0, 1.0)
K = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
prior = lawof(theta)
ch = kchain(prior, K)`],
  ['hole-kernel over a RECORD base prior (cat-arity case A)', 'ch', `
mu_p = 0.0
joint_indep = joint(t1 = Normal(mu = mu_p, sigma = 1.0), t2 = Exponential(rate = 1.0))
K = functionof(Normal(mu = t1, sigma = t2), t1 = t1, t2 = t2)
ch = kchain(joint_indep, K)`],
];

for (const [id, mname, src] of GREEN) {
  test(`[GREEN] ${id}`, async () => {
    const r = await agreement(src, mname, { N: 60000, tol: 0.4 });
    assert.ok(r.ok, `expected GREEN but got ${r.crashed ? 'CRASH:' + r.reason : 'RED maxErr=' + (r.maxErr || 0).toFixed(2)} — ` + JSON.stringify(r.probes || r.reason));
  });
}

// ---- WILL-FLIP (red today, for the right reason; flip loudly when fixed) --

test('[GREEN H8] lawof(draw with a stochastic ancestor) marginalises — density ≡ histogram', async () => {
  // pp = lawof(obs), obs~Normal(theta,1), theta~Normal(0,1). Sample ⇒ marginal
  // Normal(0,√2). The density used to score the per-atom conditional Normal(·,1)
  // — a VARIANCE mismatch glaring in the tails (maxErr≈4.15). FIXED (CLM Phase 3):
  // lowerMeasure recognises the marginalised stochastic ancestor (theta is a
  // `shared` body ref to a stochastic binding, not a retained variate) and sets
  // reduce={marginal}. That reduction was an MC logsumexp − logN over theta ~
  // prior; it is now the EXACT linear-Gaussian marginal Normal(0,√2)
  // (linear-gaussian.ts, spec §06 "Density of composed measures" — closed form
  // or refuse, never an estimate), pinned against Distributions.jl in
  // joint-shared-ancestor-density.test.ts. Now a regression guard.
  const r = await agreement(`
theta ~ Normal(0.0, 1.0)
obs ~ Normal(mu = theta, sigma = 1.0)
pp = lawof(obs)`, 'pp', { N: 60000, tol: 0.35 });
  assert.ok(r.ok, `H8 regression — expected marginal agreement (Normal(0,√2)), got ` +
    `${r.crashed ? 'CRASH:' + r.reason : 'maxErr=' + (r.maxErr || 0).toFixed(2)}: ` +
    JSON.stringify(r.probes || r.reason));
});

test('[GREEN F1] a hierarchical CHAIN marginal — density ≡ histogram', async () => {
  // z ~ N(0,1); a ~ N(z,1); b ~ N(a,1) ⇒ lawof(b) is Normal(0, √3). The
  // recogniser reaches `z` only through `a`'s law (only `a` is a body self-ref),
  // so this is the transitive ancestor closure checked against SAMPLING — a
  // path that never touches linear-gaussian.ts. Dropping the closure refuses;
  // composing the maps wrongly gives the wrong variance, glaring in the tails.
  const r = await agreement(`
z ~ Normal(0.0, 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = a, sigma = 1.0)
m = lawof(b)`, 'm', { N: 60000, tol: 0.35 });
  assert.ok(r.ok, `chain marginal — expected Normal(0,√3) agreement, got ` +
    `${r.crashed ? 'CRASH:' + r.reason : 'maxErr=' + (r.maxErr || 0).toFixed(2)}: ` +
    JSON.stringify(r.probes || r.reason));
});

test('[GREEN F1] an ENUMERATED discrete latent — mixture density ≡ histogram', async () => {
  // s ~ Categorical([0.2,0.5,0.3]); x ~ N(s, 0.3) is trimodal at 1, 2, 3 with
  // well-separated modes. Sampling draws s from the registry's own sampler
  // (1-based, spec §08) while the density enumerates the atoms, so this pins the
  // atom CONVENTION and the mass WEIGHTS against an independent path: a 0-based
  // enumeration would put all three modes one unit low.
  const r = await agreement(`
s ~ Categorical(p = [0.2, 0.5, 0.3])
x ~ Normal(mu = s, sigma = 0.3)
m = lawof(x)`, 'm', { N: 60000, tol: 0.35 });
  assert.ok(r.ok, `enumerated mixture — expected trimodal agreement, got ` +
    `${r.crashed ? 'CRASH:' + r.reason : 'maxErr=' + (r.maxErr || 0).toFixed(2)}: ` +
    JSON.stringify(r.probes || r.reason));
});

test('[GREEN gen] standalone logdensityof(lawof(generative composite)) marginalises — density ≡ histogram', async () => {
  // Single-event transport: z = f(x, uniform) at fixed pars. The MC-marginal
  // density used to be a GAP as a bare logdensityof(lawof(z), x) (crash).
  // FIXED (CLM Phase 4): buildMcMarginalForm gained a non-broadcast branch
  // (_buildFromValueExpr) — it inlines z's value closure, finds the draws, and
  // picks the bijective retained innovation (the Uniform, invertExpr succeeds)
  // vs the marginalised latent (the Normal x), emitting the bare `mcmarginal`
  // node density.walkMcMarginal scores (batched in-worker MC over x — the SAME
  // estimator the broadcast/iid generative composites use; that batched path is
  // untouched). Converges to the sampled marginal as M grows (maxErr 0.51 → 0.12
  // → 0.11 at M=200/1000/4000), so this is a true agreement at M=1000, not a
  // loosened tolerance. Now a regression guard.
  const r = await agreement(`
sigma = 0.2
mu = 1.1
a = 0.1
b = 0.3
x ~ Normal(mu, sigma)
delta = (2.0 * draw(Uniform(interval(0, 1))) + 1.0) * a
y = (x + delta)^3 * exp(x - b)
z = y / 2.0
m = lawof(z)`, 'm', { N: 40000, tol: 0.4, mc: 1000 });
  assert.ok(r.ok, `generative-standalone regression — expected MC-marginal agreement, got ` +
    `${r.crashed ? 'CRASH:' + r.reason : 'maxErr=' + (r.maxErr || 0).toFixed(2)}: ` +
    JSON.stringify(r.probes || r.reason));
});

test('[GREEN H7] joint(m, m) is the INDEPENDENT product (Corr≈0)', async () => {
  // joint is the INDEPENDENT product (spec §06) ⇒ corr≈0; the bug returned the
  // memoised atom batch ⇒ corr=1. FIXED (CLM Phase 4): matRecord/matTuple
  // re-seed a DUPLICATE direct factor in a child ctx (the first occurrence
  // stays on the shared cache so derived factors joint(a=x,b=g(x)) keep their
  // shared-ancestor alignment). Now a regression guard.
  const corr = await fieldCorrelation(`
m = Normal(0.0, 1.0)
j = joint(a = m, b = m)`, 'j', 'a', 'b', { N: 30000 });
  assert.ok(Math.abs(corr) < 0.1,
    `H7 regression — joint(m,m) must be independent (|corr|<0.1), got corr=${corr.toFixed(3)}`);
});

test('[GREEN H7c] joint(m, m) over a constructor with a LATENT shares the latent '
  + '(Corr = s0²/(s0²+σ²))', async () => {
  // The H7 sibling above is ancestor-free, where §04's Identity law gives the
  // independent product. Add a latent and §06 "Joint composition" splits the two
  // roles: "A component contributes a fresh coordinate; a stochastic node shared
  // between component traces (through a reified component … or a stochastic
  // constructor parameter) remains a single node of the composed trace". So the
  // coordinates are fresh but `z` is ONE draw, and
  //   Corr(a, b) = Var(z) / (Var(z) + σ²) = 4 / 4.36 = 0.9174311926605504.
  // This is the sampling side of the density path's compound law
  // (joint-shared-ancestor-density.test.ts pins that at -10.903201177191129), so
  // the two must not disagree. The re-seeded duplicate factor used to redraw its
  // whole sub-DAG, giving two independent copies at Corr ≈ 0 — `iid`'s semantics,
  // not `joint`'s — while the density scored the correlated law.
  const corr = await fieldCorrelation(`
z ~ Normal(mu = 0.5, sigma = 2.0)
m = Normal(mu = z, sigma = 0.6)
j = joint(a = m, b = m)`, 'j', 'a', 'b', { N: 30000 });
  const want = 4 / 4.36;
  assert.ok(Math.abs(corr - want) < 0.02,
    `shared latent must survive the fresh coordinates: want Corr≈${want.toFixed(4)}, `
    + `got ${corr.toFixed(4)} (≈0 means the duplicate redrew z; ≈1 means the `
    + 'coordinates collapsed onto one draw)');
  assert.ok(corr < 0.999,
    `the coordinates must stay FRESH, got Corr=${corr.toFixed(6)} — a shared `
    + 'coordinate is the singular diagonal, which has no density');
});

test('[WILL-FLIP H7d] a NESTED constructor joint reuses the inner draw for the '
  + 'outer component', async () => {
  // `joint(u = joint(a = q, b = q), c = q)` over `q = Normal(mu = z, sigma = 0.6)`
  // should be THREE fresh coordinates over one shared `z` (§06: "A component
  // contributes a fresh coordinate"), so all three pairwise correlations are
  // s0²/(s0²+σ²) = 0.9174. The inner pair is right. But `c` is BIT-IDENTICAL to
  // `u.a`: the outer joint's `q` is not a duplicate BY NAME of `u`, so it takes
  // the first-occurrence branch in `_materialiseFactorsIndependent`
  // (materialiser.ts:781) and gets the cached batch `u.a` already drew. §06
  // "Singular joints" is what that pair actually is — the same draw twice.
  //
  // PRE-EXISTING (identical at clean 0ab097d) and SAMPLING-ONLY: the density path
  // refuses this shape, pinned in joint-shared-ancestor-density.test.ts, so no
  // wrong number is scored. It nonetheless breaks the [GREEN H7] invariant above
  // under nesting, so it is tagged WILL-FLIP and asserted TIED TO THE FAILURE
  // MODE rather than to the wrong value: this goes red the moment the defect is
  // fixed, which is the signal to re-tag it GREEN and assert 0.9174.
  const src = `
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
u = joint(a = q, b = q)
j = joint(uu = u, c = q)`;
  const inner = await fieldCorrelation(src, 'u', 'a', 'b', { N: 20000 });
  const want = 4 / 4.36;
  assert.ok(Math.abs(inner - want) < 0.03,
    `the INNER pair must already share the latent: want ≈${want.toFixed(4)}, got ${inner.toFixed(4)}`);

  const { ctx } = require('./_agreement-harness.ts').buildCtx(src, 20000, 99);
  const m = await ctx.getMeasure('j');
  const ua = m.fields.uu.fields.a.samples;
  const c = m.fields.c.samples;
  let identical = 0;
  for (let i = 0; i < c.length; i++) if (c[i] === ua[i]) identical++;
  assert.equal(identical, c.length,
    'THE DEFECT IS FIXED — `c` is no longer the same batch as `u.a`. Re-tag this '
    + 'test GREEN and assert all three pairwise correlations ≈ '
    + `${want.toFixed(4)} instead (got ${identical}/${c.length} identical)`);
});

test('[GREEN H7b/B] joint(posterior, posterior) — reused WEIGHTED factor refused loudly', async () => {
  // The critique's high-severity case (B): re-seeding a reused posterior gives
  // corr≈0 but the sample-side outer weight (w1+w2) disagrees with the density
  // (outer-only) — a silent IS-weight asymmetry a corr test can't see. CLM
  // Phase 4 REFUSES a reused weighted/posterior factor with a loud error rather
  // than be silently wrong (combining the sub-field weight streams is the
  // deferred enhancement). This guards that the refusal stays loud.
  let crashed = false;
  try {
    await fieldCorrelation(`
theta ~ Normal(0.0, 2.0)
obs ~ iid(Normal(mu = theta, sigma = 1.0), 5)
fwd = kernelof(record(obs = obs), theta = theta)
data = [2.0, 2.1, 1.9, 2.2, 1.8]
L = likelihoodof(fwd, record(obs = data))
prior = lawof(record(theta = theta))
post = bayesupdate(L, prior)
j = joint(a = post, b = post)`, 'j', 'a', 'b', { N: 6000 });
  } catch (_) { crashed = true; }
  assert.ok(crashed,
    'joint(posterior, posterior) must be refused loudly (reused WEIGHTED factor — '
    + 'IS-weight asymmetry); if it now succeeds, weight-stream combination landed '
    + '→ replace with an IS-weight agreement guard');
});

// ---- OUT-OF-SCOPE for CLM (documented; not gated by the "all green" goal) --

test.skip('[OUT-OF-SCOPE M2] comparison-selector mixture density pools the selector (not fixed by CLM)', () => {});
test.skip('[OUT-OF-SCOPE] MvNormal / vector-variate kchain kernel (sampling unimplemented; not fixed by CLM)', () => {});
