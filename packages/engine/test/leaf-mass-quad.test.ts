'use strict';
// `normalize(weighted(w, <continuous scalar probability leaf>))` with a
// θ-DEPENDENT `w` resolves Z(θ) = ∫₀¹ w(F_B⁻¹(u); θ) du as an IR EXPRESSION in
// θ — a fixed graded Gauss-Legendre rule — so the IS density route, the MH route
// and the sampling route all score one measure. Spec §06: "logdensityof(
// normalize(M), x) = logdensityof(M, x) − log Z, with Z = totalmass(M) finite
// and nonzero", and §06 `normalize` makes every θ-slice a probability measure.
//
// THE ORACLE IS CLOSED-FORM. e^{θx}·φ(x) = e^{θ²/2}·φ(x−θ), so
// Z(θ) = e^{θ²/2} exactly and `normalize(weighted(fn(exp(θ·_)), Normal(0,1)))`
// is Normal(θ, 1). Every density figure below is that Gaussian's log-pdf; the
// sampling figure is the prior mean, which §06 fixes without any quadrature.
//
// BEFORE THIS ARM the two routes disagreed: the density route baked −log Ẑ from
// the inner measure's tracked `logTotalmass` (an importance-sampling estimate
// over the base's own ensemble, so it moved with the sample count — likOf(θ=1)
// was −1.885 at N = 1, −1.019 at N = 64, −1.654 at N = 4000 against the exact
// −1.0439385332046727) while the sampling route REFUSED the same measure.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { leafMassExpr, leafRecognize, LEAF_TAIL_GATE } = require('../leaf-mass-quad.ts');
const { evaluateExpr } = require('../sampler.ts');
const { gradedUnitCells } = require('../quadrature.ts');

const H = 'flatppl_compat = "0.1"\n';
const TILT = 'm = normalize(weighted(x -> exp(theta * x), Normal(mu = 0.0, sigma = 1.0)))\n';
const OBS = 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

// log N(0.5; θ, 1) — the normalized measure's own log-pdf at the observation.
const exactAt = (th: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (0.5 - th) ** 2;

async function likAt(src: string, N: number, theta: number, seed?: number): Promise<number> {
  const { proc, ctx } = ctxFor(H + src, N);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  if (seed != null) { ctx.rootKey = seed; ctx.rootSeed = seed; }
  let deriv: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') deriv = v;
  }
  const { likOf } = await buildLogPi(ctx, deriv);
  return likOf({ theta });
}

// =====================================================================
// The density routes
// =====================================================================

test('the MH route matches the closed-form Z(θ) = e^{θ²/2} at every θ', async () => {
  const PRIOR = 'theta ~ Uniform(interval(0.5, 2.0))\n';
  for (const th of [0.5, 1.0, 2.0]) {
    const v = await likAt(PRIOR + TILT + OBS, 8, th);
    assert.ok(Math.abs(v - exactAt(th)) < 1e-7,
      `θ = ${th}: likelihoodof = ${v}, closed form ${exactAt(th)}`);
  }
});

test('the scored value does not move with the sample count or the seed', async () => {
  const src = 'theta ~ Uniform(interval(0.5, 2.0))\n' + TILT + OBS;
  const vals: number[] = [];
  for (const seed of [0xBA5E, 4711]) {
    for (const N of [1, 64, 4000]) vals.push(await likAt(src, N, 1.0, seed));
  }
  const spread = Math.max(...vals) - Math.min(...vals);
  assert.equal(spread, 0, `Z(θ) must be deterministic; spread ${spread} over ${vals.length} runs`);
  assert.ok(Math.abs(vals[0] - exactAt(1.0)) < 1e-7, `likelihoodof = ${vals[0]}`);
});

test('the IS posterior mean matches the exact posterior', async () => {
  // Prior θ ~ U(0.5, 2), likelihood p(0.5 | θ) = φ(0.5 − θ). The posterior mean
  // ∫θφ(0.5−θ)dθ / ∫φ(0.5−θ)dθ over [0.5, 2] is 1.1219509778 (scipy `quad`, and
  // it is the observation-side check the MH figures above cannot give: it runs
  // through mat-density's own copy of the resolver).
  const { proc, ctx } = ctxFor(H + 'theta ~ Uniform(interval(0.5, 2.0))\n' + TILT + OBS, 60000);
  assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const p = await ctx.getMeasure('posterior');
  const d: Float64Array = p.samples;
  const lw: Float64Array = p.logWeights;
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  let s = 0, w = 0;
  for (let i = 0; i < d.length; i++) { const e = Math.exp(lw[i] - mx); s += e * d[i]; w += e; }
  const mean = s / w;
  assert.ok(Math.abs(mean - 1.1219509778) < 0.01,
    `IS posterior E[θ] = ${mean}, exact 1.1219509778`);
});

test('a latent θ and a constant θ score the same measure alike', async () => {
  // With θ fixed the measure is θ-INDEPENDENT and takes `weightedLeafQuadLogZ`'s
  // ADAPTIVE quadrature at tolerance 1e-10; with θ latent it takes this arm's
  // fixed graded rule. Two spellings of one measure, so the two numbers must
  // agree — the property a Monte-Carlo estimator here would break by ~0.1 nats.
  const { ctx } = ctxFor(H
    + 'm = normalize(weighted(x -> exp(1.0 * x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'ld = logdensityof(m, 0.5)\n', 1);
  const fixed = (await ctx.getMeasure('ld')).samples[0];
  const latent = await likAt('theta ~ Uniform(interval(0.5, 2.0))\n' + TILT + OBS, 8, 1.0);
  assert.ok(Math.abs(latent - fixed) < 1e-7,
    `latent-θ ${latent} vs constant-θ ${fixed}`);
});

// =====================================================================
// The sampling route
// =====================================================================

test('the sampled θ-marginal is the prior, not the Z-tilted prior', async () => {
  // §06 makes each θ-slice a probability measure, so the θ-marginal of the
  // sampled joint is the prior exactly. This shape used to REFUSE on this route
  // (no per-θ expression existed); before that refusal the pooled divisor gave
  // E[θ] = 1.3557 against the prior's 1.0, matching the Z-tilted
  // ∫θe^{θ²/2}/∫e^{θ²/2} = 1.3510637950 over [0, 2].
  const n = 60000;
  const { ctx } = ctxFor(H + 'theta ~ Uniform(interval(0.0, 2.0))\n' + TILT + 'y ~ m\n', n);
  const y = await ctx.getMeasure('y');
  const th = await ctx.getMeasure('theta');
  const lw: Float64Array = y.logWeights;
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  let et = 0, ey = 0, w = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(lw[i] - mx);
    et += e * th.samples[i]; ey += e * y.samples[i]; w += e;
  }
  et /= w; ey /= w;
  assert.ok(Math.abs(et - 1.0) < 0.03, `E[θ] = ${et}, prior mean 1.0`);
  assert.ok(Math.abs(et - 1.3510637950) > 0.2, `E[θ] = ${et} is the Z-TILTED 1.3510637950`);
  // y | θ ~ Normal(θ, 1), so E[y] = E[θ] = 1.0 — the variate side of the same
  // statement, which a divisor right for θ but wrong per atom would miss.
  assert.ok(Math.abs(ey - 1.0) < 0.06, `E[y] = ${ey}, exact 1.0`);
});

// =====================================================================
// The mass expression itself
// =====================================================================

const leafIR = (op: string, body: any, kwargs?: any) => ({
  kind: 'call', op: 'normalize', args: [{
    kind: 'call', op,
    args: [{ kind: 'call', op: 'functionof', params: ['x'], body }, {
      kind: 'call', op: 'Normal',
      kwargs: kwargs || { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
    }],
  }],
});
const thetaTimesX = () => ({ kind: 'call', op: 'mul', args: [
  { kind: 'ref', ns: 'self', name: 'theta' }, { kind: 'ref', ns: '%local', name: 'x' }] });
const LATENT_CTX = {
  bindings: new Map([['theta', { ir: { kind: 'lit', value: 1 } }]]), fixedValues: new Map(),
};

test('the emitted Ẑ(θ) tracks e^{θ²/2} across θ', () => {
  const expr = leafMassExpr(leafIR('weighted', { kind: 'call', op: 'exp', args: [thetaTimesX()] }),
    LATENT_CTX);
  for (const th of [0.5, 1, 2, 3]) {
    const got = evaluateExpr(expr, { theta: th });
    const want = Math.exp(th * th / 2);
    assert.ok(Math.abs(got / want - 1) < 1e-5, `θ = ${th}: Ẑ = ${got}, Z = ${want}`);
  }
});

test('the log-space spelling integrates e^ℓ, not ℓ', () => {
  // `logweighted(ℓ, B)` has mass ∫ e^ℓ dB (§06 logweighted), so the two
  // spellings of one measure must emit the same Ẑ. (The density WALKER does not
  // yet score a general function-of-variate `logweighted` numerator — it asks
  // for the §12 product_dist add-fold — so this is pinned at the mass builder.)
  const asLog = leafMassExpr(leafIR('logweighted', thetaTimesX()), LATENT_CTX);
  const asExp = leafMassExpr(leafIR('weighted',
    { kind: 'call', op: 'exp', args: [thetaTimesX()] }), LATENT_CTX);
  for (const th of [0.5, 1, 2]) {
    assert.equal(evaluateExpr(asLog, { theta: th }), evaluateExpr(asExp, { theta: th }));
  }
});

test('the accuracy gate hands back +∞ where the rule cannot resolve Z', () => {
  // ∫ e^{θx²} dΦ = (1−2θ)^{-1/2} for θ < 1/2 and does not exist for θ ≥ 1/2, so
  // one expression must produce both answers. Without the gate the fixed rule
  // returned a large finite number for the divergent side.
  const expr = leafMassExpr(leafIR('weighted', { kind: 'call', op: 'exp', args: [
    { kind: 'call', op: 'mul', args: [
      { kind: 'ref', ns: 'self', name: 'theta' },
      { kind: 'call', op: 'mul', args: [
        { kind: 'ref', ns: '%local', name: 'x' }, { kind: 'ref', ns: '%local', name: 'x' }] }] }] }),
    LATENT_CTX);
  for (const th of [0.1, 0.3]) {
    const got = evaluateExpr(expr, { theta: th });
    assert.ok(Math.abs(got / Math.pow(1 - 2 * th, -0.5) - 1) < 1e-3,
      `θ = ${th}: Ẑ = ${got}, Z = ${Math.pow(1 - 2 * th, -0.5)}`);
  }
  for (const th of [0.5, 1.0, 4.0]) {
    assert.equal(evaluateExpr(expr, { theta: th }), Infinity, `θ = ${th} must refuse`);
  }
});

test('a divergent Z reaches the caller as a loud refusal, not a number', async () => {
  const { ctx } = ctxFor(H + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'm = normalize(weighted(x -> exp(theta * x * x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'y ~ m\n', 64);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('y')),
    /total mass of the inner measure is Infinity/);
});

test('a θ-INDEPENDENT weight is left to the adaptive arm', () => {
  const noTheta = { kind: 'call', op: 'exp', args: [{ kind: 'ref', ns: '%local', name: 'x' }] };
  assert.equal(leafRecognize(leafIR('weighted', noTheta), LATENT_CTX), null);
  assert.equal(leafMassExpr(leafIR('weighted', noTheta), LATENT_CTX), null);
});

test('a base whose PARAMETERS move with a latent declines — the open shape', () => {
  // F_B⁻¹ is evaluated at rewrite time, so a latent base parameter has no value
  // here and the arm declines. §07's `builtin_fromuniform` is the spec surface
  // that would express it per θ; recorded OPEN in
  // flatppl-dev/measure-algebra-audit.md rather than closed here.
  const latentMu = { mu: { kind: 'ref', ns: 'self', name: 'theta' },
    sigma: { kind: 'lit', value: 1 } };
  const body = { kind: 'call', op: 'exp', args: [thetaTimesX()] };
  assert.equal(leafMassExpr(leafIR('weighted', body, latentMu), LATENT_CTX), null);
});

test('a discrete base declines (no entry in the inverse-CDF ladder)', () => {
  const node = {
    kind: 'call', op: 'normalize', args: [{
      kind: 'call', op: 'weighted', args: [
        { kind: 'call', op: 'functionof', params: ['x'], body: thetaTimesX() },
        { kind: 'call', op: 'Poisson', kwargs: { rate: { kind: 'lit', value: 3 } } }],
    }],
  };
  assert.equal(leafMassExpr(node, LATENT_CTX), null);
});

test('a weight body too large to inline is refused, not pooled', () => {
  // The fallback at every call site is the mass POOLED over the atom ensemble,
  // which is E[Z] rather than Z(θ) — the number this arm exists to remove — so
  // an over-budget body must throw rather than return null.
  let body: any = thetaTimesX();
  for (let i = 0; i < 120; i++) body = { kind: 'call', op: 'add', args: [body, thetaTimesX()] };
  const big = { kind: 'call', op: 'exp', args: [body] };
  assert.throws(() => leafMassExpr(leafIR('weighted', big), LATENT_CTX),
    /over the budget of/);
});

test('the graded rule is a partition of (0,1) with unit total weight', () => {
  const cells = gradedUnitCells(40);
  assert.equal(cells.length, 80);
  let total = 0;
  let prev = 0;
  for (const c of cells) {
    for (let i = 0; i < c.us.length; i++) {
      total += c.ws[i];
      assert.ok(c.us[i] > prev - 1e-300 && c.us[i] < 1, `node ${c.us[i]} outside (0,1)`);
    }
    prev = c.us[c.us.length - 1];
  }
  assert.ok(Math.abs(total - 1) < 1e-15, `Σ weights = ${total}`);
  assert.ok(cells[cells.length - 1].us[0] > 1 - Math.pow(2, -39),
    'the outermost cell must sit inside 2^-39 of the endpoint');
  assert.equal(LEAF_TAIL_GATE, 1e-2);
});
