'use strict';

// =====================================================================
// leaf-spelling-equivalence.test.ts
// =====================================================================
//
// §04's calling convention makes `Normal(0.0, 1.0)` and
// `Normal(mu = 0.0, sigma = 1.0)` the SAME call, and §08 fixes the binding
// order: "The names and order of the distribution parameters specified below
// define the names and positional order of the kernel arguments." A mass
// recogniser that reads `kwargs` alone therefore splits one measure in two.
//
// `mat-density.asScalarFactor` did exactly that, and four mass routes hang off
// it. Measured before the fix, against the closed forms asserted below:
//
//   route                                     positional        keyword
//   normalize(truncate(Normal(p,1), S))       REFUSED           exact
//   §12 shared-variate product normalizer     REFUSED           exact
//   normalize(weighted(w, Normal(0,1)))       −1.4635483100     exact
//   the same with a θ-dependent weight        −1.6923606584     exact
//
// The two refusals were loud. The two numbers were not: half a nat out with no
// diagnostic. Each route below is asserted TWICE — once for the two spellings
// being bit-identical, once for the value against a closed form computed here
// (mpmath, 50 digits), because equality between two spellings of a wrong
// number proves nothing.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { asScalarFactor } = require('../mat-density.ts');

const H = 'flatppl_compat = "0.1"\n';

// The two spellings of Normal(0, 1). §08's row is `mu`, `sigma`, in that order.
const N01_POS = 'Normal(0.0, 1.0)';
const N01_KW = 'Normal(mu = 0.0, sigma = 1.0)';

async function scoreOf(src: string, name: string, N = 1): Promise<number> {
  const { proc, ctx } = ctxFor(src, N);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const m = await ctx.getMeasure(name);
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error(`scoreOf: ${name} produced no data`);
  return s[0];
}

// The MH route's likelihood at one θ — how the θ-dependent leaf-mass arm is
// reached (`leaf-mass-quad` emits an expression in θ, so it needs a latent).
async function likAt(src: string, theta: number, N = 8): Promise<number> {
  const { proc, ctx } = ctxFor(H + src, N);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  let deriv: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') deriv = v;
  }
  const { likOf } = await buildLogPi(ctx, deriv);
  return likOf({ theta });
}

// =====================================================================
// The binding rule itself
// =====================================================================

test('asScalarFactor binds both spellings to the declared parameter names', () => {
  const lit = (v: number) => ({ kind: 'lit', value: v });
  const expected = { kernel: 'Normal', input: { mu: 0, sigma: 1 } };
  assert.deepEqual(
    asScalarFactor({ kind: 'call', op: 'Normal', kwargs: { mu: lit(0), sigma: lit(1) } }, null),
    expected, 'keyword');
  assert.deepEqual(
    asScalarFactor({ kind: 'call', op: 'Normal', args: [lit(0), lit(1)] }, null),
    expected, 'positional — §08 order is mu, sigma');
  // A MIXED call binds the keyword by name and the rest by declared index, the
  // precedence `sampler-registry.resolveParamsN` uses.
  assert.deepEqual(
    asScalarFactor({ kind: 'call', op: 'Normal', args: [lit(9)], kwargs: { sigma: lit(1) } }, null),
    { kernel: 'Normal', input: { mu: 9, sigma: 1 } }, 'mixed');
  // The accepted set is the REGISTRY's, which is also the key set forward-cdf,
  // inverse-cdf and builtin_logdensityof read.
  assert.equal(
    asScalarFactor({ kind: 'call', op: 'Lebesgue', args: [{ kind: 'const', name: 'reals' }] }, null),
    null, 'a reference measure is not a distribution leaf');
  assert.equal(
    asScalarFactor({ kind: 'call', op: 'Normal', args: [lit(0)] }, null),
    null, 'a leaf missing a parameter has no binding for it');
  // §04 makes a surplus keyword a static error, raised by whichever call site
  // recognises the distribution; this recogniser's contract is a binding or
  // null, so it declines rather than throwing from here.
  assert.equal(
    asScalarFactor({ kind: 'call', op: 'Normal', kwargs: { mu: lit(0), sigma: lit(1), zz: lit(9) } }, null),
    null, 'a surplus keyword declines');
  // `Uniform` carries its bounds inside `support`, which is a SET and not a
  // scalar, so it stays outside this recogniser's set in either spelling.
  assert.equal(
    asScalarFactor({
      kind: 'call', op: 'Uniform',
      args: [{ kind: 'call', op: 'interval', args: [lit(0), lit(1)] }],
    }, null),
    null, 'Uniform support is a set, not a scalar parameter');
});

// =====================================================================
// Route 1 — the inline normalize(truncate(M, S)) normalizer
// =====================================================================

// A truncated Normal(p, 1) on [−1, 2], scored at x = 0.5 with p fed from θ:
//   logφ(0.5 − p) − log(Φ(2 − p) − Φ(−1 − p))
// at p = 0.3 — mpmath, 50 digits.
const TRUNC_EXACT = -0.7865260699018717795;

const TRUNC_MODEL = (leaf: string) => H
  + 'p = elementof(reals)\n'
  + `tk = normalize(truncate(${leaf}, interval(-1.0, 2.0)))\n`
  + 'L = likelihoodof(tk, 0.5)\n'
  + 'ld = logdensityof(L, record(p = 0.3))\n';

test('the inline truncate normalizer resolves in both spellings', async () => {
  const pos = await scoreOf(TRUNC_MODEL('Normal(p, 1.0)'), 'ld');
  const kw = await scoreOf(TRUNC_MODEL('Normal(mu = p, sigma = 1.0)'), 'ld');
  assert.equal(pos, kw, `positional ${pos} vs keyword ${kw}`);
  assert.ok(Math.abs(kw - TRUNC_EXACT) < 1e-12,
    `engine ${kw} vs closed form ${TRUNC_EXACT}`);
});

// =====================================================================
// Route 2 — the §12 shared-variate product_dist normalizer
// =====================================================================

// The normalized product of Normal(0,1) and Normal(1,2) is Normal(μ*, σ*) with
// 1/σ*² = Σ 1/σᵢ² and μ* = σ*² Σ μᵢ/σᵢ², so μ* = 0.2 and σ* = √0.8. Its log-pdf
// at x = 0.83 — mpmath, 50 digits.
const PROD_EXACT = -1.0554292575475678639;

const PROD_MODEL = (g1: string, g2: string) => `
mu1 = elementof(reals)
sigma1 = elementof(posreals)
mu2 = elementof(reals)
sigma2 = elementof(posreals)
g1 = ${g1}
g2 = ${g2}
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
L = likelihoodof(prod, 0.83)
ld = logdensityof(L, record(mu1 = 0.0, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0))
`;

test('the shared-variate product normalizer resolves in both spellings', async () => {
  const pos = await scoreOf(PROD_MODEL('Normal(mu1, sigma1)', 'Normal(mu2, sigma2)'), 'ld');
  const kw = await scoreOf(
    PROD_MODEL('Normal(mu = mu1, sigma = sigma1)', 'Normal(mu = mu2, sigma = sigma2)'), 'ld');
  assert.equal(pos, kw, `positional ${pos} vs keyword ${kw}`);
  assert.ok(Math.abs(kw - PROD_EXACT) < 1e-12,
    `engine ${kw} vs closed form ${PROD_EXACT}`);
});

// =====================================================================
// Routes 3 and 4 — the exponential tilt of a leaf base
// =====================================================================
//
// e^{cx}·φ(x) = e^{c²/2}·φ(x − c), so `normalize(weighted(fn(exp(c·_)),
// Normal(0, 1)))` IS Normal(c, 1) and its log-pdf at 0.5 is the closed form
// below. A CONSTANT c takes mat-density's adaptive quadrature
// (`weightedLeafQuadLogZ`); c = θ takes `leaf-mass-quad`'s fixed graded rule.

// log N(0.5; 0.7, 1) — mpmath, 50 digits.
const TILT_FIXED_EXACT = -0.9389385332046727418;
// log N(0.5; 1.0, 1) — mpmath, 50 digits.
const TILT_THETA_EXACT = -1.0439385332046727418;

const TILT_FIXED = (leaf: string) => H
  + `m = normalize(weighted(x -> exp(0.7 * x), ${leaf}))\n`
  + 'ld = logdensityof(m, 0.5)\n';

test('a θ-independent tilt of a leaf base resolves in both spellings', async () => {
  const pos = await scoreOf(TILT_FIXED(N01_POS), 'ld');
  const kw = await scoreOf(TILT_FIXED(N01_KW), 'ld');
  assert.equal(pos, kw, `positional ${pos} vs keyword ${kw}`);
  assert.ok(Math.abs(kw - TILT_FIXED_EXACT) < 1e-12,
    `engine ${kw} vs closed form ${TILT_FIXED_EXACT}`);
  // The pre-fix positional value was the pooled-ensemble bake, 0.52 nats away,
  // so it is excluded rather than merely un-asserted.
  assert.ok(Math.abs(pos - (-1.4635483099639996)) > 0.4,
    `engine ${pos} is the pooled-mass value −1.4635483099639996`);
});

const TILT_THETA = (leaf: string) =>
  'theta ~ Uniform(interval(0.5, 2.0))\n'
  + `m = normalize(weighted(x -> exp(theta * x), ${leaf}))\n`
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

test('a θ-dependent tilt of a leaf base resolves in both spellings', async () => {
  const pos = await likAt(TILT_THETA(N01_POS), 1.0);
  const kw = await likAt(TILT_THETA(N01_KW), 1.0);
  assert.equal(pos, kw, `positional ${pos} vs keyword ${kw}`);
  // The graded rule's own accuracy, not the spelling fix, sets this tolerance.
  assert.ok(Math.abs(kw - TILT_THETA_EXACT) < 1e-7,
    `engine ${kw} vs closed form ${TILT_THETA_EXACT}`);
  assert.ok(Math.abs(pos - (-1.6923606583542328)) > 0.6,
    `engine ${pos} is the pooled-mass value −1.6923606583542328`);
});
