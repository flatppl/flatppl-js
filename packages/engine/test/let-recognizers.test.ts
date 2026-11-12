'use strict';
// A recognizer that matches an expression by shape reads a `let` graph the
// way it reads the inlined tree (let-view.ts). Each model below binds an
// intermediate read twice, so the engine evaluates it once in a `let`. It and
// its plain spelling score the same closed form.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function score(source: string, name = 'lp') {
  const { ctx } = ctxFor(source, 1);
  return (await ctx.getMeasure(name)).samples[0];
}
function close(got: number, want: number, tol = 2e-12) {
  assert.ok(Math.abs(got - want) < tol, `${got} vs ${want}`);
}
function hasLet(ir: any): boolean {
  const seen = new Set<any>();
  const go = (n: any): boolean => {
    if (!n || typeof n !== 'object' || seen.has(n)) return false;
    seen.add(n);
    if (n.kind === 'call' && n.op === 'let') return true;
    return Object.keys(n).some((k) => k !== 'loc' && go(n[k]));
  };
  return go(ir);
}
function bodyHasLet(source: string, fn: string) {
  const { ctx } = ctxFor(source, 1);
  assert.ok(hasLet(ctx.bindings.get(fn).ir.body), `${fn} has no let`);
}
const normal = (x: number, mu: number, sigma: number) =>
  -0.5 * ((x - mu) / sigma) ** 2 - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);

// The truncated-box normalizer over [0, 1]^d at theta = 1, as in
// exact-nd-truncate.test.ts, with the weight a reified function that reads
// theta.
function boxModel(defs: string, body: string, d: number, point: string, region?: string) {
  const box = region || `cartprod(${Array(d).fill('interval(0.0, 1.0)').join(', ')})`;
  return `theta ~ Uniform(interval(0.5, 2.0))\nvv = elementof(cartpow(reals, ${d}))\n`
    + defs + `w = functionof(${body}, vv)\n`
    + `M = normalize(truncate(weighted(w, Lebesgue(${box})), ${box}))\n`
    + `y ~ M\nL = likelihoodof(kernelof(y, theta = theta), ${point})\nlp = logdensityof(L, record(theta = 1.0))\n`;
}

test('a polynomial box integral reads a shared term', async () => {
  // s = yz is read twice: ∫[0,1]^3 (t + x^9 + 2yz) = 1.6.
  const src = boxModel('s = vv[2] * vv[3]\n', 'theta + vv[1]^9 + s + s', 3, '[0.2, 0.3, 0.4]');
  bodyHasLet(src, 'w');
  const want = Math.log((1 + 0.2 ** 9 + 2 * 0.3 * 0.4) / 1.6);
  close(await score(src), want);
  close(await score(boxModel('', 'theta + vv[1]^9 + 2.0*vv[2]*vv[3]', 3, '[0.2, 0.3, 0.4]')), want);
});

test('an affine-exponential box integral reads a shared exponent term', async () => {
  // a = t x is read twice: exp(a + 2y + a - a) = exp(x + 2y) at t = 1, with
  // ∫[0,1]^2 = expm1(1) expm1(2) / 2.
  const src = boxModel('a = theta * vv[1]\n', 'exp(a + 2.0*vv[2] + a - a)', 2, '[0.9, 0.9]');
  bodyHasLet(src, 'w');
  const want = 2.7 - Math.log(Math.expm1(1) * Math.expm1(2) / 2);
  close(await score(src), want);
  close(await score(boxModel('', 'exp(theta*vv[1] + 2.0*vv[2])', 2, '[0.9, 0.9]')), want);
});

test('a separable Gaussian box mass reads a shared scale', async () => {
  // k = t/2 is read twice and cancels: the O1 oracle of exact-nd-truncate.
  const region = 'cartprod(interval(-2.0, 3.0), interval(-1.0, 2.0))';
  const src = boxModel('k = theta * 0.5\n', 'k*exp(-0.5*(vv[1]^2 + vv[2]^2))/(k*2.0*pi)', 2, '[0.0, 0.0]', region);
  bodyHasLet(src, 'w');
  const want = -Math.log(2 * Math.PI) + 0.2245614818793499;
  close(await score(src), want);
  close(await score(boxModel('', 'theta*0.5*exp(-0.5*(vv[1]^2 + vv[2]^2))/(theta*0.5*2.0*pi)', 2, '[0.0, 0.0]', region)), want);
});

test('step cuts read a shared step argument', async () => {
  // u = 4y is read twice, both times inside a test or a step: strips weigh
  // 1, 2, 3, 4, so Z = 2.5 and the queried weight is 4.
  const src = boxModel('u = 4.0 * vv[2]\n', 'theta + floor(u) + ifelse(u < 2.0, 0.0, 0.0)', 2, '[0.9, 0.9]');
  bodyHasLet(src, 'w');
  close(await score(src), Math.log(4 / 2.5));
  close(await score(boxModel('', 'theta + floor(4.0*vv[2]) + ifelse(4.0*vv[2] < 2.0, 0.0, 0.0)', 2, '[0.9, 0.9]')),
    Math.log(4 / 2.5));
});

test('a polynomial bin intensity reads shared coefficients', async () => {
  // c = 1 is read twice: polynomial([c, c + c], x) = 1 + 2x, so the bins
  // [0, 0.5] and [0.5, 1] have rates 3/4 and 5/4.
  const src = 'xx = elementof(reals)\nc = exp(0.0) + 0.0\n'
    + 'p = functionof(polynomial([c, c + c], xx), xx)\n'
    + 'I = weighted(p, Lebesgue(interval(0, 1)))\nM = BinnedPoissonProcess([-1, 0, 0.5, 1, 2], I)\n'
    + 'lp = logdensityof(M, [0, 1, 2, 0])\n';
  bodyHasLet(src, 'p');
  const want = Math.log(0.75) + 2 * Math.log(1.25) - Math.log(2) - 0.75 - 1.25;
  close(await score(src), want);
  close(await score(src.replace('c = exp(0.0) + 0.0\n', '').replace('[c, c + c]', '[1.0, 2.0]')), want);
});

test('a symbolic inverse reads a shared frozen factor', () => {
  // s = e^0.5 is let-bound and read twice: y = s x + s, so x = (y − s)/s and
  // the forward log-Jacobian is log s.
  const { invertExpr } = require('../bijection-registry.ts');
  const { evaluateExpr } = require('../sampler.ts');
  const ref = (name: string) => ({ kind: 'ref', ns: '%local', name });
  const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
  const outputExpr = { kind: 'call', op: 'let',
    assigns: [{ name: 's', value: call('exp', { kind: 'lit', value: 0.5 }) }],
    body: call('add', call('mul', ref('s'), ref('x')), ref('s')) };
  const sIR = call('exp', { kind: 'lit', value: 0.5 });
  const plainExpr = call('add', call('mul', sIR, ref('x')), sIR);
  const s = Math.exp(0.5);
  for (const expr of [outputExpr, plainExpr]) {
    const inv = invertExpr({ outputExpr: expr, freeRef: { ns: '%local', name: 'x' }, outputValue: ref('y') });
    assert.ok(inv, 'not inverted');
    close(+evaluateExpr(inv.inverseIR, { y: 2 }), (2 - s) / s, 1e-15);
    close(+evaluateExpr(inv.ladjIR, { x: (2 - s) / s }), 0.5, 1e-15);
  }
});

test('a linear-Gaussian marginal reads a shared affine term', async () => {
  // a = 2 mu is read twice in y's mean, so y = 4 mu + e has the marginal
  // Normal(0, √17).
  const src = 'mu ~ Normal(0, 1)\na = 2.0 * mu\ny ~ Normal(a + a, 1)\nlp = logdensityof(lawof(y), 1.0)\n';
  const { ctx } = ctxFor(src, 1);
  const { clm, orchestrator } = require('..');
  const lowered = clm.lowerMeasure(orchestrator.expandMeasure('y', ctx), ctx, {});
  assert.ok(hasLet(lowered?.body), 'the mean is not a let');
  close(await score(src), normal(1, 0, Math.sqrt(17)), 1e-12);
  close(await score('mu ~ Normal(0, 1)\ny ~ Normal(4.0 * mu, 1)\nlp = logdensityof(lawof(y), 1.0)\n'),
    normal(1, 0, Math.sqrt(17)), 1e-12);
});

test('a truncation set reads a shared bound', async () => {
  // c = θ/2 is read twice in the set: at θ = 1 it is interval(−0.5, 1.5).
  const src = 'theta ~ Uniform(interval(0.5, 2.0))\nc = theta * 0.5\n'
    + 'y ~ truncate(Normal(0, 1), interval(c - 1.0, c + 1.0))\nK = kernelof(y, theta = theta)\n'
    + 'inside = logdensityof(likelihoodof(K, 0.3), record(theta = 1.0))\n'
    + 'outside = logdensityof(likelihoodof(K, 1.8), record(theta = 1.0))\n';
  const plain = src.replace('c = theta * 0.5\n', '').replace('interval(c - 1.0, c + 1.0)',
    'interval(theta * 0.5 - 1.0, theta * 0.5 + 1.0)');
  for (const model of [src, plain]) {
    close(await score(model, 'inside'), normal(0.3, 0, 1));
    assert.equal(await score(model, 'outside'), -Infinity);
  }
});

test('an aggregate body evaluates a let-form user call', async () => {
  // normal_lpdf reads m and s twice, so its call in the aggregate body binds
  // p.mu[.k] and p.sigma[.k] once per cell. The scan sums every cell: the
  // log density of each x_t under each (mu_k, sigma_k).
  const src = 'x = [9.0, 18.0, 10.0]\nnormal_lpdf(y, m, s) = -0.5 * ((y - m) / s)^2 - log(s) - 0.5 * log(2 * pi)\n'
    + 'step(acc, t, q) = acc + sum(q.ll[t, :])\n'
    + 'log_target(p) = scan(step, 0.0, [1, 2, 3], record(ll = aggregate(sum, [.t, .k], '
    + 'normal_lpdf(x[.t], p.mu[.k], p.sigma[.k]))))[3]\n'
    + 'Pi = Lebesgue(cartprod(mu = cartpow(reals, 2), sigma = cartpow(posreals, 2)))\n'
    + 'M = logweighted(log_target, Pi)\nlp = logdensityof(M, record(mu = [9.0, 18.0], sigma = [1.0, 2.0]))\n';
  let want = 0;
  for (const y of [9, 18, 10]) for (const [m, s] of [[9, 1], [18, 2]]) want += normal(y, m, s);
  close(await score(src), want);
  close(await score(src.replace('normal_lpdf(x[.t], p.mu[.k], p.sigma[.k])',
    '-0.5 * ((x[.t] - p.mu[.k]) / p.sigma[.k])^2 - log(p.sigma[.k]) - 0.5 * log(2 * pi)')), want);
});
