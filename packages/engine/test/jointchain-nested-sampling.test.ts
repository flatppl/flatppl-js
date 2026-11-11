'use strict';
// flatppl-js #612: a jointchain body is inline measure IR. Sampling it must
// reach the same handlers as a named binding (§06 jointchain, joint, iid,
// broadcast, pushfwd; §08 Dirichlet, Multinomial, MvNormal). Each case runs in
// chain form and in the direct form, where the auxiliary is a named draw.
//
// Oracles are closed form. Each moment check allows 6 standard errors of the
// per-atom statistic over N atoms. Every prior here is a product of
// normalized laws, so totalmass is 1. The joint log-densities are scipy 1.18.0
// values at fixed points.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { measureToRefValue } = require('../materialiser-shared.ts');

const N = 20000;

async function draw(src: string) {
  const { ctx } = makeMatCtx(src + 'tm = totalmass(prior)\n', { sampleCount: N, rootSeed: 612 });
  const m = await ctx.getMeasure('prior');
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs((tm.value ? tm.value.data : tm.samples)[0] - 1) < 1e-12, 'totalmass 1');
  assert.ok(!m.logWeights || m.logWeights.every((w: number) => w === m.logWeights[0]), 'unweighted atoms');
  return m;
}

// The [N, …] value of a field, and its per-atom shape.
function field(m: any, name: string) {
  assert.ok(m.fields && m.fields[name], `field '${name}' kept`);
  const v = measureToRefValue(m.fields[name], name, 'test');
  assert.equal(v.shape[0], N);
  return { shape: v.shape.slice(1), data: v.data, width: v.data.length / N };
}

// Mean of a per-atom statistic within 6 standard errors of `want`.
function mean(stat: (i: number) => number, want: number, label: string) {
  let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) { const x = stat(i); s += x; s2 += x * x; }
  const mu = s / N, se = Math.sqrt(Math.max(s2 / N - mu * mu, 0) / N);
  assert.ok(Math.abs(mu - want) < 6 * se + 1e-12, `${label}: ${mu} vs ${want} (se ${se})`);
}

async function logdensity(src: string, point: string) {
  const { ctx } = makeMatCtx(src + `lp = logdensityof(prior, ${point})\n`, { sampleCount: 1, rootSeed: 1 });
  const m = await ctx.getMeasure('lp');
  return (m.value ? m.value.data : m.samples)[0];
}

function close(got: number, want: number, label: string) {
  assert.ok(Math.abs(got - want) < 1e-12 * Math.max(1, Math.abs(want)), `${label}: ${got} vs ${want}`);
}

const CASES: Record<string, { chain: string; direct: string; check: (m: any) => void }> = {
  // Dirichlet(2, 3, 5): E[v] = alpha / 10. y | v ~ Normal(0, v1), so
  // E[y²] = E[v1²] = Var + mean² = 2·8/(100·11) + 0.04.
  dirichlet: {
    chain: 'prior = jointchain(joint(v = Dirichlet([2.0, 3.0, 5.0])), r -> joint(y = Normal(0.0, r.v[1])))\n',
    direct: 'v ~ Dirichlet([2.0, 3.0, 5.0])\ny ~ Normal(0.0, v[1])\nprior = lawof(record(v = v, y = y))\n',
    check(m) {
      const v = field(m, 'v'), y = field(m, 'y');
      assert.deepEqual(v.shape, [3]);
      assert.deepEqual(y.shape, []);
      for (let i = 0; i < N; i++) {
        assert.ok(Math.abs(v.data[3 * i] + v.data[3 * i + 1] + v.data[3 * i + 2] - 1) < 1e-12, 'on the simplex');
      }
      [0.2, 0.3, 0.5].forEach((a, j) => mean((i) => v.data[3 * i + j], a, `E[v${j + 1}]`));
      mean((i) => y.data[i], 0, 'E[y]');
      mean((i) => y.data[i] ** 2, 16 / 1100 + 0.04, 'E[y²]');
    },
  },
  // Multinomial(10, p): E[n] = 10 p and Σn = 10. y | n ~ Normal(n1, 1), so
  // E[y] = 2 and Var(y) = 1 + 10·0.2·0.8.
  multinomial: {
    chain: 'prior = jointchain(joint(n = Multinomial(10, [0.2, 0.3, 0.5])), r -> joint(y = Normal(r.n[1], 1.0)))\n',
    direct: 'n ~ Multinomial(10, [0.2, 0.3, 0.5])\ny ~ Normal(n[1], 1.0)\nprior = lawof(record(n = n, y = y))\n',
    check(m) {
      const n = field(m, 'n'), y = field(m, 'y');
      assert.deepEqual(n.shape, [3]);
      for (let i = 0; i < N; i++) {
        assert.equal(n.data[3 * i] + n.data[3 * i + 1] + n.data[3 * i + 2], 10, 'counts sum to 10');
      }
      [2, 3, 5].forEach((e, j) => mean((i) => n.data[3 * i + j], e, `E[n${j + 1}]`));
      mean((i) => y.data[i], 2, 'E[y]');
      mean((i) => (y.data[i] - 2) ** 2, 2.6, 'Var(y)');
    },
  },
  // s ~ Beta(2, 3): E[s] = 0.4, E[s²] = 0.04 + 0.16. y | s ~ MvNormal(0, diag(1, s)),
  // so E[y1²] = 1, E[y2²] = E[s] and E[y2² s] = E[s²]: the second ties y2 to its own s.
  mvnormal: {
    chain: 'prior = jointchain(joint(s = Beta(2.0, 3.0)), r -> joint(y = MvNormal([0.0, 0.0], diagmat([1.0, r.s]))))\n',
    direct: 's ~ Beta(2.0, 3.0)\ny ~ MvNormal([0.0, 0.0], diagmat([1.0, s]))\nprior = lawof(record(s = s, y = y))\n',
    check(m) {
      const s = field(m, 's'), y = field(m, 'y');
      assert.deepEqual(y.shape, [2]);
      mean((i) => s.data[i], 0.4, 'E[s]');
      mean((i) => y.data[2 * i], 0, 'E[y1]');
      mean((i) => y.data[2 * i + 1], 0, 'E[y2]');
      mean((i) => y.data[2 * i] ** 2, 1, 'E[y1²]');
      mean((i) => y.data[2 * i + 1] ** 2, 0.4, 'E[y2²]');
      mean((i) => y.data[2 * i + 1] ** 2 * s.data[i], 0.2, 'E[y2² s]');
      mean((i) => y.data[2 * i] * y.data[2 * i + 1], 0, 'E[y1 y2]');
    },
  },
  // Gamma(2, 1) cells: mean 2, variance 2. y ~ Normal(0, 1) is independent.
  matrixIid: {
    chain: 'prior = jointchain(joint(v = iid(Gamma(2.0, 1.0), [2, 3])), r -> joint(y = Normal(0.0, 1.0)))\n',
    direct: 'v ~ iid(Gamma(2.0, 1.0), [2, 3])\ny ~ Normal(0.0, 1.0)\nprior = lawof(record(v = v, y = y))\n',
    check(m) {
      const v = field(m, 'v'), y = field(m, 'y');
      assert.deepEqual(v.shape, [2, 3]);
      for (let j = 0; j < 6; j++) {
        mean((i) => v.data[6 * i + j], 2, `E[v cell ${j}]`);
        mean((i) => (v.data[6 * i + j] - 2) ** 2, 2, `Var(v cell ${j})`);
      }
      mean((i) => y.data[i] ** 2, 1, 'E[y²]');
    },
  },
  // v = (Gamma(2, 1), Gamma(3, 1)): E[v] = (2, 3). y | v ~ Normal(0, sqrt(v1)),
  // so E[y²] = E[v1] = 2 and E[y²/v1] = 1.
  heterogeneous: {
    chain: 'prior = jointchain(joint(v = joint(Gamma(2.0, 1.0), Gamma(3.0, 1.0))), r -> joint(y = Normal(0.0, sqrt(r.v[1]))))\n',
    direct: 'v ~ joint(Gamma(2.0, 1.0), Gamma(3.0, 1.0))\ny ~ Normal(0.0, sqrt(v[1]))\nprior = lawof(record(v = v, y = y))\n',
    check(m) {
      const v = field(m, 'v'), y = field(m, 'y');
      assert.equal(v.width, 2);
      mean((i) => v.data[2 * i], 2, 'E[v1]');
      mean((i) => v.data[2 * i + 1], 3, 'E[v2]');
      mean((i) => y.data[i] ** 2, 2, 'E[y²]');
      mean((i) => y.data[i] ** 2 / v.data[2 * i], 1, 'E[y²/v1]');
    },
  },
  // v ~ iid(Gamma(2, 1), 2), y_k | v ~ Normal(0, sqrt(v_k)): E[y_k²] = 2 and
  // E[y_k²/v_k] = 1 for each k.
  broadcast: {
    chain: 'prior = jointchain(joint(v = iid(Gamma(2.0, 1.0), 2)), r -> joint(y = Normal.(0.0, sqrt.(r.v))))\n',
    direct: 'v ~ iid(Gamma(2.0, 1.0), 2)\ny ~ Normal.(0.0, sqrt.(v))\nprior = lawof(record(v = v, y = y))\n',
    check(m) {
      const v = field(m, 'v'), y = field(m, 'y');
      assert.deepEqual(v.shape, [2]);
      assert.deepEqual(y.shape, [2]);
      for (let k = 0; k < 2; k++) {
        mean((i) => y.data[2 * i + k] ** 2, 2, `E[y${k + 1}²]`);
        mean((i) => y.data[2 * i + k] ** 2 / v.data[2 * i + k], 1, `E[y${k + 1}²/v${k + 1}]`);
      }
    },
  },
};

for (const [name, c] of Object.entries(CASES)) {
  test(`${name}: the chain form samples with its fields, shapes and moments`, async () => {
    c.check(await draw(c.chain));
  });
  test(`${name}: the direct form samples with its fields, shapes and moments`, async () => {
    c.check(await draw(c.direct));
  });
}

test('the chain joint log-densities match scipy', async () => {
  const pin: Array<[string, string, number]> = [
    [CASES.dirichlet.chain, 'record(v = [0.2, 0.3, 0.5], y = 0.1)', 2.7061536050772528],
    [CASES.multinomial.chain, 'record(n = [2, 3, 5], y = 2.5)', -3.508454493344939],
    [CASES.mvnormal.chain, 'record(s = 0.4, y = [0.3, -0.2])', -0.927767030090404],
    [CASES.matrixIid.chain, 'record(v = rowstack([[1.0, 2.0, 0.5], [1.5, 0.7, 3.0]]), y = 0.2)', -8.49153608036713],
    [CASES.heterogeneous.chain, 'record(v = [1.5, 2.5], y = 0.7)', -3.7401050292955587],
    [CASES.broadcast.chain, 'record(v = [1.5, 2.5], y = [0.7, -0.4])', -5.372332479751519],
  ];
  for (const [src, point, want] of pin) close(await logdensity(src, point), want, point);
});
