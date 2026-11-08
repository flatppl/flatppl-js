'use strict';
// §06 pushfwd: log p_{f*M}(y) = log p_M(f_inv(y)) − logvolume(f_inv(y)). A
// pushforward may change the variate space: §04 relabel(M, names) ≡
// pushfwd(fn(relabel(_, names)), M) maps reals onto records. A record point
// or a record preimage used to throw "cannot consume scalar from value of
// type object".
//
// §07 ifelse(cond, a, b) "returns a if cond is true, b otherwise" for any a,
// b, measures included. A fixed condition, and an inline composite branch,
// used to get no derivation.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

async function score(src: string, name = 'lp'): Promise<number> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure(name);
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}
function close(got: number, want: number, label: string) {
  assert.ok(Math.abs(got - want) < 1e-12, `${label}: got ${got}, want ${want}`);
}

const P = 'p = joint(mu = Normal(0.0, 2.0))\n';

test('a pushforward between record and scalar variate spaces scores by §06', async () => {
  close(await score(P + 'lp = logdensityof(pushfwd(bijection(fn(record(nu = _.mu)), fn(record(mu = _.nu)), 0.0), p), record(nu = 0.7))\n'),
    lnN(0.7, 0, 2), 'record relabel');
  close(await score(P + `lp = logdensityof(pushfwd(bijection(fn(exp(_.mu)), fn(record(mu = log(_))), fn(_.mu)), p), ${Math.exp(0.7)})\n`),
    lnN(0.7, 0, 2) - 0.7, 'record to scalar exp');
  assert.equal(await score(P + 'lp = logdensityof(pushfwd(bijection(fn(exp(_.mu)), fn(record(mu = log(_))), fn(_.mu)), p), -1.0)\n'),
    -Infinity, 'outside the image');
});

test('an inverse that does not return the base\'s variates refuses accurately', async () => {
  for (const [src, fields] of [
    [P + 'lp = logdensityof(pushfwd(bijection(fn(record(nu = _.mu)), fn(record(zz = _.nu)), 0.0), p), record(nu = 0.7))\n', 'zz'],
    ['lp = logdensityof(pushfwd(bijection(fn(_ * 2.0), fn(record(a = _ / 2.0)), log(2.0)), Normal(0.0, 1.0)), 1.0)\n', 'a'],
  ]) {
    await assert.rejects(() => score(src), new RegExp(`pushfwd: the bijection's inverse returned a record with fields ${fields}, `
      + 'which is not a variate of the base measure; f_inv must return the base measure\'s variates \\(spec §06 pushfwd\\)'));
  }
});

const BRANCHES: Array<[string, string, string, number]> = [
  ['inline scalar branches', 'Normal(0.0, 2.0)', 'Normal(0.0, 1.0)', 0.7],
  ['inline record branches', 'joint(mu = Normal(0.0, 2.0))', 'joint(mu = Normal(0.0, 1.0))', 0.7],
];

test('a fixed ifelse condition selects its measure (§07)', async () => {
  for (const [label, a, b] of BRANCHES) {
    const at = a.startsWith('joint') ? 'record(mu = 0.7)' : '0.7';
    close(await score(`c = 1.0\nm = ifelse(c > 0.0, ${a}, ${b})\nlp = logdensityof(m, ${at})\n`), lnN(0.7, 0, 2), label + ', true');
    close(await score(`c = -1.0\nm = ifelse(c > 0.0, ${a}, ${b})\nlp = logdensityof(m, ${at})\n`), lnN(0.7, 0, 1), label + ', false');
  }
  close(await score('c = 1.0\nA = joint(mu = Normal(0.0, 2.0))\nB = joint(mu = Normal(0.0, 1.0))\nm = ifelse(c > 0.0, A, B)\nlp = logdensityof(m, record(mu = 0.7))\n'),
    lnN(0.7, 0, 2), 'named record branches');
});

test('a Bernoulli selector over inline record branches scores the mixture', async () => {
  const want = Math.log(0.3 * Math.exp(lnN(0.7, 0, 2)) + 0.7 * Math.exp(lnN(0.7, 0, 1)));
  close(await score('b = draw(Bernoulli(0.3))\nm = ifelse(b, joint(mu = Normal(0.0, 2.0)), joint(mu = Normal(0.0, 1.0)))\nlp = logdensityof(m, record(mu = 0.7))\n'),
    want, 'mixture');
});

test('a fixed ifelse condition samples only the selected branch', async () => {
  // Branch a is Normal(0, 2). 4000 draws: the mean has sd 2/sqrt(4000) ≈ 0.032,
  // and the sample sd is within a few percent of 2; branch b (sd 1) is far off.
  const N = 4000;
  const { ctx } = makeMatCtx('c = 1.0\nm = ifelse(c > 0.0, Normal(0.0, 2.0), Normal(0.0, 1.0))\ny = draw(m)\n', { sampleCount: N });
  const s = (await ctx.getMeasure('y')).samples as Float64Array;
  let mean = 0; for (let i = 0; i < N; i++) mean += s[i]; mean /= N;
  let v = 0; for (let i = 0; i < N; i++) v += (s[i] - mean) ** 2; const sd = Math.sqrt(v / (N - 1));
  assert.ok(Math.abs(mean) < 6 * 2 / Math.sqrt(N), `mean ${mean}`);
  assert.ok(Math.abs(sd - 2) < 0.15, `sd ${sd}`);
});
