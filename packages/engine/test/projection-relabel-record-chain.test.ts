'use strict';

// §06 case 2: an engine must support projections onto whole factors of a
// `joint` product and onto a `jointchain` prefix, and omitted factors
// contribute their total masses. By-name projection over `relabel(joint(...))`
// and over a chain of record-valued steps took the general pushfwd path and
// asked for a bijection annotation. Each expected value is a closed-form sum
// of log densities: log N(x; m, s) and log Exponential(x; rate).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const lnNormal = (x: number, m: number, s: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const lnExponential = (x: number, rate: number) => Math.log(rate) - rate * x;

async function score(src: string) {
  const { proc, ctx } = ctxFor(src, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return (await ctx.getMeasure('lp')).samples[0];
}

function close(got: number, want: number) {
  assert.ok(Math.abs(got - want) < 1e-12, `${got} != ${want}`);
}

const RELABEL = 'jc = relabel(joint(Normal(0.0, 1.0), Exponential(1.0), Normal(1.0, 2.0)), ["a", "b", "c"])\n';

test('a by-name projection of relabel(joint) scores the kept factor', async () => {
  close(await score(RELABEL + 'lp = logdensityof(pushfwd(fn(get(_, ["a"])), jc), record(a = 0.3))\n'),
    lnNormal(0.3, 0, 1));
  close(await score(RELABEL + 'lp = logdensityof(pushfwd(fn(get(_, "b")), jc), 0.3)\n'),
    lnExponential(0.3, 1));
});

test('a by-name projection of relabel(joint) keeps factors in selector order', async () => {
  close(await score(RELABEL + 'lp = logdensityof(pushfwd(fn(get(_, ["c", "a"])), jc), record(c = 0.5, a = 0.3))\n'),
    lnNormal(0.5, 1, 2) + lnNormal(0.3, 0, 1));
});

test('a dropped factor of relabel(joint) contributes its total mass', async () => {
  const src = 'jc = relabel(joint(Normal(0.0, 1.0), weighted(2.0, Exponential(1.0))), ["a", "b"])\n'
    + 'lp = logdensityof(pushfwd(fn(get(_, ["a"])), jc), record(a = 0.3))\n';
  close(await score(src), lnNormal(0.3, 0, 1) + Math.log(2));
});

test('a projection of relabel(joint) samples the kept factor', async () => {
  const { ctx } = ctxFor(RELABEL + 'P = pushfwd(fn(get(_, ["b"])), jc)\n', 4000);
  const P = await ctx.getMeasure('P');
  assert.deepEqual(Object.keys(P.fields), ['b']);
  const b = P.fields.b.samples;
  const mean = b.reduce((s: number, x: number) => s + x, 0) / b.length;
  assert.ok(Math.abs(mean - 1) < 0.08, `Exponential(1) mean ${mean}`);
});

// a ~ N(0, 1), b | a ~ Exponential(a), c | b ~ N(b, 2), each step a record.
const CHAIN = `
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Exponential(rate = z))), z = z)
w = draw(Normal(mu = 0.0, sigma = 1.0))
kc = kernelof(record(c = draw(Normal(mu = w, sigma = 2.0))), w = w)
jc = jointchain(lawof(record(a = z)), pars -> k(z = pars.a), pars -> kc(w = pars.b))
`;

test('a by-name prefix projection of a record jointchain scores the first step', async () => {
  close(await score(CHAIN + 'lp = logdensityof(pushfwd(fn(get(_, ["a"])), jc), record(a = 0.3))\n'),
    lnNormal(0.3, 0, 1));
  close(await score(CHAIN + 'lp = logdensityof(pushfwd(fn(get(_, "a")), jc), 0.3)\n'),
    lnNormal(0.3, 0, 1));
});

test('a by-name prefix projection of a record jointchain scores the kept steps', async () => {
  close(await score(CHAIN + 'lp = logdensityof(pushfwd(fn(get(_, ["a", "b"])), jc), record(a = 0.3, b = 0.5))\n'),
    lnNormal(0.3, 0, 1) + lnExponential(0.5, 0.3));
});

test('a record jointchain selector that is not a prefix keeps the refusal', async () => {
  for (const sel of ['["b", "a"]', '["b"]', '["a", "c"]']) {
    const { ctx } = ctxFor(CHAIN + `lp = logdensityof(pushfwd(fn(get(_, ${sel})), jc), record(a = 0.3, b = 0.5))\n`, 1);
    await assert.rejects(ctx.getMeasure('lp'), /requires a bijection annotation/, sel);
  }
});
