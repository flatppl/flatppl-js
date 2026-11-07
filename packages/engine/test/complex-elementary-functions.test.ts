// §07 "Elementary functions": the rows whose domain lists `complexes` take a
// complex argument and return a complex result on the principal branch.
// Oracle: Python `cmath` (principal branches, C99 Annex G), values frozen
// below. acosh used log(z + sqrt(z² − 1)), which is −acosh(z) for Re(z) < 0;
// typeinfer refused tan, asin, acos, atan and the hyperbolic rows on a complex
// argument and typed exp, log, sqrt, sin, cos of a complex as real.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { _cAcosh } = require('../sampler-complex.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// cmath.<f>(complex(-2, 0.5))
const AT: Record<string, [number, number]> = {
  exp: [0.11876788457694579, 0.06488319105786541],
  log: [0.7234594914681627, 2.896613990462929],
  sqrt: [0.17543205637629383, 1.425053124063947],
  sin: [-1.0253473885839877, -0.21685216292078974],
  cos: [-0.4692579782290534, 0.473830620416407],
  tan: [0.8508781211449379, 1.3212865837711916],
  asin: [-1.2930420702371825, 1.3618009008578458],
  acos: [2.863838397032079, -1.3618009008578458],
  atan: [-1.1265564408348223, 0.09641562020299617],
  sinh: [-3.182869448337149, 1.8036926955321817],
  cosh: [3.3016373329140944, -1.7388095044743164],
  tanh: [-0.979940849961738, 0.030215987322877582],
  asinh: [-1.4657153519472905, 0.22101863562288385],
  acosh: [1.3618009008578458, 2.863838397032079],
  atanh: [-0.5003700000525311, 1.421546861001807],
};

// [re z, im z, re acosh z, im acosh z] from cmath.acosh
const ACOSH = [
  [-2.0, 0.0, 1.3169578969248166, 3.141592653589793],
  [-3.0, -2.0, 1.9686379257930964, -2.5354548312024994],
  [-1.5, 0.4, 1.0357222955512628, 2.8106691284403063],
  [-0.5, 1.8, 1.3758943430333128, 1.8105469262044116],
  [0.9, -0.7, 0.831390099756731, -0.8514814390102962],
  [2.5, 1.8, 1.8107911995694492, 0.6495940406405247],
  [0.5, 0.0, 0.0, 1.0471975511965979],
  [3.0, 0.0, 1.762747174039086, 0.0],
];

const close = (a: number, b: number, scale: number) => Math.abs(a - b) <= 1e-14 * Math.max(1, scale);

test('complex acosh is the principal branch in every quadrant', () => {
  for (const [re, im, wr, wi] of ACOSH) {
    const r = _cAcosh({ re, im });
    const s = Math.hypot(wr, wi);
    assert.ok(close(r.re, wr, s) && close(r.im, wi, s),
      `acosh(${re}${im < 0 ? '' : '+'}${im}i) = ${r.re}+${r.im}i, cmath ${wr}+${wi}i`);
  }
});

test('each complex-domain elementary function types and evaluates a complex argument', async () => {
  for (const [f, [wr, wi]] of Object.entries(AT)) {
    const src = `z = ${f}(complex(-2.0, 0.5))\n`;
    const lifted = processSource(src);
    assert.deepEqual(lifted.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), [], f);
    const { ctx } = makeMatCtx(src, { sampleCount: 2 });
    const v = (await ctx.getMeasure('z')).value;
    assert.equal(v.dtype, 'complex', `${f} result is complex`);
    const s = Math.hypot(wr, wi);
    assert.ok(Math.abs(v.data[0] - wr) <= 1e-14 * s && Math.abs(v.im[0] - wi) <= 1e-14 * s,
      `${f}(-2+0.5i) = ${v.data[0]}+${v.im[0]}i, cmath ${wr}+${wi}i`);
  }
});

test('the real-only rows refuse a complex argument statically', () => {
  for (const f of ['log2', 'log10', 'floor', 'ceil', 'round']) {
    const errs = processSource(`z = ${f}(complex(-2.0, 0.5))\n`).diagnostics
      .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
    assert.deepEqual(errs, [`${f}: arg 1 expects real, got complex`], f);
  }
});
