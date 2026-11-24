'use strict';

// Densities the measure algebra does not give, and the maps it does.
//
//   * §06 measure addition needs one reference measure. §03 gives a
//     `stdsimplex(n)` or `zerosum(n)` support n − 1 free coordinates, so such a
//     law beside a full-dimensional one has no density, mapped or not.
//   * §06 `pushfwd` applies f to the variate. A scalar-domain built-in (§07)
//     over an array variate is ill-typed: the elementwise map is `fn(exp.(_))`
//     (§04 "Broadcasting"). An array-domain built-in such as `neg` is a map of
//     the array.
//   * §03 and §06 give the image of a simplex or zero-sum law a density only
//     under a uniform affine map. Any other image, and any image of LKJ or
//     LKJCholesky, is singular (§06 "Singular laws").
//   * §06 `tilt` onto a point mass is that point mass. An engine that cannot
//     evaluate the normalizer refuses the query, and this one evaluates none
//     over a point mass.
//
// Expected values come from scipy 1.18.1 or closed forms, never from an engine.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const DIRICHLET = 'Dirichlet(alpha = [2.0, 3.0, 4.0])';

function compiled(measure: string, point: string, prefix = '') {
  return ctxFor(`${prefix}\nM = ${measure}\n__score__ = logdensityof(M, ${point})`, 1);
}

async function score(measure: string, point: string, prefix = ''): Promise<number> {
  const { proc, ctx } = compiled(measure, point, prefix);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const result = await ctx.getMeasure('__score__');
  return result.value.data[0];
}

async function refused(measure: string, point: string, reason: RegExp, prefix = '') {
  const { ctx } = compiled(measure, point, prefix);
  await assert.rejects(() => ctx.getMeasure('__score__'), reason);
}

function staticErrors(measure: string, point: string): string[] {
  const { proc } = compiled(measure, point);
  return proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

test('a superposition of a simplex law and a full-dimensional law has no density', async () => {
  const mixed = `superpose(weighted(0.5, ${DIRICHLET}), weighted(0.5, iid(Normal(0.0, 1.0), 3)))`;
  const reason = /mixture branches have different reference dimensions/;
  await refused(mixed, '[0.2, 0.3, 0.5]', reason);
  await refused('S', '[0.2, 0.3, 0.5]', reason, `S = ${mixed}`);
  await refused(`pushfwd(x -> 2.0 .* x .+ 1.0, ${mixed})`, '[1.4, 1.6, 2.0]', reason);
});

test('a product of simplex laws lacks the sum of their free coordinates', async () => {
  // Both components lack two coordinates of five, so the mixture has a density:
  // 0.5 · Dir(0.2, 0.3, 0.5; 2, 3, 4) · Dir(0.4, 0.6; 1, 2)
  //   + 0.5 · Dir(0.2, 0.3, 0.5; 1, 1, 1) · Dir(0.4, 0.6; 2, 2)
  //   = 0.5 · 9.072 + 0.5 · 2.88 = 5.976 (scipy: log = 1.7877514478305174).
  const product = (a: string, b: string) => `joint(Dirichlet(alpha = ${a}), Dirichlet(alpha = ${b}))`;
  const charts = `normalize(superpose(weighted(0.5, ${product('[2.0, 3.0, 4.0]', '[1.0, 2.0]')}), `
    + `weighted(0.5, ${product('[1.0, 1.0, 1.0]', '[2.0, 2.0]')})))`;
  assert.ok(Math.abs(await score(charts, '[0.2, 0.3, 0.5, 0.4, 0.6]') - 1.7877514478305174) < 1e-12);
  // Beside a five-cell full-dimensional law the product shares no reference measure.
  await refused(`superpose(weighted(0.5, ${product('[2.0, 3.0, 4.0]', '[1.0, 2.0]')}), `
    + 'weighted(0.5, iid(Normal(0.0, 1.0), 5)))', '[0.2, 0.3, 0.5, 0.4, 0.6]',
  /mixture branches have different reference dimensions/);
});

test('superpositions over one reference measure keep their density', async () => {
  // Dirichlet(1, 1, 1) has density Γ(3) = 2 on the simplex.
  const simplex = `superpose(weighted(0.5, ${DIRICHLET}), weighted(0.5, Dirichlet(alpha = [1.0, 1.0, 1.0])))`;
  assert.ok(Math.abs(await score(simplex, '[0.2, 0.3, 0.5]') - 1.564440546503366) < 1e-12);
  const normals = 'superpose(weighted(0.5, Normal(0.0, 1.0)), weighted(0.5, Normal(1.0, 1.0)))';
  assert.ok(Math.abs(await score(normals, '0.5') - -1.0439385332046727) < 1e-12);
  const counts = 'superpose(weighted(0.5, Bernoulli(0.3)), weighted(0.5, Bernoulli(0.6)))';
  assert.ok(Math.abs(await score(counts, 'true') - Math.log(0.45)) < 1e-14);
});

test('a scalar built-in over an array variate names the elementwise map', () => {
  for (const map of ['exp', 'bijection(exp, log, x -> x)']) {
    const errors = staticErrors(`pushfwd(${map}, iid(Normal(0.0, 1.0), 3))`, '[1.0, 2.0, 3.0]');
    assert.equal(errors.length, 1, map);
    assert.match(errors[0], /pushfwd: the map `exp` takes a scalar.*write the elementwise map `fn\(exp\.\(_\)\)`/, map);
  }
});

test('the elementwise and scalar spellings of a built-in map keep their density', async () => {
  // Σ log φ(log y) − Σ log y at y = [1, 2, 3], and log φ(log 2) − log 2.
  const lognormal3 = -5.392276056207464;
  assert.ok(Math.abs(await score('pushfwd(fn(exp.(_)), iid(Normal(0.0, 1.0), 3))', '[1.0, 2.0, 3.0]') - lognormal3) < 1e-12);
  assert.ok(Math.abs(await score('pushfwd(exp, Normal(0.0, 1.0))', '2.0') - -1.8523122207237188) < 1e-12);
});

test('an array-domain built-in maps the whole variate', async () => {
  // neg is volume preserving: Σ log φ(x) at x = [0.1, -0.2, 0.3].
  assert.ok(Math.abs(await score('pushfwd(neg, iid(Normal(0.0, 1.0), 3))', '[-0.1, 0.2, -0.3]') - -2.826815599614018) < 1e-12);
});

test('an array-domain built-in with no inverse names itself in the refusal', async () => {
  for (const [map, base, point] of [
    ['cumsum', 'iid(Normal(0.0, 1.0), 3)', '[0.5, 1.0, 0.2]'],
    ['softmax', 'iid(Normal(0.0, 1.0), 3)', '[0.2, 0.3, 0.5]'],
    ['inv', 'iid(Normal(0.0, 1.0), [2, 2])', 'rowstack([[1.0, 0.2], [0.1, 1.0]])'],
  ]) {
    await refused(`pushfwd(${map}, ${base})`, point,
      new RegExp(`pushfwd of '${map}' requires a bijection annotation`));
  }
});

test('a map of a lower-dimensional law that is not uniform affine has no density', async () => {
  const reason = /pushfwd of a law on a lower-dimensional support.*singular/;
  await refused(`pushfwd(x -> exp.(x), ${DIRICHLET})`,
    '[1.2214027581601699, 1.3498588075760032, 1.6487212707001282]', reason);
  await refused(`pushfwd(x -> [1.0, 2.0, 3.0] .* x, ${DIRICHLET})`, '[0.2, 0.6, 1.5]', reason);
  await refused('pushfwd(x -> exp.(x), Lebesgue(zerosum(3)))', '[1.0, 1.0, 1.0]', reason);
  const twoI = 'rowstack([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])';
  await refused('pushfwd(x -> 2.0 .* x, LKJ(3, 2.0))', twoI, reason);
  await refused('pushfwd(x -> 2.0 .* x, LKJCholesky(3, 2.0))', twoI, reason);
});

test('a uniform affine map of a lower-dimensional law counts the free coordinates', async () => {
  // log Dirichlet((y − 1) / 2) − 2 log 2; mpmath 0.63657682907155101.
  assert.ok(Math.abs(await score(`pushfwd(x -> 2.0 .* x .+ 1.0, ${DIRICHLET})`, '[1.4, 1.6, 2.0]') - 0.63657682907155101) < 1e-12);
  assert.ok(Math.abs(await score('pushfwd(x -> 2.0 .* x .+ 3.0, Lebesgue(zerosum(3)))', '[3.5, 2.5, 3.0]') + 2 * Math.log(2)) < 1e-12);
});

test('a tilt onto a point mass is refused with its reason, mapped or not', async () => {
  const reason = /refused a tilt onto a point mass/;
  await refused('pushfwd(fn(2.0 * _ + 1.0), tilt(Normal(0.0, 1.0), Dirac(0.5)))', '2.0', reason);
  await refused('tilt(Normal(0.0, 1.0), Dirac(0.5))', '0.5', reason);
});
