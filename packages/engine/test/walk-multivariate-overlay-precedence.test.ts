// A joint whose later MvNormal field reads an earlier observed field `a`
// while refArrays also carries per-atom prior draws under the name `a`
// (matLogdensityof passes prior samples under the binding names the
// observation pins). The observed value must win for every atom, as the
// overlay > refArrays > baseEnv rule says. walkMultivariate resolved atom 0's
// kwargs with refArrays written over the overlay, so atom 0 scored the prior
// draw. Oracle: the closed-form Normal and independent-Normal densities.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const density = require('../density.ts');

const lit = (v: number) => ({ kind: 'lit', value: v });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const vec = (args: any[]) => ({ kind: 'call', op: 'vector', args });
const lnN = (x: number, m: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;

test('an observed field threaded into a multivariate sibling wins over a per-atom draw of the same name', () => {
  const N = 4;
  const aPrior = new Float64Array([9, -9, 7, -7]);
  const c = new Float64Array([0, 1, -1, 2]);
  const cov = { kind: 'call', op: 'rowstack', args: [vec([vec([lit(1), lit(0)]), vec([lit(0), lit(1)])])] };
  const ir = {
    kind: 'call', op: 'joint', fields: [
      { name: 'a', value: { kind: 'call', op: 'Normal', kwargs: { mu: ref('c'), sigma: lit(1) } } },
      { name: 'b', value: { kind: 'call', op: 'MvNormal', kwargs: { mu: vec([ref('a'), ref('c')]), cov } } },
    ],
  };
  const point = { a: 0.5, b: { shape: [2], data: new Float64Array([0.1, 0.2]) } };
  const lp = density.logDensityN(ir, point, { a: aPrior, c }, N, {});
  for (let i = 0; i < N; i++) {
    const want = lnN(0.5, c[i]) + lnN(0.1, 0.5) + lnN(0.2, c[i]);
    assert.ok(Math.abs(lp[i] - want) < 1e-12, `atom ${i}: ${lp[i]} vs ${want}`);
  }
});
