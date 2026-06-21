const { test } = require('node:test');
const assert = require('node:assert/strict');
const T = require('../transforms.ts');

// Finite-difference check of d(constrain)/dy against logDetJ = log|d theta/dy|.
function fdLogDetJ(tr: any, y: number) {
  const e = 1e-6;
  const d = (tr.constrain(y + e) - tr.constrain(y - e)) / (2 * e);
  return Math.log(Math.abs(d));
}

test('positive transform: round-trips and LADJ matches finite difference', () => {
  const tr = T.transformFor({ kind: 'positive' });
  for (const theta of [0.01, 0.5, 1, 7.3, 250]) {
    const y = tr.unconstrain(theta);
    assert.ok(Math.abs(tr.constrain(y) - theta) < 1e-9, `roundtrip ${theta}`);
  }
  for (const y of [-3, -0.5, 0, 1.2, 5]) {
    assert.ok(Math.abs(tr.logDetJ(y) - fdLogDetJ(tr, y)) < 1e-5, `LADJ at y=${y}`);
  }
});

test('interval(2,5) transform: round-trips, stays in (2,5), LADJ matches finite difference', () => {
  const tr = T.transformFor({ kind: 'interval', a: 2, b: 5 });
  for (const theta of [2.001, 3, 4.5, 4.999]) {
    const y = tr.unconstrain(theta);
    assert.ok(tr.constrain(y) > 2 && tr.constrain(y) < 5);
    assert.ok(Math.abs(tr.constrain(y) - theta) < 1e-9, `roundtrip ${theta}`);
  }
  for (const y of [-4, -1, 0, 2, 6]) {
    assert.ok(Math.abs(tr.logDetJ(y) - fdLogDetJ(tr, y)) < 1e-5, `LADJ at y=${y}`);
  }
});

test('real transform: identity with zero log-Jacobian', () => {
  const tr = T.transformFor({ kind: 'real' });
  assert.equal(tr.unconstrain(3.5), 3.5);
  assert.equal(tr.constrain(3.5), 3.5);
  assert.equal(tr.logDetJ(3.5), 0);
});

test('simplexTransform(3): round-trips, sums to 1, LADJ matches finite difference', () => {
  const st = T.simplexTransform(3);            // 3-simplex <-> R^2
  const p = Float64Array.from([0.2, 0.5, 0.3]);
  const y = st.unconstrain(p);
  assert.equal(y.length, 2);
  const back = st.constrain(y);
  let s = 0; for (let i = 0; i < 3; i++) { assert.ok(Math.abs(back[i] - p[i]) < 1e-9); s += back[i]; }
  assert.ok(Math.abs(s - 1) < 1e-12, 'sums to 1');

  // FD of the full Jacobian determinant of constrain: R^2 -> first 2 free coords.
  const e = 1e-6;
  const J = [[0, 0], [0, 0]];
  for (let j = 0; j < 2; j++) {
    const yp = Float64Array.from(y); yp[j] += e;
    const ym = Float64Array.from(y); ym[j] -= e;
    const cp = st.constrain(yp), cm = st.constrain(ym);
    for (let i = 0; i < 2; i++) J[i][j] = (cp[i] - cm[i]) / (2 * e);
  }
  const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  assert.ok(Math.abs(st.logDetJ(y) - Math.log(Math.abs(det))) < 1e-4, 'simplex LADJ vs FD');
});

test('supportOf maps distributions to their support', () => {
  assert.deepEqual(T.supportOf('Normal'), { kind: 'real' });
  assert.deepEqual(T.supportOf('Exponential'), { kind: 'positive' });
  assert.deepEqual(T.supportOf('HalfCauchy'), { kind: 'positive' });
  assert.deepEqual(T.supportOf('Gamma'), { kind: 'positive' });
  assert.deepEqual(T.supportOf('Beta'), { kind: 'interval', a: 0, b: 1 });
});

test('supportOf throws for an unsupported (e.g. discrete or unknown) dist', () => {
  assert.throws(() => T.supportOf('Poisson'), /no continuous support/);
});

test('supportOf(Uniform) uses provided bounds and refuses to default to [0,1]', () => {
  // With finite bounds it returns the interval as given.
  assert.deepEqual(T.supportOf('Uniform', { a: 0.1, b: 20 }), { kind: 'interval', a: 0.1, b: 20 });
  // Without bounds it must NOT silently return [0,1] — that default constrained
  // every bounded-Uniform latent into (0,1) and capped scale parameters at 1.
  assert.throws(() => T.supportOf('Uniform', {}), /support bounds unavailable/);
  assert.throws(() => T.supportOf('Uniform'), /support bounds unavailable/);
});
