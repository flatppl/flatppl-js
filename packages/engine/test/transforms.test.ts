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

// log|det| of the finite-difference Jacobian of `constrain`, restricted to the
// first `m` output coordinates. Gaussian elimination with partial pivoting.
function fdLogAbsDet(tr: any, y: Float64Array, m: number): number {
  const e = 1e-6;
  const J: number[][] = Array.from({ length: m }, () => new Array(y.length).fill(0));
  for (let j = 0; j < y.length; j++) {
    const yp = Float64Array.from(y); yp[j] += e;
    const ym = Float64Array.from(y); ym[j] -= e;
    const cp = tr.constrain(yp), cm = tr.constrain(ym);
    for (let i = 0; i < m; i++) J[i][j] = (cp[i] - cm[i]) / (2 * e);
  }
  let acc = 0;
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(J[r][c]) > Math.abs(J[piv][c])) piv = r;
    [J[c], J[piv]] = [J[piv], J[c]];
    acc += Math.log(Math.abs(J[c][c]));
    for (let r = c + 1; r < m; r++) {
      const f = J[r][c] / J[c][c];
      for (let k = c; k < m; k++) J[r][k] -= f * J[c][k];
    }
  }
  return acc;
}

test('orderedTransform(4): images are strictly increasing, round-trip, closed-form LADJ', () => {
  const ot = T.orderedTransform(4);
  const y = Float64Array.from([-0.3, 0.1, -1.2, 0.7]);
  const x = ot.constrain(y);
  for (let k = 1; k < 4; k++) assert.ok(x[k] > x[k - 1], `x[${k}] > x[${k - 1}]`);
  const back = ot.unconstrain(x);
  for (let k = 0; k < 4; k++) assert.ok(Math.abs(back[k] - y[k]) < 1e-12);
  // By hand: log|det J| = y2 + y3 + y4 = 0.1 - 1.2 + 0.7.
  assert.ok(Math.abs(ot.logDetJ(y) - (-0.4)) < 1e-12);
  assert.ok(Math.abs(fdLogAbsDet(ot, y, 4) - (-0.4)) < 1e-6, 'ordered LADJ vs FD');
});

test('orderedTransform(2): Lebesgue mass of ordered(2) inside [0,1]^2 is 1/2', () => {
  // By hand: {0 <= x1 < x2 <= 1} is half the unit square. Integrate the indicator
  // in the chart, weighted by exp(logDetJ), on a midpoint grid; y2 below -30
  // contributes under e^-30.
  const ot = T.orderedTransform(2);
  const n1 = 500, n2 = 30000, lo2 = -30;
  const d1 = 1 / n1, d2 = -lo2 / n2;
  const y = new Float64Array(2);
  let mass = 0;
  for (let i = 0; i < n1; i++) {
    y[0] = (i + 0.5) * d1;
    for (let j = 0; j < n2; j++) {
      y[1] = lo2 + (j + 0.5) * d2;
      const x = ot.constrain(y);
      if (x[1] <= 1) mass += Math.exp(ot.logDetJ(y)) * d1 * d2;
    }
  }
  assert.ok(Math.abs(mass - 0.5) < 2e-3, `mass ${mass}`);
});

test('zerosumTransform(3): images sum to zero, round-trip, zero LADJ on the free coordinates', () => {
  const zt = T.zerosumTransform(3);
  const y = Float64Array.from([0.5, 0.25]);
  const x = zt.constrain(y);
  assert.deepEqual(Array.from(x), [0.5, 0.25, -0.75]);
  assert.deepEqual(Array.from(zt.unconstrain(x)), [0.5, 0.25]);
  assert.equal(zt.logDetJ(y), 0);
  // §03: the reference measure is the coordinate measure of x1, x2.
  assert.ok(Math.abs(fdLogAbsDet(zt, y, 2)) < 1e-8, 'zerosum LADJ vs FD');
});

test('zerosumTransform(3): Lebesgue mass of zerosum(3) inside [-1,1]^3 is 3', () => {
  // By hand, in the chart (x1, x2) with x3 = -x1 - x2: the square [-1,1]^2 minus
  // the corners x1 + x2 > 1 and x1 + x2 < -1, each of area 1/2, leaves 3. The
  // surface measure would give 3*sqrt(3) instead.
  const zt = T.zerosumTransform(3);
  const n = 2000, d = 2 / n;
  const y = new Float64Array(2);
  let mass = 0;
  for (let i = 0; i < n; i++) {
    y[0] = -1 + (i + 0.5) * d;
    for (let j = 0; j < n; j++) {
      y[1] = -1 + (j + 0.5) * d;
      const x = zt.constrain(y);
      if (x.every((v: number) => Math.abs(v) <= 1)) mass += Math.exp(zt.logDetJ(y)) * d * d;
    }
  }
  assert.ok(Math.abs(mass - 3) < 5e-3, `mass ${mass}`);
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
