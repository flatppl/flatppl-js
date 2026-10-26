'use strict';

// Three §07 builtins that type-checked but did not evaluate.
//
// qr(A): "QR decomposition (unpivoted) A = QR; for m × n A with m >= n, Q is
// m × n with orthonormal columns and R is n × n upper-triangular; returns
// record(Q, R)". §07 states no sign convention, so this follows numpy (LAPACK
// Householder): R[k, k] = -sign(A[k, k]) · norm, and no reflector, so no sign
// flip, where a column has nothing below the diagonal.
//
// ORACLE: numpy 2.5.1 `np.linalg.qr(A, mode="reduced")`, run through the python
// MCP. No value comes from the engine.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function errors(src: string): string[] {
  return (processSource(src).diagnostics || []).filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function fixed(src: string, name: string): any {
  assert.deepEqual(errors(src), [], src);
  return makeMatCtx(src).built.fixedValues.get(name);
}

const lit = (M: number[][]) => 'rowstack(['
  + M.map((r) => '[' + r.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(', ') + ']')
    .join(', ') + '])';
const rows = (V: any): number[][] => {
  const [m, n] = V.shape;
  return Array.from({ length: m }, (_, i) => Array.from(V.data.slice(i * n, i * n + n)) as number[]);
};

const NUMPY: Record<string, { A: number[][], Q: number[][], R: number[][] }> = {
  sq2: {
    A: [[3.0, 0.0], [4.0, 5.0]],
    Q: [[-0.6000000000000001, -0.8], [-0.8, 0.6]],
    R: [[-5.0, -4.0], [0.0, 3.0]],
  },
  sq3: {
    A: [[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [-4.0, 0.25, 1.0]],
    Q: [[-0.4364357804719847, 0.31524416249564025, 0.8427009716003844],
      [-0.21821789023599236, -0.9457324874869207, 0.24077170617153842],
      [0.8728715609439694, -0.07881104062391003, 0.48154341234307685]],
    R: [[-4.58257569495584, 6.918226771341065e-17, 1.0910894511799618],
      [0.0, -3.1721443851123796, 1.970276015597752], [0.0, 0.0, 0.42135048580019224]],
  },
  tall32: {
    A: [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]],
    Q: [[-0.16903085094570325, 0.9486832980505133], [-0.50709255283711, -0.31622776601683833],
      [-0.8451542547285166, 2.3208668709370606e-16]],
    R: [[-5.916079783099615, -8.282511696339462], [0.0, 0.6324555320336759]],
  },
  // A zero leading entry: numpy's reflector still gives R[0, 0] = -norm.
  tall42: {
    A: [[0.0, 1.0], [2.0, -1.0], [-3.0, 0.5], [1.0, 2.0]],
    Q: [[0.0, 0.4052442151348903], [-0.5345224838248488, -0.3184061690345566],
      [0.8017837257372732, 0.07236503841694475], [-0.2672612419124244, 0.8539074533199476]],
    R: [[-3.741657386773941, 0.40089186286863665], [0.0, 2.467647810017814]],
  },
  // Already upper-triangular: no reflector, so R = A and Q = I.
  upper: {
    A: [[-2.0, 1.0], [0.0, 3.0]],
    Q: [[1.0, 0.0], [-0.0, 1.0]],
    R: [[-2.0, 1.0], [0.0, 3.0]],
  },
  negdiag: {
    A: [[-1.0, 2.0], [-2.0, 1.0], [2.0, 0.0]],
    Q: [[-0.33333333333333326, 0.866578244826242], [-0.6666666666666666, 0.06189844605901729],
      [0.6666666666666666, 0.4951875684721383]],
    R: [[3.0, -1.333333333333333], [0.0, 1.795054935711501]],
  },
};

for (const [name, { A, Q, R }] of Object.entries(NUMPY)) {
  test(`qr matches numpy and its §07 contract: ${name}`, () => {
    const src = `A = ${lit(A)}\nf = qr(A)\nQ = f.Q\nR = f.R`;
    const q = rows(fixed(src, 'Q')), r = rows(fixed(src, 'R'));
    const m = A.length, n = A[0].length;
    assert.deepEqual([q.length, q[0].length, r.length, r[0].length], [m, n, n, n]);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      assert.ok(Math.abs(q[i][j] - Q[i][j]) < 1e-12, `Q[${i},${j}] ${q[i][j]} vs ${Q[i][j]}`);
      let a = 0;
      for (let k = 0; k < n; k++) a += q[i][k] * r[k][j];
      assert.ok(Math.abs(a - A[i][j]) < 1e-12, `(QR)[${i},${j}] ${a} vs ${A[i][j]}`);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      assert.ok(Math.abs(r[i][j] - R[i][j]) < 1e-12, `R[${i},${j}] ${r[i][j]} vs ${R[i][j]}`);
      if (i > j) assert.equal(r[i][j], 0, 'R is upper-triangular');
      let g = 0;
      for (let k = 0; k < m; k++) g += q[k][i] * q[k][j];
      assert.ok(Math.abs(g - (i === j ? 1 : 0)) < 1e-12, `QᵀQ[${i},${j}] = ${g}`);
    }
  });
}

test('qr takes its spec input name, and a field of the call reads directly', () => {
  const x = fixed('x = qr(A = rowstack([[3.0, 0.0], [4.0, 5.0]])).R', 'x');
  assert.deepEqual(Array.from(x.data), [-5, -4, 0, 3]);
});

// §07 "Scalar restrictions and constructors": `string(x)`, identity on strings.
test('string is the identity on strings and refuses a number statically', () => {
  assert.equal(fixed('x = string("a")', 'x'), 'a');
  assert.equal(fixed('x = string(x = "b")', 'x'), 'b');
  assert.deepEqual(errors('x = string(2)'), ['string: arg 1 expects string, got integer']);
});

// §07 "Membership": `x in S`. The call spelling `in(x, S)` must give the same
// answer as the operator in every case, and bind its inputs by name.
test('in(x, S) agrees with x in S', () => {
  const cases: [string, string][] = [
    ['0.5', 'unitinterval'], ['1.5', 'unitinterval'], ['2', 'interval(0.0, 1.0)'],
    ['-1.0', 'posreals'], ['3', 'posintegers'], ['[0.2, 0.8]', 'stdsimplex(2)'],
    ['[0.2, 0.9]', 'stdsimplex(2)'], ['"a"', 'unitinterval'],
  ];
  for (const [x, S] of cases) {
    const call = fixed(`y = in(${x}, ${S})`, 'y');
    assert.equal(typeof call, 'boolean', `in(${x}, ${S})`);
    assert.equal(call, fixed(`y = ${x} in ${S}`, 'y'), `in(${x}, ${S})`);
  }
  assert.equal(fixed('y = in(S = unitinterval, x = 0.5)', 'y'), true);
  assert.equal(fixed('y = in(S = unitinterval, x = 1.5)', 'y'), false);
  assert.equal(fixed('y = in(record(x = 0.5, S = unitinterval))', 'y'), true);
});
