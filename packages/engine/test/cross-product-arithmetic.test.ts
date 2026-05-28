'use strict';

// Cross-product matrix-arithmetic sweep: for every array producer that
// the engine offers, exercise the four basic arithmetic ops (* + - .^)
// against the result of another producer. This is the regression net
// for the bug class that was shipping silently before the §2.1 Value
// migration — `A * B` between two rowstack matrices evaluated to NaN
// because neither operand was a shape-explicit Value.
//
// The contract: producer × producer × op should NEVER yield NaN /
// undefined / a JS array of NaN. The result is a Value with shape
// matching the operation's broadcast / matmul rules.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');

function ev(src: string) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(lifted.bindings);
  return built.fixedValues;
}

function valuesOK(v: any, name: string): any {
  const valueLib = require('../value.ts');
  assert.ok(v != null, `${name}: produced null/undefined`);
  if (typeof v === 'number') {
    assert.ok(Number.isFinite(v) || Number.isNaN(v) === false,
      `${name}: produced NaN scalar`);
    return v;
  }
  if (valueLib.isValue(v)) {
    for (let i = 0; i < v.data.length; i++) {
      assert.ok(!Number.isNaN(v.data[i]),
        `${name}: NaN at index ${i} (shape=${JSON.stringify(v.shape)})`);
    }
    return v;
  }
  assert.fail(`${name}: not a scalar or Value (got ${typeof v})`);
}

// ---------------------------------------------------------------------
// Matrix × Matrix arithmetic
// ---------------------------------------------------------------------

test('cross: rowstack matrix * rowstack matrix → matmul Value', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
P = A * B
S = A + B
D = A - B
`);
  const P = valuesOK(fv.get('P'), 'P');
  assert.deepEqual(P.shape, [2, 2]);
  // Matrix product: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  assert.deepEqual(Array.from(P.data), [19, 22, 43, 50]);
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [6, 8, 10, 12]);
  const D = valuesOK(fv.get('D'), 'D');
  assert.deepEqual(Array.from(D.data), [-4, -4, -4, -4]);
});

test('cross: rand+iid([m,n]) matrix * rand+iid([m,n]) matrix', () => {
  const fv = ev(`
rstate = rnginit([1,2,3,4])
A, rs1 = rand(rstate, iid(Normal(0,1), [3, 3]))
B, _   = rand(rs1,    iid(Normal(0,1), [3, 3]))
P = A * B
S = A + B
D = A - B
`);
  for (const name of ['P', 'S', 'D']) {
    const m = valuesOK(fv.get(name), name);
    assert.deepEqual(m.shape, [3, 3]);
  }
});

test('cross: eye(n) * rowstack matrix preserves the matrix', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
I = eye(3)
R = I * A
`);
  const R = valuesOK(fv.get('R'), 'R');
  assert.deepEqual(R.shape, [3, 3]);
  assert.deepEqual(Array.from(R.data), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test('cross: diagmat * rowstack matrix scales rows', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
D = diagmat([10.0, 100.0])
R = D * A
`);
  const R = valuesOK(fv.get('R'), 'R');
  // diag([10, 100]) · A = [[10, 20], [300, 400]]
  assert.deepEqual(Array.from(R.data), [10, 20, 300, 400]);
});

test('cross: zeros([m,n]) * rowstack matrix is zero', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
Z = zeros([2, 2])
R = Z * A
`);
  const R = valuesOK(fv.get('R'), 'R');
  assert.deepEqual(Array.from(R.data), [0, 0, 0, 0]);
});

// ---------------------------------------------------------------------
// Vector × Vector arithmetic (rank-1)
// ---------------------------------------------------------------------

test('cross: array literal + linspace vector elementwise', () => {
  const fv = ev(`
v = [1.0, 2.0, 3.0]
w = linspace(0.0, 2.0, 3)
S = v + w
D = v - w
`);
  // linspace(0, 2, 3) = [0, 1, 2]
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [1, 3, 5]);
  const D = valuesOK(fv.get('D'), 'D');
  assert.deepEqual(Array.from(D.data), [1, 1, 1]);
});

test('cross: ones + zeros + fill', () => {
  const fv = ev(`
a = ones(4)
b = zeros(4)
c = fill(7.0, 4)
S = a + c
D = c - b
`);
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [8, 8, 8, 8]);
  const D = valuesOK(fv.get('D'), 'D');
  assert.deepEqual(Array.from(D.data), [7, 7, 7, 7]);
});

test('cross: onehot + linspace', () => {
  const fv = ev(`
o = onehot(2, 4)
l = linspace(0.0, 3.0, 4)
S = o + l
`);
  // onehot(2,4) = [0, 1, 0, 0]; linspace(0, 3, 4) = [0, 1, 2, 3]
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [0, 2, 2, 3]);
});

test('cross: cumsum and cumprod participate in arithmetic', () => {
  const fv = ev(`
v = [1.0, 2.0, 3.0, 4.0]
cs = cumsum(v)
cp = cumprod(v)
S = cs + cp
`);
  // cumsum([1,2,3,4]) = [1, 3, 6, 10]; cumprod = [1, 2, 6, 24]
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [2, 5, 12, 34]);
});

test('cross: cat(vectors) feeds arithmetic', () => {
  const fv = ev(`
a = [1.0, 2.0]
b = [3.0, 4.0]
ab = cat(a, b)
S = ab + ab
`);
  const S = valuesOK(fv.get('S'), 'S');
  assert.deepEqual(Array.from(S.data), [2, 4, 6, 8]);
});

test('cross: softmax and l1unit normalise consistently', () => {
  const fv = ev(`
logits = [1.0, 2.0, 3.0]
sm = softmax(logits)
sum_sm = sum(sm)
l1 = l1unit([1.0, 2.0, 3.0])
sum_l1 = sum(l1)
`);
  // sum of softmax + sum of l1unit are both 1
  assert.ok(Math.abs(fv.get('sum_sm') - 1) < 1e-12);
  assert.ok(Math.abs(fv.get('sum_l1') - 1) < 1e-12);
});

// ---------------------------------------------------------------------
// Dotted broadcast over various producers
// ---------------------------------------------------------------------

test('cross: dotted ^2 over linspace output', () => {
  const fv = ev(`
v = linspace(1.0, 4.0, 4)
sq = v .^ 2
`);
  const sq = valuesOK(fv.get('sq'), 'sq');
  assert.deepEqual(Array.from(sq.data), [1, 4, 9, 16]);
});

test('cross: dotted .* over rowstack outputs', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[2.0, 2.0], [2.0, 2.0]])
H = A .* B
`);
  // Hadamard: [[2,4],[6,8]]
  const H = valuesOK(fv.get('H'), 'H');
  assert.deepEqual(Array.from(H.data), [2, 4, 6, 8]);
});

// ---------------------------------------------------------------------
// Reductions over various producers (spec §07 domains)
// ---------------------------------------------------------------------

test('cross: sum / mean / prod over rowstack matrix entries', () => {
  // Spec §07: sum/mean/prod accept "real/complex arrays" — should
  // work for n-dim arrays, reducing over ALL entries to a scalar.
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
S = sum(A)
M = mean(A)
P = prod(A)
`);
  assert.equal(fv.get('S'), 10);
  assert.equal(fv.get('M'), 2.5);
  assert.equal(fv.get('P'), 24);
});

test('cross: F = sum((C - D) .^ 2) and G = mean((C - D) .^ 2)', () => {
  // The user's tmp.flatppl pattern — Frobenius-style squared error
  // from comparing matmul-via-aggregate to matmul-via-`*`.
  const fv = ev(`
rstate = rnginit([1,2,3,4])
A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
B, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
C[.i, .k] := A[.i, .j] * B[.j, .k]
D = A * B
F = sum((C - D) .^ 2)
G = mean((C - D) .^ 2)
`);
  // C and D should be equal (both are matmul(A, B)), so F = G = 0.
  assert.ok(Math.abs(fv.get('F')) < 1e-20,
    `F should be ≈ 0, got ${fv.get('F')}`);
  assert.ok(Math.abs(fv.get('G')) < 1e-20,
    `G should be ≈ 0, got ${fv.get('G')}`);
});

// ---------------------------------------------------------------------
// Vector × matrix dispatch
// ---------------------------------------------------------------------

test('cross: matrix * vector → vector (matvec dispatch)', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
v = [10.0, 20.0]
y = A * v
`);
  const y = valuesOK(fv.get('y'), 'y');
  // [[1, 2]; [3, 4]] · [10; 20] = [50; 110]
  assert.deepEqual(y.shape, [2]);
  assert.deepEqual(Array.from(y.data), [50, 110]);
});

test('cross: scalar * vector broadcasts', () => {
  const fv = ev(`
v = [1.0, 2.0, 3.0]
s = 5.0
r = s * v
`);
  const r = valuesOK(fv.get('r'), 'r');
  assert.deepEqual(Array.from(r.data), [5, 10, 15]);
});

// ---------------------------------------------------------------------
// Sanity: vector-of-vectors arithmetic should NOT silently work
// ---------------------------------------------------------------------

test('cross: nested literal [[a,b],[c,d]] is vector-of-vectors (NOT matrix)', () => {
  // Per spec §03: `[[1,2],[3,4]]` is a vector of vectors. It is NOT
  // implicitly a matrix. Engine-concepts §2.1 (C7 outerRank tag):
  // the runtime carries this distinction via the `outerRank` field
  // on Value — a vector-of-vectors has outerRank set + less than
  // shape.length (some axes are nested-collection outer axes). A
  // true matrix has no `outerRank` tag (or outerRank == shape.length)
  // — every axis is a loop axis at the matrix level. Either a JS
  // array of inner Values (legacy host form), null, or a tagged
  // nested-vector Value are all valid runtime forms; what's NOT
  // valid is auto-lifting to a flat matrix Value.
  const valueLib = require('../value.ts');
  const fv = ev(`
M = [[1.0, 2.0], [3.0, 4.0]]
`);
  const m = fv.get('M');
  const isFlatMatrix = valueLib.isValue(m) && m.shape.length === 2
    && !valueLib.isNestedVectorValue(m);
  assert.ok(!isFlatMatrix,
    'nested literal must NOT auto-lift to a flat matrix Value (spec §03); '
    + 'expected JS array OR tagged nested-vector Value (outerRank set), got '
    + JSON.stringify(m && { shape: m.shape, outerRank: m.outerRank }));
});
