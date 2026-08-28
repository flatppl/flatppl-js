'use strict';

// `broadcastN` over a SHAPED atom batch — shape=[N, ...cell], the form an
// `iid(<scalar dist>, D)` variate takes and the value a vector-atom
// `pushfwd` body is applied to.
//
// Before the fix `broadcastN`'s `isBatch` (`valueLib.isAtomBatchedScalar`)
// answered false for a [N, D] Value, so every scalar primitive fell through
// to `_scalarVal`, which returns a rank ≥ 1 Value unchanged — the primitive
// received the Value object itself and produced NaN or a shape-[] result the
// callers rejected. `mulN` and `addN`/`subN` had the matching gap on the
// shape-aware side: a rank-0 operand has no axis to align against the atom
// axis, and both refused the combination outright.
//
// Spec anchors:
//   - §07 "Operator-equivalent functions": `mul`'s domains include
//     **scalar-vector** and **scalar-matrix**, so `2.0 * x` over a vector
//     variate is required, not optional.
//   - §04 "Broadcasting": elementwise application of a scalar function over
//     an array is `broadcast(exp, A)` / `exp.(A)`. Both spellings, and the
//     engine's bare-`exp`-over-an-array behaviour, land on this path.
//
// Oracle (INDEPENDENT — closed form, cross-checked against
// Distributions.jl; NOT the other engine):
//   exp of a standard normal is LogNormal(0, 1):
//     mean   = exp(1/2)        = 1.6487212707001282
//     var    = (e - 1)·e       = 4.670774270471604
//     median = 1
//   2 · a standard normal is Normal(0, 2): var = 4, cross-covariance 0.
//
// The moment checks are statistical, so they carry a second, EXACT leg: the
// pushforward is deterministic, so every output cell must equal the scalar
// function applied to the corresponding base cell bit-for-bit. That is what
// catches a wrong stride or a dropped atom axis — a rank-0 read produced
// NaN past the first N entries, which a mean/variance test can also pass
// through, but an exact cell comparison cannot.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const valueLib = require('../value.ts');
const { broadcastN, _packUniformCells, _perAtomFallback } = require('../sampler-eval-batched.ts');
const ROOT_SEED = 0x10C5CA1E;

const LOGNORMAL_MEAN = 1.6487212707001282;
const LOGNORMAL_VAR = 4.670774270471604;

function makeCtx(source: any, opts?: any) {
  opts = opts || {};
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((e: any) => e.message), [],
    'unexpected diagnostics');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx); cache.set(name, p); return p;
    },
    sendWorker: (msg: any) => {
      const r = worker.handle(msg);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: opts.sampleCount != null ? opts.sampleCount : 256,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

// Materialise the base and the pushforward from ONE ctx so the cached base
// measure is exactly the variate the body consumed.
async function baseAndImage(body: string, D: number, N: number) {
  const ctx = makeCtx(
    `Z = iid(Normal(0.0, 1.0), ${D})\n`
    + `X = pushfwd(x -> ${body}, Z)\n`, { sampleCount: N });
  const Z = await ctx.getMeasure('Z');
  const X = await ctx.getMeasure('X');
  return { Z, X };
}

// Per-component mean / variance of an atom-major [N, D] batch.
function moments(value: any, d: number) {
  const N = value.shape[0], D = value.shape[1], data = value.data;
  let m = 0;
  for (let i = 0; i < N; i++) m += data[i * D + d];
  m /= N;
  let v = 0;
  for (let i = 0; i < N; i++) {
    const e = data[i * D + d] - m;
    v += e * e;
  }
  return { mean: m, var: v / N };
}

// ---------------------------------------------------------------------
// Exact leg — every cell equals the scalar function of the base cell
// ---------------------------------------------------------------------

// The elementary functions carry §07's scalar-only signature, so over a vector
// variate they appear dotted; `*`, `/`, `+`, `-` and unary `-` are the
// operator-equivalent rows §07 gives array domains.
const EXACT: [string, (z: number) => number][] = [
  ['exp.(x)', (z) => Math.exp(z)],
  ['abs.(x)', (z) => Math.abs(z)],
  ['tanh.(x)', (z) => Math.tanh(z)],
  ['2.0 * x', (z) => 2.0 * z],
  ['x * 2.0', (z) => z * 2.0],
  ['x / 2.0', (z) => z / 2.0],
  ['x + 1.0', (z) => z + 1.0],
  ['x - 1.0', (z) => z - 1.0],
  ['1.0 - x', (z) => 1.0 - z],
  ['-x', (z) => -z],
  ['atan2.(x, 1.0)', (z) => Math.atan2(z, 1.0)],
];

for (const [body, f] of EXACT) {
  test(`elementwise \`${body}\` over a [N,2] variate maps cell-wise`, async () => {
    const N = 64;
    const { Z, X } = await baseAndImage(body, 2, N);
    assert.deepEqual(Z.value.shape, [N, 2], 'base is a shaped atom batch');
    assert.deepEqual(X.value.shape, [N, 2],
      'the image keeps the atom axis and the cell axis');
    for (let i = 0; i < N * 2; i++) {
      assert.equal(X.value.data[i], f(Z.value.data[i]),
        `cell ${i}: got ${X.value.data[i]}, expected `
        + `${f(Z.value.data[i])} from base ${Z.value.data[i]}`);
    }
  });
}

// ---------------------------------------------------------------------
// The DOTTED spelling over a rank-≥2 cell
// ---------------------------------------------------------------------
//
// `exp.(x)` lowers to `broadcast(exp, x)` (§04 "Broadcasting": "`f.(<args>)`
// lowers to `broadcast(f, <args>)`"), which "maps a function or kernel
// elementwise over arrays". §07 "Elementary functions" makes `exp` scalar-only
// ("All accept scalar arguments and return scalar results"), so the dotted
// spelling is the only one the spec licenses over a matrix variate.
//
// `broadcast` is a higher-order op with no batched slot, so it takes
// `_perAtomFallback`: N per-atom `evaluateExpr` calls, each returning one
// rank-2 Value. Packing those back into a batch only handled a rank-1 cell,
// so a [3, 2] cell fell through to the raw JS array and `evaluateN` refused it
// ("expression produced non-scalar per-atom result (got object)").
//
// The two rank-≥2 variates part company at inference, which is where §04's ONE
// broadcast level bites. `iid(Normal, [3, 2])` is an n-D array of scalars, so a
// single dot reaches scalars and types clean. `iid(iid(M, 2), 3)` is a vector
// OF vectors (§03 keeps the two distinct), so the same single dot hands `exp` a
// 2-vector and the scalar-only rule refuses it — the admissible spelling nests
// one dot per array level, and that is not on the batched evaluator. The engine
// represents both as one flat [N,3,2] Value, so only inference tells them apart.

const RANK2_MULTI_AXIS = 'iid(Normal(0.0, 1.0), [3, 2])';
const RANK2_NESTED = 'iid(iid(Normal(0.0, 1.0), 2), 3)';

async function rank2Image(base: string, body: string, N: number) {
  const ctx = makeCtx(`Z = ${base}\nX = pushfwd(x -> ${body}, Z)\n`,
    { sampleCount: N });
  return { Z: await ctx.getMeasure('Z'), X: await ctx.getMeasure('X') };
}

test('dotted `exp` maps cell-for-cell over a multi-axis iid variate', async () => {
  const N = 32;
  const { Z, X } = await rank2Image(RANK2_MULTI_AXIS, 'exp.(x)', N);
  assert.deepEqual(Z.value.shape, [N, 3, 2], 'base cell is 3×2');
  assert.deepEqual(X.value.shape, [N, 3, 2],
    'the dotted image keeps the atom axis and both cell axes');
  for (let i = 0; i < N * 6; i++) {
    assert.equal(X.value.data[i], Math.exp(Z.value.data[i]),
      `cell ${i}: got ${X.value.data[i]}, expected `
      + `${Math.exp(Z.value.data[i])} from base ${Z.value.data[i]}`);
  }
});

// The bare spelling is a static error on both, so it is no longer available as
// a cross-check on the dotted result. §07's refusal is the pin instead.
for (const [label, base, got] of [
  ['a multi-axis iid', RANK2_MULTI_AXIS, '2d array of real'],
  ['an iid-of-iid', RANK2_NESTED, 'array of array of real'],
] as [string, string, string][]) {
  test(`bare \`exp\` over ${label} variate is refused by §07`, async () => {
    await assert.rejects(
      () => rank2Image(base, 'exp(x)', 4),
      (e: any) => e.message.includes('exp: arg 1 expects a scalar, got ' + got)
        && e.message.includes('All accept scalar arguments and return scalar results'));
  });
}

test('dotted `exp` over an iid-of-iid variate is refused: one broadcast level',
  async () => {
    // §04 maps `broadcast` over the array's ELEMENTS, and this variate's
    // elements are 2-vectors, so a single dot is still a scalar-only violation.
    await assert.rejects(
      () => rank2Image(RANK2_NESTED, 'exp.(x)', 4),
      (e: any) => e.message.includes(
        'exp: arg 1 expects a scalar, got array of real (length 2)'));
  });

test('dotted `exp` over a matrix variate is a LogNormal(0,1) in every cell',
  async () => {
    const N = 20000;
    const ctx = makeCtx(
      'Z = iid(Normal(0.0, 1.0), [3, 2])\n'
      + 'X = pushfwd(x -> exp.(x), Z)\n', { sampleCount: N });
    const X = await ctx.getMeasure('X');
    assert.deepEqual(X.value.shape, [N, 3, 2]);
    // A LogNormal's sample variance is a heavy-tailed estimator, so the
    // tolerance comes from its own standard error, not a round number.
    // Kurtosis of LogNormal(0, 1) is e⁴ + 2e³ + 3e² − 3 = 113.9363..., so
    // sd(s²)/σ² = sqrt((kurt − 1)/N) = 7.51% at N = 20000. Four of those.
    const KURTOSIS = Math.exp(4) + 2 * Math.exp(3) + 3 * Math.exp(2) - 3;
    const VAR_TOL = 4 * Math.sqrt((KURTOSIS - 1) / N);
    const CELLS = 6, data = X.value.data;
    for (let c = 0; c < CELLS; c++) {
      let m = 0;
      for (let i = 0; i < N; i++) m += data[i * CELLS + c];
      m /= N;
      let v = 0;
      for (let i = 0; i < N; i++) {
        const e = data[i * CELLS + c] - m;
        v += e * e;
      }
      v /= N;
      assert.ok(Math.abs(m - LOGNORMAL_MEAN) < 0.06,
        `cell ${c} mean ${m} vs LogNormal mean ${LOGNORMAL_MEAN}`);
      assert.ok(Math.abs(v - LOGNORMAL_VAR) / LOGNORMAL_VAR < VAR_TOL,
        `cell ${c} variance ${v} vs LogNormal variance ${LOGNORMAL_VAR}`);
      for (let i = 0; i < N; i++) {
        assert.ok(data[i * CELLS + c] > 0,
          `atom ${i} cell ${c} is ${data[i * CELLS + c]}, not in the support`);
      }
    }
  });

// ---------------------------------------------------------------------
// _packUniformCells unit level — which per-atom results pack, and which
// stay a raw JS array for the caller to surface
// ---------------------------------------------------------------------

const cell = (flat: number[], shape: number[]) =>
  valueLib.withShape(Float64Array.from(flat), shape);

test('_packUniformCells: rank-2 cells pack atom-major into [N, r, c]', () => {
  const out = _packUniformCells(
    [cell([1, 2, 3, 4, 5, 6], [3, 2]), cell([7, 8, 9, 10, 11, 12], [3, 2])], 2);
  assert.deepEqual(out.shape, [2, 3, 2]);
  assert.deepEqual(Array.from(out.data), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
});

test('_packUniformCells: rank-1 cells and bare arrays still pack to [N, k]', () => {
  const values = _packUniformCells([cell([1, 2], [2]), cell([3, 4], [2])], 2);
  assert.deepEqual(values.shape, [2, 2]);
  assert.deepEqual(Array.from(values.data), [1, 2, 3, 4]);
  const bare = _packUniformCells([[1, 2], Float64Array.from([3, 4])], 2);
  assert.deepEqual(bare.shape, [2, 2]);
  assert.deepEqual(Array.from(bare.data), [1, 2, 3, 4]);
});

test('_packUniformCells: a cell shape that varies across atoms does not pack',
  () => {
    // Differing rank, then differing size at equal rank, then a first
    // result that is no cell at all. Each must stay unpacked rather than
    // stride one atom's cell over the batch.
    assert.equal(
      _packUniformCells([cell([1, 2], [2]), cell([3, 4], [2, 1])], 2), null,
      'rank mismatch');
    assert.equal(
      _packUniformCells([cell([1, 2, 3, 4], [2, 2]), cell([5, 6], [1, 2])], 2),
      null, 'size mismatch at equal rank');
    assert.equal(
      _packUniformCells([{ a: 1 }, cell([1, 2], [2])], 2), null,
      'first result is not a cell');
  });

test('_packUniformCells: non-numeric, empty and complex cells do not pack',
  () => {
    assert.equal(_packUniformCells([['a', 'b'], ['c', 'd']], 2), null,
      'non-numeric entries');
    assert.equal(_packUniformCells([cell([], [0]), cell([], [0])], 2), null,
      'empty cell');
    // A complex Value keeps its imaginary part in `.im`, so packing `.data`
    // alone would drop it silently.
    const cx = valueLib.complexValue(
      Float64Array.from([1, 2]), Float64Array.from([3, 4]), [2]);
    assert.equal(_packUniformCells([cx, cx], 2), null, 'complex cell');
  });

test('_packUniformCells: boolean cell entries pack as 0/1', () => {
  const out = _packUniformCells([[true, false], [false, true]], 2);
  assert.deepEqual(out.shape, [2, 2]);
  assert.deepEqual(Array.from(out.data), [1, 0, 0, 1]);
});

test('_perAtomFallback: per-atom records stay a raw JS array', () => {
  // A record result is no cell, so the batch keeps the per-atom array and
  // the caller decides how to surface it — `evaluateN` refuses it, which is
  // the pre-existing contract for a non-scalar per-atom result.
  const ir = {
    kind: 'call', op: 'record',
    fields: [{ name: 'a', value: { kind: 'ref', ns: 'self', name: 'x' } }],
  };
  const refArrays = { x: valueLib.batchedScalar(Float64Array.from([1, 2])) };
  const out = _perAtomFallback(ir, refArrays, 2, {}, null);
  assert.ok(Array.isArray(out), 'unpackable results stay per-atom');
  assert.deepEqual(out.map((r: any) => r.a), [1, 2]);
});

// ---------------------------------------------------------------------
// Oracle leg — the pushed-forward law matches closed form
// ---------------------------------------------------------------------

test('exp over a vector variate is a LogNormal(0,1) per component', async () => {
  const N = 20000;
  const { X } = await baseAndImage('exp.(x)', 2, N);
  for (const d of [0, 1]) {
    const { mean, var: v } = moments(X.value, d);
    assert.ok(Math.abs(mean - LOGNORMAL_MEAN) < 0.06,
      `component ${d} mean ${mean} vs LogNormal mean ${LOGNORMAL_MEAN}`);
    assert.ok(Math.abs(v - LOGNORMAL_VAR) / LOGNORMAL_VAR < 0.15,
      `component ${d} variance ${v} vs LogNormal variance ${LOGNORMAL_VAR}`);
  }
  // Every atom is strictly positive — the support of a LogNormal, and the
  // cheapest witness that no cell went through as NaN.
  for (let i = 0; i < X.value.data.length; i++) {
    assert.ok(X.value.data[i] > 0,
      `cell ${i} is ${X.value.data[i]}, not in the LogNormal support`);
  }
});

test('a scalar scale over a vector variate is Normal(0,2) per component', async () => {
  const N = 20000;
  const { X } = await baseAndImage('2.0 * x', 2, N);
  for (const d of [0, 1]) {
    const { mean, var: v } = moments(X.value, d);
    assert.ok(Math.abs(mean) < 0.06, `component ${d} mean ${mean} vs 0`);
    assert.ok(Math.abs(v - 4) < 0.2, `component ${d} variance ${v} vs 4`);
  }
  // The components stay independent: a stride bug that reused one cell
  // across the atom would show up as a correlation, and matches the
  // per-component moments otherwise.
  const D = 2, data = X.value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += data[i * D]; m1 += data[i * D + 1]; }
  m0 /= N; m1 /= N;
  let c01 = 0;
  for (let i = 0; i < N; i++) {
    c01 += (data[i * D] - m0) * (data[i * D + 1] - m1);
  }
  assert.ok(Math.abs(c01 / N) < 0.15,
    `cross-covariance ${c01 / N} vs 0`);
});

// ---------------------------------------------------------------------
// Batched / atom-indep agreement
// ---------------------------------------------------------------------

// `x / 2.0` is here now that typeinfer admits §07's "array-scalar" `divide`
// domain on a literal vector; it used to have no atom-indep counterpart to
// compare against because `v / 2.0` was a static error.
test('the batched path agrees with the atom-indep path on a literal vector', async () => {
  for (const [body, indepBody] of [
    ['exp.(x)', 'exp.(v)'], ['2.0 * x', '2.0 * v'], ['x + 1.0', 'v + 1.0'],
    ['-x', '-v'], ['x * 2.0', 'v * 2.0'], ['x / 2.0', 'v / 2.0'],
  ]) {
    const indep = await makeCtx(
      `v = [0.25, -1.5]\nw = ${indepBody}\n`, { sampleCount: 1 },
    ).getMeasure('w');
    const ctx = makeCtx(
      `Z = iid(Normal(0.0, 1.0), 2)\nX = pushfwd(x -> ${body}, Z)\n`,
      { sampleCount: 4 });
    const Z = await ctx.getMeasure('Z');
    const X = await ctx.getMeasure('X');
    // Rebuild the atom-indep answer for atom 0's actual cell rather than
    // asserting equal numbers — the point is that the same body over the
    // same two reals gives the same two reals on both paths.
    const literal = await makeCtx(
      `v = [${Z.value.data[0]}, ${Z.value.data[1]}]\nw = ${indepBody}\n`,
      { sampleCount: 1 }).getMeasure('w');
    assert.deepEqual(indep.value.shape, [2], `${indepBody} stays rank-1`);
    for (let i = 0; i < 2; i++) {
      assert.equal(X.value.data[i], literal.value.data[i],
        `${body} cell ${i}: batched ${X.value.data[i]} vs atom-indep `
        + `${literal.value.data[i]}`);
    }
  }
});

// ---------------------------------------------------------------------
// broadcastN unit level — operand kinds and the mismatch refusal
// ---------------------------------------------------------------------

const N4 = 4;

function cellValue(flat: number[], shape: number[]) {
  const v: any = valueLib.withShape(Float64Array.from(flat), shape);
  v.outerRank = 1;
  return v;
}

test('broadcastN: cell × cell reads both per cell entry', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const b = cellValue([2, 2, 3, 3, 4, 4, 5, 5], [N4, 2]);
  const out = broadcastN((x: number, y: number) => x * y, [a, b], N4);
  assert.deepEqual(out.shape, [N4, 2]);
  assert.deepEqual(Array.from(out.data), [2, 4, 9, 12, 20, 24, 35, 40]);
});

test('broadcastN: cell × atom-batched scalar holds the scalar across the cell', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const s = valueLib.batchedScalar(Float64Array.from([10, 20, 30, 40]));
  const out = broadcastN((x: number, y: number) => x + y, [a, s], N4);
  assert.deepEqual(out.shape, [N4, 2]);
  assert.deepEqual(Array.from(out.data), [11, 12, 23, 24, 35, 36, 47, 48]);
});

test('broadcastN: cell × rank-0 Value and cell × bare number agree', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const wrapped = broadcastN((x: number, y: number) => x - y,
    [a, valueLib.scalar(1)], N4);
  const bare = broadcastN((x: number, y: number) => x - y, [a, 1], N4);
  assert.deepEqual(Array.from(wrapped.data), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(Array.from(bare.data), Array.from(wrapped.data));
  assert.deepEqual(bare.shape, [N4, 2]);
});

test('broadcastN: arity-3 over a cell, with a batched-scalar condition', () => {
  const t = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const e = cellValue([-1, -2, -3, -4, -5, -6, -7, -8], [N4, 2]);
  const cond = valueLib.batchedScalar(Float64Array.from([1, 0, 1, 0]));
  const out = broadcastN((c: number, x: number, y: number) => (c ? x : y),
    [cond, t, e], N4);
  assert.deepEqual(Array.from(out.data), [1, 2, -3, -4, 5, 6, -7, -8]);
});

test('broadcastN: arity > 3 over a cell takes the generic loop', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const out = broadcastN(
    (w: number, x: number, y: number, z: number) => w + x + y + z,
    [a, 1, 2, 3], N4);
  assert.deepEqual(Array.from(out.data), [7, 8, 9, 10, 11, 12, 13, 14]);
});

test('broadcastN: a rank-3 cell keeps all three axes', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
  const out = broadcastN((x: number) => x * x, [a], 2);
  assert.deepEqual(out.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(out.data), [1, 4, 9, 16, 25, 36, 49, 64]);
});

test('broadcastN: mismatched cell shapes are refused, not silently strided', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8], [N4, 2]);
  const b = cellValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [N4, 3]);
  assert.throws(() => broadcastN((x: number, y: number) => x + y, [a, b], N4),
    /broadcastN: per-atom cell shape \[3\] does not match \[2\] with N=4/);
});

// Equal-COUNT, different-axes cells (B2): `_cellLen` is a product, so
// [2,3] and [3,2] both give 6 — an equal-count check would pass this
// through and silently pair a 2×3 with a 3×2 flat-index-wise. Both
// argument orders must refuse.
test('broadcastN: same cell element count but different axes is refused', () => {
  const a = cellValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
  const b = cellValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
  assert.throws(() => broadcastN((x: number, y: number) => x + y, [a, b], 2),
    /broadcastN: per-atom cell shape \[3,2\] does not match \[2,3\] with N=2/);
  assert.throws(() => broadcastN((x: number, y: number) => x + y, [b, a], 2),
    /broadcastN: per-atom cell shape \[2,3\] does not match \[3,2\] with N=2/);
});

// Equal count, different RANK: [N,6] vs [N,2,3] also share cellLen=6.
test('broadcastN: same cell element count but different rank is refused', () => {
  const flat = cellValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 6]);
  const nested = cellValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
  assert.throws(() => broadcastN((x: number, y: number) => x + y, [flat, nested], 2),
    /broadcastN: per-atom cell shape \[2,3\] does not match \[6\] with N=2/);
  assert.throws(() => broadcastN((x: number, y: number) => x + y, [nested, flat], 2),
    /broadcastN: per-atom cell shape \[6\] does not match \[2,3\] with N=2/);
});

test('broadcastN: a shape=[N] batch is NOT treated as a cell batch', () => {
  const s = valueLib.batchedScalar(Float64Array.from([1, 2, 3, 4]));
  const out = broadcastN((x: number) => x + 1, [s], N4);
  assert.deepEqual(out.shape, [N4], 'stays on the rank-1 scalar path');
  assert.deepEqual(Array.from(out.data), [2, 3, 4, 5]);
});

test('broadcastN: outerRank=0 opts a [N,D] Value out of the cell path', () => {
  // `outerRank = 0` says no leading axis is a loop axis, so the value is
  // not atom-batched at all — `isAtomBatched` refuses it and the cell
  // detection must agree rather than claiming the leading axis.
  const v: any = valueLib.withShape(
    Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]), [N4, 2]);
  v.outerRank = 0;
  const out = broadcastN((x: number) => x + 1, [v], N4);
  assert.equal(out.shape.length, 0, 'falls back to the rank-0 read');
});

test('broadcastN: a raw Float64Array still returns raw when no Value is present', () => {
  const raw = Float64Array.from([1, 2, 3, 4]);
  const out = broadcastN((x: number) => x * 3, [raw], N4);
  assert.ok(out instanceof Float64Array, 'no Value operand ⇒ raw result');
  assert.deepEqual(Array.from(out), [3, 6, 9, 12]);
});
