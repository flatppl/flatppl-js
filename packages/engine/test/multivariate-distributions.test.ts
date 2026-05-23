'use strict';

// Tests for Dirichlet + Multinomial — the two M-sized multivariate
// distributions added in the §08 fill-in. Same makeCtx scaffolding as
// mvnormal.test.js so the materialiser pipeline (classifier → mat*
// handler → atom-major Value storage) is exercised end-to-end.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 4096;
const ROOT_SEED    = 0xCAFEBEEF;

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

function atomMean(samples: any, K: any) {
  const N = samples.length / K;
  const out = new Float64Array(K);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < K; k++) out[k] += samples[i * K + k];
  }
  for (let k = 0; k < K; k++) out[k] /= N;
  return out;
}

// =====================================================================
// Dirichlet
// =====================================================================

test('Dirichlet: classifier recognises and produces shape=[N, K] atoms', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.value, 'matDirichlet must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
  assert.deepEqual(m.dims, [3]);
});

test('Dirichlet: every atom sums to 1 (l1-normalized)', async () => {
  const ctx = makeCtx(`
alpha = [2.0, 3.0, 5.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const N = m.samples.length / K;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += m.samples[i * K + k];
    assert.ok(Math.abs(s - 1) < 1e-10,
      `atom ${i} sum = ${s} (expected 1 within 1e-10)`);
  }
});

test('Dirichlet: every coordinate is non-negative', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] >= 0, `samples[${i}] = ${m.samples[i]} must be >= 0`);
  }
});

test('Dirichlet: empirical mean ≈ alpha / sum(alpha)', async () => {
  // Asymmetric Dirichlet — easier diagnostic than the symmetric case.
  const ctx = makeCtx(`
alpha = [1.0, 2.0, 3.0, 4.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  const K = 4;
  const alphaSum = 1 + 2 + 3 + 4; // 10
  const expected = [0.1, 0.2, 0.3, 0.4];
  const observed = atomMean(m.samples, K);
  // Dirichlet variance per coord: a_k (a0 - a_k) / (a0² (a0 + 1)). For
  // a_k=4, a0=10 → var ≈ 4*6 / (100*11) ≈ 0.022; sd ≈ 0.15; MC error
  // ≈ 0.15 / sqrt(4096) ≈ 0.0023. Tolerate 6× for the largest coord.
  for (let k = 0; k < K; k++) {
    assert.ok(Math.abs(observed[k] - expected[k]) < 0.015,
      `coord ${k}: observed=${observed[k]} expected=${expected[k]}`);
  }
});

test('Dirichlet: rejects non-positive alpha', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 0.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  await assert.rejects(ctx.getMeasure('m'), /alpha\[1\] = 0 must be positive/);
});

test('Dirichlet: rejects scalar alpha', async () => {
  const ctx = makeCtx(`
m = Dirichlet(alpha = 1.0)
`);
  await assert.rejects(ctx.getMeasure('m'), /alpha must be a vector/);
});

test('Dirichlet: positional alpha argument also accepted', async () => {
  // Spec §08 doesn't mandate kwarg-only; the classifier picks up either
  // form. Test the positional version for completeness.
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
});

// =====================================================================
// Multinomial
// =====================================================================

test('Multinomial: classifier recognises and produces shape=[N, K] atoms', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.3, 0.2]
m = Multinomial(n = 10, p = p)
`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.value, 'matMultinomial must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
  assert.deepEqual(m.dims, [3]);
});

test('Multinomial: every atom is a length-K vector summing to n', async () => {
  const N_TRIALS = 12;
  const ctx = makeCtx(`
p = [0.25, 0.5, 0.25]
m = Multinomial(n = 12, p = p)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const N = m.samples.length / K;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) {
      const c = m.samples[i * K + k];
      assert.ok(Number.isInteger(c), `samples[${i},${k}] = ${c} must be integer`);
      assert.ok(c >= 0, `samples[${i},${k}] = ${c} must be >= 0`);
      s += c;
    }
    assert.equal(s, N_TRIALS, `atom ${i} sum = ${s} expected ${N_TRIALS}`);
  }
});

test('Multinomial: empirical mean ≈ n * p per coord', async () => {
  const n = 20;
  const ctx = makeCtx(`
p = [0.1, 0.4, 0.5]
m = Multinomial(n = 20, p = p)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const expected = [n * 0.1, n * 0.4, n * 0.5];
  const observed = atomMean(m.samples, K);
  // Multinomial variance per coord: n*p_k*(1-p_k). For p=0.5, n=20:
  // var=5; sd≈2.24. MC error ≈ 2.24/sqrt(4096) ≈ 0.035. Tolerate 6×.
  for (let k = 0; k < K; k++) {
    assert.ok(Math.abs(observed[k] - expected[k]) < 0.2,
      `coord ${k}: observed=${observed[k]} expected=${expected[k]}`);
  }
});

test('Multinomial: n=0 produces zero-count atoms', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = 0, p = p)
`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    assert.equal(m.samples[i], 0, `samples[${i}] = ${m.samples[i]} expected 0`);
  }
});

test('Multinomial: rejects negative n', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = -1, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /n must be a non-negative integer/);
});

test('Multinomial: rejects non-integer n', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = 1.5, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /n must be a non-negative integer/);
});

test('Multinomial: rejects negative p coord', async () => {
  const ctx = makeCtx(`
p = [0.5, -0.1, 0.6]
m = Multinomial(n = 10, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /p\[1\] = -0\.1 must be non-negative/);
});

test('Multinomial: rejects all-zero p', async () => {
  const ctx = makeCtx(`
p = [0.0, 0.0, 0.0]
m = Multinomial(n = 10, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /p must sum to > 0/);
});

// =====================================================================
// Wishart
// =====================================================================

function matrixAtom(samples: any, atom: any, n: any) {
  // Extract atom's n×n matrix from atom-major shape=[N, n, n] storage.
  const out = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) out[i] = samples[atom * n * n + i];
  return out;
}

function matrixMean(samples: any, n: any) {
  const N = samples.length / (n * n);
  const out = new Float64Array(n * n);
  for (let atom = 0; atom < N; atom++) {
    for (let i = 0; i < n * n; i++) out[i] += samples[atom * n * n + i];
  }
  for (let i = 0; i < n * n; i++) out[i] /= N;
  return out;
}

test('Wishart: classifier recognises and produces matrix atoms', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0]]
m = Wishart(nu = 5, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.value, 'matWishart must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 2, 2]);
  assert.deepEqual(m.dims, [2, 2]);
});

test('Wishart: every atom is symmetric', async () => {
  const ctx = makeCtx(`
scale = [[2.0, 0.5], [0.5, 1.0]]
m = Wishart(nu = 5, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  const N = m.samples.length / 4;
  for (let atom = 0; atom < N; atom++) {
    const a01 = m.samples[atom * 4 + 1];
    const a10 = m.samples[atom * 4 + 2];
    assert.ok(Math.abs(a01 - a10) < 1e-12,
      `atom ${atom}: W[0,1]=${a01} W[1,0]=${a10}`);
  }
});

test('Wishart: every atom is positive-semidefinite (positive diagonal)', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0]]
m = Wishart(nu = 5, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  const N = m.samples.length / 4;
  for (let atom = 0; atom < N; atom++) {
    assert.ok(m.samples[atom * 4]     > 0, `atom ${atom}: W[0,0] not positive`);
    assert.ok(m.samples[atom * 4 + 3] > 0, `atom ${atom}: W[1,1] not positive`);
    // 2×2 SPD ⇔ both diag > 0 AND det > 0.
    const det = m.samples[atom * 4] * m.samples[atom * 4 + 3]
              - m.samples[atom * 4 + 1] * m.samples[atom * 4 + 2];
    assert.ok(det > 0, `atom ${atom}: det not positive (${det})`);
  }
});

test('Wishart: empirical mean ≈ nu * scale', async () => {
  const nu = 10;
  const ctx = makeCtx(`
scale = [[2.0, 0.3], [0.3, 1.0]]
m = Wishart(nu = 10, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  const observed = matrixMean(m.samples, 2);
  const expected = [nu * 2.0, nu * 0.3, nu * 0.3, nu * 1.0];
  // Wishart variance E[W_ij²] = nu * (S_ii * S_jj + S_ij²); sd of each
  // entry is ~sqrt(nu) * S magnitudes. MC error ~sqrt(nu)*S/sqrt(N).
  // For nu=10 with N=4096, ~sqrt(10)*2/sqrt(4096) ≈ 0.10. Tolerate 6×.
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(observed[i] - expected[i]) < 0.6,
      `entry ${i}: observed=${observed[i]} expected=${expected[i]}`);
  }
});

test('Wishart: rejects nu ≤ n - 1', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0]]
m = Wishart(nu = 1, scale = scale)
`);
  await assert.rejects(ctx.getMeasure('m'), /must be > n - 1/);
});

test('Wishart: rejects non-square scale', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
m = Wishart(nu = 5, scale = scale)
`);
  await assert.rejects(ctx.getMeasure('m'), /scale must be a square matrix/);
});

// =====================================================================
// InverseWishart
// =====================================================================

test('InverseWishart: classifier recognises and produces matrix atoms', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0]]
m = InverseWishart(nu = 5, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 2, 2]);
  assert.deepEqual(m.dims, [2, 2]);
});

test('InverseWishart: every atom is symmetric + SPD', async () => {
  const ctx = makeCtx(`
scale = [[2.0, 0.5], [0.5, 1.0]]
m = InverseWishart(nu = 6, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  const N = m.samples.length / 4;
  for (let atom = 0; atom < N; atom++) {
    const a = m.samples[atom * 4];
    const b = m.samples[atom * 4 + 1];
    const c = m.samples[atom * 4 + 2];
    const d = m.samples[atom * 4 + 3];
    assert.ok(Math.abs(b - c) < 1e-10,
      `atom ${atom}: symmetry ${b} vs ${c}`);
    assert.ok(a > 0 && d > 0 && (a * d - b * c) > 0,
      `atom ${atom}: not SPD`);
  }
});

test('InverseWishart: empirical mean ≈ scale / (nu - n - 1)', async () => {
  // E[X] = S / (nu - n - 1) for InverseWishart(nu, S) with nu > n + 1.
  // For n=2, nu=10: scale/(10 - 2 - 1) = scale / 7.
  const ctx = makeCtx(`
scale = [[2.0, 0.0], [0.0, 1.0]]
m = InverseWishart(nu = 10, scale = scale)
`);
  const m = await ctx.getMeasure('m');
  const observed = matrixMean(m.samples, 2);
  const expected = [2.0 / 7, 0, 0, 1.0 / 7];
  // InverseWishart variance is higher than Wishart's; tolerate generously.
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(observed[i] - expected[i]) < 0.1,
      `entry ${i}: observed=${observed[i]} expected=${expected[i]}`);
  }
});

// =====================================================================
// LKJCholesky / LKJ
// =====================================================================

test('LKJCholesky: classifier recognises and produces shape=[N, n, n] atoms', async () => {
  const ctx = makeCtx(`
m = LKJCholesky(n = 3, eta = 1.0)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3, 3]);
  assert.deepEqual(m.dims, [3, 3]);
});

test('LKJCholesky: every atom is lower-triangular with positive diagonal', async () => {
  const ctx = makeCtx(`
m = LKJCholesky(n = 3, eta = 2.0)
`);
  const m = await ctx.getMeasure('m');
  const n = 3;
  const N = m.samples.length / (n * n);
  for (let atom = 0; atom < N; atom++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = m.samples[atom * n * n + i * n + j];
        if (j > i) {
          assert.equal(v, 0, `atom ${atom}: L[${i},${j}]=${v} expected 0 (upper tri)`);
        } else if (j === i) {
          assert.ok(v > 0, `atom ${atom}: L[${i},${i}]=${v} expected positive diagonal`);
        }
      }
    }
  }
});

test('LKJCholesky: every row has unit norm (so L L^T is a correlation matrix)', async () => {
  const ctx = makeCtx(`
m = LKJCholesky(n = 4, eta = 1.0)
`);
  const m = await ctx.getMeasure('m');
  const n = 4;
  const N = m.samples.length / (n * n);
  for (let atom = 0; atom < N; atom++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j <= i; j++) {
        const v = m.samples[atom * n * n + i * n + j];
        s += v * v;
      }
      assert.ok(Math.abs(s - 1) < 1e-10,
        `atom ${atom}: row ${i} norm² = ${s} (expected 1)`);
    }
  }
});

test('LKJ: classifier produces shape=[N, n, n]; diagonals are 1', async () => {
  const ctx = makeCtx(`
m = LKJ(n = 3, eta = 1.0)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3, 3]);
  const n = 3;
  const N = m.samples.length / (n * n);
  for (let atom = 0; atom < N; atom++) {
    for (let i = 0; i < n; i++) {
      const diag = m.samples[atom * n * n + i * n + i];
      assert.ok(Math.abs(diag - 1) < 1e-10,
        `atom ${atom}: R[${i},${i}] = ${diag} (expected 1 for correlation matrix)`);
    }
  }
});

test('LKJ: off-diagonals are in [-1, 1] and symmetric', async () => {
  const ctx = makeCtx(`
m = LKJ(n = 3, eta = 1.0)
`);
  const m = await ctx.getMeasure('m');
  const n = 3;
  const N = m.samples.length / (n * n);
  for (let atom = 0; atom < N; atom++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const off = m.samples[atom * n * n + i * n + j];
        const sym = m.samples[atom * n * n + j * n + i];
        assert.ok(Math.abs(off - sym) < 1e-10,
          `atom ${atom}: R[${i},${j}]=${off} vs R[${j},${i}]=${sym}`);
        assert.ok(off >= -1 && off <= 1,
          `atom ${atom}: R[${i},${j}]=${off} out of [-1, 1]`);
      }
    }
  }
});

// =====================================================================
// BinnedPoissonProcess
// =====================================================================

test('BinnedPoissonProcess: classifier recognises and produces shape=[N, K]', async () => {
  const ctx = makeCtx(`
rates = [1.0, 2.0, 3.0]
m = BinnedPoissonProcess(rates = rates)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
  assert.deepEqual(m.dims, [3]);
});

test('BinnedPoissonProcess: every entry is a non-negative integer', async () => {
  const ctx = makeCtx(`
rates = [0.5, 1.0, 2.0, 5.0]
m = BinnedPoissonProcess(rates = rates)
`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    const v = m.samples[i];
    assert.ok(Number.isInteger(v) && v >= 0,
      `samples[${i}] = ${v} must be non-negative integer`);
  }
});

test('BinnedPoissonProcess: empirical mean ≈ rates per coord', async () => {
  const ctx = makeCtx(`
rates = [1.0, 4.0, 9.0]
m = BinnedPoissonProcess(rates = rates)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const expected = [1.0, 4.0, 9.0];
  const observed = atomMean(m.samples, K);
  // Poisson(rate) has sd = sqrt(rate); MC error sqrt(rate)/sqrt(N).
  // Worst case rate=9: 3/sqrt(4096) ≈ 0.047. Tolerate 6×.
  for (let k = 0; k < K; k++) {
    assert.ok(Math.abs(observed[k] - expected[k]) < 0.3,
      `coord ${k}: observed=${observed[k]} expected=${expected[k]}`);
  }
});

test('BinnedPoissonProcess: empirical variance ≈ rates per coord (Poisson)', async () => {
  const ctx = makeCtx(`
rates = [4.0, 4.0, 4.0]
m = BinnedPoissonProcess(rates = rates)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const mean = atomMean(m.samples, K);
  const N = m.samples.length / K;
  const sds = new Float64Array(K);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < K; k++) {
      const d = m.samples[i * K + k] - mean[k];
      sds[k] += d * d;
    }
  }
  for (let k = 0; k < K; k++) {
    const variance = sds[k] / (N - 1);
    // Poisson(4) has variance = 4. MC error of empirical variance is
    // approximately 2*var/sqrt(N) for moderate N (here ~ 0.13).
    assert.ok(Math.abs(variance - 4.0) < 0.4,
      `coord ${k}: variance=${variance} expected ≈ 4`);
  }
});

test('BinnedPoissonProcess: zero rate produces all-zero bin', async () => {
  const ctx = makeCtx(`
rates = [0.0, 5.0]
m = BinnedPoissonProcess(rates = rates)
`);
  const m = await ctx.getMeasure('m');
  const N = m.samples.length / 2;
  for (let i = 0; i < N; i++) {
    assert.equal(m.samples[i * 2], 0, `atom ${i}: zero-rate bin should be 0`);
  }
});

test('BinnedPoissonProcess: rejects negative rates', async () => {
  const ctx = makeCtx(`
rates = [1.0, -1.0]
m = BinnedPoissonProcess(rates = rates)
`);
  await assert.rejects(ctx.getMeasure('m'), /rates\[1\] = -1 must be non-negative/);
});

// =====================================================================
// Defensive-path / rejection coverage — the validation branches we
// hadn't reached yet (cataloguing the failure modes is a real
// requirement for the materialiser surface, beyond just shape
// invariants on the happy path).
// =====================================================================

test('Dirichlet: rejects missing alpha argument', async () => {
  const ctx = makeCtx(`
m = Dirichlet()
`);
  await assert.rejects(ctx.getMeasure('m'), /requires alpha/);
});

test('Multinomial: rejects missing n/p', async () => {
  await assert.rejects(makeCtx(`m = Multinomial(p = [0.5, 0.5])\n`).getMeasure('m'),
                       /requires n and p/);
  await assert.rejects(makeCtx(`m = Multinomial(n = 5)\n`).getMeasure('m'),
                       /requires n and p/);
});

test('Multinomial: rejects scalar p', async () => {
  const ctx = makeCtx(`
m = Multinomial(n = 3, p = 0.5)
`);
  await assert.rejects(ctx.getMeasure('m'), /p must be a vector/);
});

test('Multinomial: rejects empty p', async () => {
  // Bypass the literal-array lift by using a length-via-zeros literal
  // that lowers to an empty-vector. Both empty + zero-sum trip the
  // validation; either rejection is fine.
  const ctx = makeCtx(`
p = zeros(0)
m = Multinomial(n = 3, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /(non-empty|must sum)/);
});

test('Wishart: rejects missing nu / scale', async () => {
  await assert.rejects(makeCtx(`m = Wishart(scale = [[1.0, 0.0], [0.0, 1.0]])\n`).getMeasure('m'),
                       /requires nu and scale/);
  await assert.rejects(makeCtx(`m = Wishart(nu = 5)\n`).getMeasure('m'),
                       /requires nu and scale/);
});

test('Wishart: rejects non-numeric nu', async () => {
  const ctx = makeCtx(`
scale = [[1.0, 0.0], [0.0, 1.0]]
nu = [1.0, 2.0]
m = Wishart(nu = nu, scale = scale)
`);
  await assert.rejects(ctx.getMeasure('m'), /nu must be a number/);
});

test('InverseWishart: rejects same shape problems', async () => {
  await assert.rejects(
    makeCtx(`m = InverseWishart(scale = [[1.0, 0.0], [0.0, 1.0]])\n`).getMeasure('m'),
    /requires nu and scale/);
  await assert.rejects(
    makeCtx(`m = InverseWishart(nu = 1, scale = [[1.0, 0.0], [0.0, 1.0]])\n`).getMeasure('m'),
    /must be > n - 1/);
});

test('LKJCholesky: rejects missing arguments', async () => {
  await assert.rejects(makeCtx(`m = LKJCholesky(eta = 1.0)\n`).getMeasure('m'),
                       /requires n and eta/);
  await assert.rejects(makeCtx(`m = LKJCholesky(n = 3)\n`).getMeasure('m'),
                       /requires n and eta/);
});

test('LKJCholesky: rejects invalid n / eta', async () => {
  await assert.rejects(makeCtx(`m = LKJCholesky(n = 0, eta = 1.0)\n`).getMeasure('m'),
                       /n must be a positive integer/);
  await assert.rejects(makeCtx(`m = LKJCholesky(n = 2.5, eta = 1.0)\n`).getMeasure('m'),
                       /n must be a positive integer/);
  await assert.rejects(makeCtx(`m = LKJCholesky(n = 3, eta = 0.0)\n`).getMeasure('m'),
                       /eta must be a positive number/);
  await assert.rejects(makeCtx(`m = LKJCholesky(n = 3, eta = -1.0)\n`).getMeasure('m'),
                       /eta must be a positive number/);
});

test('LKJCholesky: n=1 produces the trivial 1×1 identity', async () => {
  // Exercises the n===1 short-circuit (otherwise an LKJ-of-1 would be
  // degenerate but the onion procedure still has to no-op cleanly).
  const ctx = makeCtx(`m = LKJCholesky(n = 1, eta = 1.0)\n`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 1, 1]);
  for (let i = 0; i < m.samples.length; i++) {
    assert.equal(m.samples[i], 1, `atom ${i} of 1×1 LKJCholesky must equal 1`);
  }
});

test('BinnedPoissonProcess: rejects missing rates', async () => {
  const ctx = makeCtx(`m = BinnedPoissonProcess()\n`);
  await assert.rejects(ctx.getMeasure('m'), /requires rates/);
});

test('BinnedPoissonProcess: rejects scalar rates', async () => {
  const ctx = makeCtx(`m = BinnedPoissonProcess(rates = 1.0)\n`);
  await assert.rejects(ctx.getMeasure('m'), /rates must be a vector/);
});

test('BinnedPoissonProcess: rejects empty rates', async () => {
  const ctx = makeCtx(`
rates = zeros(0)
m = BinnedPoissonProcess(rates = rates)
`);
  await assert.rejects(ctx.getMeasure('m'), /rates must be non-empty/);
});
