'use strict';

// =====================================================================
// ksuperpose-multivariate.test.ts — spec §06 family axes by parameter rank
// =====================================================================
//
// SPEC ANCHOR — docs/06-measure-algebra.md, "Additive superposition",
// `ksuperpose`, quoted verbatim: "The family is passed as to `broadcast` —
// positional collections, keyword collections, or a table, whose columns are
// its collection arguments — with one family axis per collection argument: an
// argument's family axes are its leading axes in excess of the rank (number of
// axes) of the parameter it feeds, and any count other than one is a static
// error. Within the family the same-number-of-axes requirement of *Collection
// arguments* does not apply, so the components may be multivariate — a vector
// parameter takes an $N \times d$ matrix while a matrix parameter takes an
// $N \times d \times d$ array. Along the family axis each collection argument
// has size $N$ or is singular (size one, expanded by repetition), and
// non-collection arguments are held constant across the components."
//
// So `ksuperpose(MvNormal, w)(mu = mus, cov = covs)` with `mus` an N x d matrix
// and `covs` an N-vector of d x d matrices is a legal mixture of multivariate
// normals, and its variate is the COMPONENT variate — a vector of d, not an
// array over the family.
//
// ORACLE — scipy `multivariate_normal` + `logsumexp`, computed before any
// engine output was read:
//
//   lp(x) = logsumexp_i(log w_i + mvn(mu_i, cov_i).logpdf(x)) - log(sum w)
//
// with `mus = [[0,0], [3,3], [-2,1]]`, `covs = [c1, c2, c3]`,
// `c1 = [[1,0.2],[0.2,1]]`, `c2 = [[2,0],[0,0.5]]`, `c3 = [[1.5,-0.3],[-0.3,1.5]]`:
//
//   | w                 | x        | oracle lp          |
//   |-------------------|----------|--------------------|
//   | [0.2, 0.5, 0.3]   | [0.5,0.5]| -3.491504380062997 |
//   | [0.2, 0.5, 0.3]   | [0,0]    | -3.217733180735057 |
//   | [0.2, 0.5, 0.3]   | [3,3]    | -2.530794166141225 |
//   | [0.2, 0.5, 0.3]   | [-2,1]   | -3.379212348715778 |
//   | [0.3, 1.2, 0.5]   | [0.5,0.5]| -3.763911141798897 |  (Z = 2, log Z = 0.693147180559945)
//   | [0.0, 0.5, 0.5]   | [0.5,0.5]| -4.994646873022405 |  (a zero weight drops out)
//
// Singular arms, same weights and x = [0.5, 0.5]:
//   every component shares `c1`   (`covs = [c1]`)        -> -3.572204030622333
//   every component shares one mu (`mus` one row [1,-1]) -> -3.487708424806220
//
// SAMPLING oracle — closed-form mixture moments for w = [0.3, 0.7],
// mu = [-2,1] / [3,-1], cov = [[1,0.3],[0.3,1]] / [[0.5,0],[0,2]]:
//   mean = Sum_i w_i mu_i                            = [1.5, -0.4]
//   cov  = Sum_i w_i (cov_i + mu_i mu_i^T) - mean^2   = [[5.9, -2.01], [-2.01, 2.54]]

const { test } = require('node:test');
const assert = require('node:assert');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const TOL = 1e-9;

function ctxFor(src: string, n: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
  assert.deepEqual(errs, [], 'unexpected diagnostics');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0xB0A7 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: string) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker: (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: n,
    rootSeed: 0xB0A7,
  };
  return ctx;
}

async function score(src: string): Promise<number> {
  const m = await ctxFor(src + '__score__ = logdensityof(mix, x)\n', 16)
    .getMeasure('__score__');
  return Number(m.samples[0]);
}

const errorsOf = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

// The witness family: three bivariate normals, distinct means AND distinct
// covariances, so `mu` takes a 3 x 2 matrix beside `cov`'s 3 x 2 x 2 array.
const FAMILY = 'mus = rowstack([[0.0, 0.0], [3.0, 3.0], [-2.0, 1.0]])\n'
  + 'c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n'
  + 'c2 = rowstack([[2.0, 0.0], [0.0, 0.5]])\n'
  + 'c3 = rowstack([[1.5, -0.3], [-0.3, 1.5]])\n'
  + 'covs = [c1, c2, c3]\n';

// ── The mixture types and scores ─────────────────────────────────────────────

test('the multivariate mixture is a measure over the COMPONENT variate', () => {
  const out = processSource(FAMILY
    + 'w = [0.2, 0.5, 0.3]\n'
    + 'mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n');
  assert.deepEqual((out.diagnostics || [])
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  const t = out.bindings.get('mix').inferredType;
  assert.equal(t.kind, 'measure', `mix is ${JSON.stringify(t)}`);
  assert.equal(t.domain.kind, 'array', 'the variate is MvNormal\'s own vector');
  assert.deepEqual(t.domain.shape, [2], `variate shape ${JSON.stringify(t.domain.shape)}`);
});

for (const [x, oracle] of [
  ['[0.5, 0.5]', -3.491504380062997],
  ['[0.0, 0.0]', -3.217733180735057],
  ['[3.0, 3.0]', -2.530794166141225],
  ['[-2.0, 1.0]', -3.379212348715778],
] as [string, number][]) {
  test(`the mixture density at x = ${x} matches the scipy oracle`, async () => {
    const got = await score(FAMILY
      + 'w = [0.2, 0.5, 0.3]\n'
      + `x = ${x}\n`
      + 'mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n');
    assert.ok(Math.abs(got - oracle) <= TOL, `got ${got}, oracle ${oracle}`);
  });
}

test('UNNORMALIZED weights divide by Z = sum w, not by 1', async () => {
  const src = FAMILY + 'w = [0.3, 1.2, 0.5]\nx = [0.5, 0.5]\n';
  const norm = await score(src
    + 'mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n');
  assert.ok(Math.abs(norm - (-3.763911141798897)) <= TOL,
    `got ${norm}, oracle -3.763911141798897`);
  const bare = await score(src
    + 'mix = ksuperpose(MvNormal, w)(mu = mus, cov = covs)\n');
  // Z = 2, so the two differ by exactly log 2.
  assert.ok(Math.abs((bare - norm) - 0.693147180559945) <= TOL,
    `log Z = ${bare - norm}, want 0.693147180559945`);
});

test('a ZERO weight drops its component out of the mixture', async () => {
  const got = await score(FAMILY
    + 'w = [0.0, 0.5, 0.5]\nx = [0.5, 0.5]\n'
    + 'mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n');
  assert.ok(Math.abs(got - (-4.994646873022405)) <= TOL,
    `got ${got}, oracle -4.994646873022405`);
});

// ── Singular family axes expand by repetition, at every rank ─────────────────

test('a SINGULAR matrix-parameter axis expands: one cov serves all components',
  async () => {
    const got = await score(FAMILY
      + 'w = [0.2, 0.5, 0.3]\nx = [0.5, 0.5]\nshared = [c1]\n'
      + 'mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = shared))\n');
    assert.ok(Math.abs(got - (-3.572204030622333)) <= TOL,
      `got ${got}, oracle -3.572204030622333`);
  });

test('a SINGULAR vector-parameter axis expands: one mu serves all components',
  async () => {
    const got = await score(FAMILY
      + 'w = [0.2, 0.5, 0.3]\nx = [0.5, 0.5]\n'
      + 'onemu = rowstack([[1.0, -1.0]])\n'
      + 'mix = normalize(ksuperpose(MvNormal, w)(mu = onemu, cov = covs))\n');
    assert.ok(Math.abs(got - (-3.487708424806220)) <= TOL,
      `got ${got}, oracle -3.487708424806220`);
  });

// ── Sampling ────────────────────────────────────────────────────────────────

test('the multivariate mixture samples with the closed-form moments',
  async () => {
    const N = 40000;
    const ctx = ctxFor(
      'w = [0.3, 0.7]\n'
      + 'mus = rowstack([[-2.0, 1.0], [3.0, -1.0]])\n'
      + 'cA = rowstack([[1.0, 0.3], [0.3, 1.0]])\n'
      + 'cB = rowstack([[0.5, 0.0], [0.0, 2.0]])\n'
      + 'covs = [cA, cB]\n'
      + 'M = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n'
      + 'y ~ M\n', N);
    const m = await ctx.getMeasure('y');
    const s = Array.from(m.samples as any).map(Number);
    assert.equal(s.length, N * 2, 'atom-major [N, 2] samples');
    const mean = [0, 0];
    for (let i = 0; i < s.length; i++) mean[i % 2] += s[i] / N;
    const cov = [[0, 0], [0, 0]];
    for (let a = 0; a < N; a++) {
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          cov[i][j] += (s[a * 2 + i] - mean[i]) * (s[a * 2 + j] - mean[j]) / (N - 1);
        }
      }
    }
    // Monte-Carlo slack at N = 40000, following ksuperpose-sampling.test.ts.
    // The defect this pins — a per-CELL component choice — put BOTH means at
    // the pooled 0.58 / 0.60 and both variances at 5.5, so the tolerances sit
    // far inside the failure.
    assert.ok(Math.abs(mean[0] - 1.5) < 0.1, `mean[0] = ${mean[0]}, want 1.5`);
    assert.ok(Math.abs(mean[1] + 0.4) < 0.1, `mean[1] = ${mean[1]}, want -0.4`);
    assert.ok(Math.abs(cov[0][0] - 5.9) < 0.25, `var[0] = ${cov[0][0]}, want 5.9`);
    assert.ok(Math.abs(cov[1][1] - 2.54) < 0.15, `var[1] = ${cov[1][1]}, want 2.54`);
    assert.ok(Math.abs(cov[0][1] + 2.01) < 0.15,
      `cov[0,1] = ${cov[0][1]}, want -2.01`);
  });

// ── The refusals §06 still owes ──────────────────────────────────────────────

test('two family axes over a SCALAR parameter stays a static error', () => {
  const ds = errorsOf('w = [0.3, 1.2]\n'
    + 'grid = rowstack([[0.0, 1.0], [2.0, 3.0]])\n'
    + 'mix = ksuperpose(Normal, w)(mu = grid, sigma = 1.0)\n');
  assert.ok(ds.some((m: string) => /exactly one family axis/.test(m)
      && /`mu` has rank 0/.test(m) && /gives 2 family axes/.test(m)),
  `want a family-axis refusal, got ${JSON.stringify(ds)}`);
});

test('ZERO family axes over a matrix parameter is a static error — only a '
  + 'NON-collection is held constant', () => {
  const ds = errorsOf('w = [0.4, 0.6]\n'
    + 'mus = rowstack([[0.0, 0.0], [3.0, 3.0]])\n'
    + 'cov = rowstack([[1.0, 0.2], [0.2, 1.0]])\n'
    + 'mix = ksuperpose(MvNormal, w)(mu = mus, cov = cov)\n');
  assert.ok(ds.some((m: string) => /exactly one family axis/.test(m)
      && /`cov` has rank 2/.test(m) && /gives 0 family axes/.test(m)),
  `want a family-axis refusal, got ${JSON.stringify(ds)}`);
});

test('a vector-element table column over a SCALAR parameter is a static error',
  () => {
    const ds = errorsOf('w = [0.4, 0.6]\n'
      + 'pars = table(mu = [[0.0, 0.0], [3.0, 3.0]], sigma = [1.0, 0.5])\n'
      + 'mix = ksuperpose(Normal, w)(pars)\n');
    assert.ok(ds.some((m: string) => /exactly one family axis/.test(m)
        && /`mu` has rank 0/.test(m)),
    `want a family-axis refusal, got ${JSON.stringify(ds)}`);
  });

// A row-vector family argument is one axis too (§03 keeps a `%tvector` distinct
// from a rank-1 array, but its `len` is an axis all the same), so it feeds a
// scalar parameter with exactly one family axis.
test('a ROW-VECTOR family argument carries one family axis', async () => {
  const got = await score('w = [0.3, 0.7]\n'
    + 'mus = adjoint([-1.0, 2.0])\n'
    + 'x = 0.5\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n');
  // scipy: logsumexp(log([0.3, 0.7]) + norm.logpdf(0.5, [-1, 2], 1)) - log(1)
  assert.ok(Math.abs(got - (-2.043938533204673)) <= TOL,
    `got ${got}, oracle -2.043938533204673`);
});
