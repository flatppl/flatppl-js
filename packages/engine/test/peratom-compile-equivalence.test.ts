'use strict';

// =====================================================================
// peratom-compile-equivalence.test.ts
// =====================================================================
//
// `_perAtomFallback` compiles the body once per batch instead of
// re-walking it per atom. The compiled route must agree with the
// interpreter to the last bit, so every case here materialises the same
// binding twice — once with the compile ON, once with
// `_setCompilePerAtom(false)` — and compares the raw IEEE-754 bytes of
// every sample and log-weight, not the printed decimals.
//
// The corpus is deliberately whole models rather than synthetic IR: the
// path is reached through the residue branch of `_evalN`, which depends
// on the op mix a real lowering produces.

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('../index.ts');
const { createWorkerHandler } = require('../worker.ts');
const batched = require('../sampler-eval-batched.ts');

// Local resilience copies (flatppl-dev/CONVENTIONS.md "Examples and test
// fixtures"), so the suite does not depend on the sibling clone. Copied
// from flatppl-examples 314d745.
const FIXTURES = path.join(__dirname, 'fixtures');

function buildCtx(src: string, N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N, rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return { ctx, derivations: built.derivations };
}

// Raw bytes of every numeric buffer, so -0, NaN payloads and the last
// mantissa bit all participate in the comparison.
function fingerprint(m: any): string {
  const parts: string[] = [];
  const push = (label: string, buf: any) => {
    if (!buf) { parts.push(label + '=null'); return; }
    const f64 = buf instanceof Float64Array ? buf : Float64Array.from(buf);
    parts.push(label + '=' + Buffer.from(f64.buffer, f64.byteOffset, f64.byteLength)
      .toString('hex'));
  };
  push('samples', m.samples);
  push('logWeights', m.logWeights);
  push('imag', m.im || m.imag);
  parts.push('logTotalmass=' + fingerprintNum(m.logTotalmass));
  parts.push('n_eff=' + fingerprintNum(m.n_eff));
  parts.push('dims=' + JSON.stringify(m.dims || null));
  parts.push('shape=' + JSON.stringify(m.shape || null));
  return parts.join('|');
}

function fingerprintNum(v: any): string {
  if (typeof v !== 'number') return String(v);
  const b = new Float64Array([v]);
  return Buffer.from(b.buffer).toString('hex');
}

// Outcome of one materialisation, success or failure. A failure is part
// of the comparison: a body the compiled route throws on where the
// interpreter succeeds is exactly the regression this file exists to
// catch, so it must not be skippable.
async function outcomeOf(src: string, name: string, N: number) {
  const { ctx } = buildCtx(src, N);
  try {
    return 'ok:' + fingerprint(await ctx.getMeasure(name));
  } catch (e: any) {
    return 'threw:' + ((e && e.message) ? e.message : String(e));
  }
}

// Materialisable bindings of a model, in declaration order.
function measureNames(src: string): string[] {
  const { derivations } = buildCtx(src, 1);
  return Object.keys(derivations).filter((n) => !n.startsWith('__'));
}

async function assertRouteAgreement(file: string, N: number) {
  return assertRouteAgreementSrc(file, fs.readFileSync(path.join(FIXTURES, file), 'utf8'), N);
}

async function assertRouteAgreementSrc(label: string, src: string, N: number) {
  const file = label;
  const names = measureNames(src);
  assert.ok(names.length > 0, `${file}: no bindings to compare`);
  let compared = 0;
  let materialised = 0;
  for (const name of names) {
    batched._setCompilePerAtom(true);
    const hot = await outcomeOf(src, name, N);
    batched._setCompilePerAtom(false);
    let cold: string;
    try {
      cold = await outcomeOf(src, name, N);
    } finally {
      batched._setCompilePerAtom(true);
    }
    assert.strictEqual(hot, cold,
      `${file}: '${name}' differs between the compiled and interpreted route`);
    compared++;
    if (hot.startsWith('ok:')) materialised++;
  }
  assert.strictEqual(compared, names.length, `${file}: skipped a binding`);
  // A model whose every binding threw on both routes would pass the
  // equality check while proving nothing about the compiled numbers.
  assert.ok(materialised > 0, `${file}: no binding materialised on either route`);
  return materialised;
}

// The model the change was measured on: a coherent amplitude sum whose
// weight body is 40092 node visits over 777 objects.
test('per-atom compile: dminus-to-3pi-amplitude is bit-identical on both routes', async () => {
  const n = await assertRouteAgreement('dminus-to-3pi-amplitude.flatppl', 256);
  assert.ok(n >= 5, `expected several bindings, compared ${n}`);
});

// Non-Dalitz coverage: an aggregate/einsum body, a kernel model, a
// hierarchical model, a discrete mixture, a regression and the smallest
// model in the corpus, so the residue branch is entered by op mixes the
// amplitude model never produces — and so a small body on a small batch
// is pinned too, that being where compiling could plausibly cost.
for (const file of [
  'aggregates.flatppl',
  'bayesian_inference_2.flatppl',
  'eight-schools.flatppl',
  'zero-inflated-binomial.flatppl',
  'linear-regression.flatppl',
  'minimal.flatppl',
]) {
  test(`per-atom compile: ${file} is bit-identical on both routes`, async () => {
    await assertRouteAgreement(file, 128);
  });
}

// A `normalize` whose per-atom divisor is an EXPRESSION in a latent, not a
// pooled constant: `_perAtomLogMass` sends `totalMassExpr`'s output through
// `evaluateN`, so the divisor itself is evaluated per atom and lands on the
// same residue branch as any other body. The truncate arm makes the expression
// `p·Z_t + q·1` with Z_t a baked literal, so both routes must produce the same
// weights to the last bit — a divisor that differed between them would move the
// θ-marginal on one route only. Inline rather than a fixture: the shape is a
// §06 mixture spelling, not a corpus model.
test('per-atom compile: a θ-dependent truncate-component divisor is bit-identical',
  async () => {
    const src = 'flatppl_compat = "0.1"\n'
      + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
      + 'q = 1.0 - p\n'
      + 'm = normalize(superpose('
      + 'weighted(p, truncate(Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0))), '
      + 'weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n';
    const n = await assertRouteAgreementSrc('normalize-truncate-mixture', src, 128);
    assert.ok(n >= 3, `expected p, q, m and y to materialise, compared ${n}`);
  });

// The kill switch has to actually reach the loop, or the comparisons
// above would be two runs of the same route.
test('per-atom compile: the kill switch selects the interpreter loop', () => {
  const ir = { kind: 'call', op: 'vector', args: [{ kind: 'ref', ns: 'self', name: 'x' }] };
  batched._setCompilePerAtom(false);
  try {
    assert.strictEqual(batched._profileProgramFor(ir), null);
  } finally {
    batched._setCompilePerAtom(true);
  }
  const prog = batched._profileProgramFor(ir);
  assert.ok(prog && typeof prog.evalPoint === 'function');
  // Cached by IR identity, so a second ask is the same program.
  assert.strictEqual(batched._profileProgramFor(ir), prog);
});

// A compiler throw must degrade to the interpreter, not to an error.
test('per-atom compile: a compiler throw is contained and cached as a refusal', () => {
  const sampler = require('../sampler.ts');
  const real = sampler.compileProfileBody;
  const ir = { kind: 'call', op: 'vector', args: [] };
  sampler.compileProfileBody = () => { throw new Error('synthetic codegen failure'); };
  try {
    assert.strictEqual(batched._profileProgramFor(ir), null);
  } finally {
    sampler.compileProfileBody = real;
  }
  // Refusal is sticky: the restored compiler is not consulted again.
  assert.strictEqual(batched._profileProgramFor(ir), null);
});
