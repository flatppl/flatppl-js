'use strict';

// jointchain/kchain first-class derivation kind — STEP 1 (classifier +
// explicit step structure, behind the off-by-default
// JOINTCHAIN_STATE.firstClass flag). Dual-path migration: with the
// flag off the legacy inlineChainOps rewrite is unchanged (zero
// behaviour change — the rest of the suite proves that); with it on,
// classifyJointchain builds the explicit step structure. Materialiser
// is a step-2 stub, so these tests assert CLASSIFICATION only.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');
const { buildDerivations } = orchestrator;
const { JOINTCHAIN_STATE } = orchestrator._internal;

// Materialise `target` from `src` with the first-class flag forced on
// (restored after). Mirrors the kernel-broadcast test harness.
function materialise(src, target, sampleCount) {
  const prev = JOINTCHAIN_STATE.firstClass;
  JOINTCHAIN_STATE.firstClass = true;
  try {
    const lifted = processSource(src);
    const built = orchestrator.buildDerivations(lifted.bindings);
    const worker = createWorkerHandler();
    worker.handle({ type: 'init', seed: 4242 });
    const cache = new Map();
    const ctx = {
      derivations: built.derivations,
      bindings: built.bindings,
      fixedValues: built.fixedValues || new Map(),
      getMeasure: (n) => {
        if (cache.has(n)) return cache.get(n);
        const p = materialiser.materialiseMeasure(n, ctx);
        cache.set(n, p);
        return p;
      },
      sendWorker: (m) => {
        const r = worker.handle(m);
        return r && r.type === 'error'
          ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
      },
      sampleCount: sampleCount || 6000,
      rootSeed: 4242,
    };
    return ctx.getMeasure(target);
  } finally { JOINTCHAIN_STATE.firstClass = prev; }
}

function meanOf(arr) {
  let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}
function varOf(arr) {
  const m = meanOf(arr); let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2;
  return s / arr.length;
}

// Run `fn` with the first-class flag forced on, always restoring it
// (test isolation — every other suite must see the default off path).
function withFirstClass(fn) {
  const prev = JOINTCHAIN_STATE.firstClass;
  JOINTCHAIN_STATE.firstClass = true;
  try { return fn(); } finally { JOINTCHAIN_STATE.firstClass = prev; }
}

const KDEF =
  'M = Normal(mu = 0.0, sigma = 1.0)\n' +
  'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
  'K2 = functionof(Normal(mu = y, sigma = 1.0), y = y)\n';

test('flag OFF (default): jointchain still goes through legacy rewrite', () => {
  // No kind:'jointchain' is produced; inlineChainOps rewrites to a
  // record/tuple/joint shape exactly as before. (Zero-behaviour-change
  // guarantee for step 1.)
  const { derivations } = buildDerivations(
    processSource(KDEF + 'jc = jointchain(a = M, b = K)\n').bindings);
  const d = derivations['jc'];
  assert.ok(!d || d.kind !== 'jointchain',
    'with the flag off, no first-class jointchain kind may appear');
  assert.equal(JOINTCHAIN_STATE.firstClass, false, 'flag stays off by default');
});

test('flag ON: positional 2-arg jointchain → explicit step structure', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(M, K)\n').bindings);
    const d = derivations['jc'];
    assert.ok(d, 'jc classified');
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.marginalize, false);
    assert.equal(d.labels, null);
    assert.deepEqual(d.steps, [
      { var: 's0', role: 'base', ref: 'M', kernel: false },
      { var: 's1', role: 'kernel', ref: 'K', inputs: ['s0'] },
    ]);
  });
});

test('flag ON: kchain sets marginalize:true', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'kc = kchain(M, K)\n').bindings);
    const d = derivations['kc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.marginalize, true, 'kchain marginalizes intermediates');
  });
});

test('flag ON: kwarg form carries labels (record-shaped)', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(p = M, q = K)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.deepEqual(d.labels, ['p', 'q']);
    assert.deepEqual(d.steps.map((s) => s.var), ['p', 'q']);
    assert.deepEqual(d.steps[1].inputs, ['p']);
  });
});

test('flag ON: N-ary positional → left-assoc step inputs accumulate', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(M, K, K2)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.steps.length, 3);
    assert.deepEqual(d.steps[2],
      { var: 's2', role: 'kernel', ref: 'K2', inputs: ['s0', 's1'] });
  });
});

test('flag ON: kernel-first (step-0 ref is itself a kernel)', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(K, K2)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.steps[0].role, 'base');
    assert.equal(d.steps[0].kernel, true, 'kernel-first base flagged');
  });
});

test('flag ON: inline fn/functionof kernel arg (the kchain(Exp,fn) case)', () => {
  // Regression for the user-reported "not plottable":
  //   d = kchain(Exponential(1), fn(Normal(0, _)))
  // The lift hoists the measure Exponential(1) → a ref but leaves the
  // hole-containing fn(...) inline as a functionof IR (liftMeasure:989).
  // The IR-driven classifier must still build a step structure: base =
  // measure ref, kernel step = inline kernelIR (no ref), marginalize.
  withFirstClass(() => {
    const { derivations } = buildDerivations(processSource(
      'd = kchain(Exponential(rate = 1.0), fn(Normal(mu = 0.0, sigma = _)))\n'
    ).bindings);
    const d = derivations['d'];
    assert.ok(d, 'd classified (was "not plottable" before)');
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.marginalize, true, 'kchain');
    assert.equal(d.steps.length, 2);
    assert.equal(d.steps[0].role, 'base');
    assert.equal(d.steps[0].kernel, false, 'base is the Exponential measure');
    assert.ok(d.steps[0].ref, 'base hoisted to a measure ref');
    assert.equal(d.steps[1].role, 'kernel');
    assert.equal(d.steps[1].ref, undefined, 'inline kernel ⇒ no ref');
    assert.ok(d.steps[1].kernelIR
      && d.steps[1].kernelIR.op === 'functionof',
      'inline kernel carried structurally as kernelIR');
    assert.deepEqual(d.steps[1].inputs, ['s0']);
  });
});

test('matJointchain: kchain(Normal,fn) marginal = Normal(0, sqrt2)', async () => {
  // a ~ N(0,1); b ~ N(a,1); kchain marginalizes a ⇒ b ~ N(0, 2).
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'd = kchain(M, K)\n', 'd', 8000);
  assert.ok(m.samples, 'kchain yields a scalar-atom measure (marginal b)');
  assert.ok(Math.abs(meanOf(m.samples)) < 0.08, 'mean ~ 0');
  assert.ok(Math.abs(varOf(m.samples) - 2.0) < 0.25, 'var ~ 2');
});

test('matJointchain: the user case kchain(Exp(1), fn(Normal(0,_)))', async () => {
  // a ~ Exp(1); b ~ Normal(0, a). E[b]=0, Var(b)=E[a^2]=2 (rate 1).
  const m = await materialise(
    'd = kchain(Exponential(rate = 1.0), fn(Normal(mu = 0.0, sigma = _)))\n',
    'd', 8000);
  assert.ok(m.samples, 'now plottable (was the reported gap)');
  assert.ok(Math.abs(meanOf(m.samples)) < 0.1, 'symmetric, mean ~ 0');
  assert.ok(varOf(m.samples) > 1.0, 'heavy-tailed scale mixture, var ~ 2');
});

test('matJointchain: jointchain(M,K) retains both variates (tuple)', async () => {
  // a ~ N(0,1); b ~ N(a,1); jointchain keeps (a,b).
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'jc = jointchain(M, K)\n', 'jc', 8000);
  assert.ok(m.elems && m.elems.length === 2, 'tuple of (a, b)');
  const a = m.elems[0].samples, b = m.elems[1].samples;
  assert.ok(Math.abs(meanOf(a)) < 0.08 && Math.abs(meanOf(b)) < 0.1);
  // Cov(a,b) = Var(a) = 1 (b = a + noise).
  let cov = 0; const ma = meanOf(a), mb = meanOf(b);
  for (let i = 0; i < a.length; i++) cov += (a[i] - ma) * (b[i] - mb);
  cov /= a.length;
  assert.ok(Math.abs(cov - 1.0) < 0.15, 'cov(a,b) ~ 1, got ' + cov);
});

test('matJointchain: kwarg jointchain(p=M,q=K) → record-shaped', async () => {
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'jc = jointchain(p = M, q = K)\n', 'jc', 4000);
  assert.ok(m.fields && m.fields.p && m.fields.q, 'record { p, q }');
});

test('matJointchain 2b-ext: N-ary kchain marginal (3-step) ≈ N(0,3)', async () => {
  // Left-assoc (spec §06): a0~N(0,1); a1~N(a0,1); a2~N(a1,1). The 3rd
  // kernel takes the cat [a0,a1] (`c~K3([a,b])`) and uses a1=get(_,2).
  // kchain marginal a2 ~ N(0, var 1+1+1 = 3).
  const m = await materialise(
    'M  = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K1 = fn(Normal(mu = _, sigma = 1.0))\n' +
    'K2 = fn(Normal(mu = get(_, 2), sigma = 1.0))\n' +
    'd  = kchain(M, K1, K2)\n', 'd', 12000);
  assert.ok(m.samples, 'scalar marginal (a2)');
  assert.ok(Math.abs(meanOf(m.samples)) < 0.1, 'mean ~ 0');
  assert.ok(Math.abs(varOf(m.samples) - 3.0) < 0.35,
    'var ~ 3, got ' + varOf(m.samples));
});

test('matJointchain 2b-ext: N-ary jointchain (3-step) retains (a0,a1,a2)', async () => {
  const m = await materialise(
    'M  = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K1 = fn(Normal(mu = _, sigma = 1.0))\n' +
    'K2 = fn(Normal(mu = get(_, 2), sigma = 1.0))\n' +
    'jc = jointchain(M, K1, K2)\n', 'jc', 8000);
  assert.ok(m.elems && m.elems.length === 3, 'tuple of (a0,a1,a2)');
  const a0 = m.elems[0].samples, a1 = m.elems[1].samples,
        a2 = m.elems[2].samples;
  assert.ok(Math.abs(varOf(a0) - 1) < 0.15, 'var a0 ~ 1');
  assert.ok(Math.abs(varOf(a1) - 2) < 0.3, 'var a1 ~ 2');
  assert.ok(Math.abs(varOf(a2) - 3) < 0.45, 'var a2 ~ 3');
});

// ---- 2c: density (consume/rest via expandMeasureIR) ----
const LOG2PI = Math.log(2 * Math.PI);
const normLogpdf = (x, mu, sig) =>
  -0.5 * LOG2PI - Math.log(sig) - ((x - mu) ** 2) / (2 * sig * sig);

test('density 2c: kchain marginal logdensity ≈ Normal(0, sqrt2)', async () => {
  // a~N(0,1); b~N(a,1); kchain marginal = N(0, sqrt2). MC estimator
  // logsumexp_i logp(0 | N(a_i,1)) − logN → logpdf_{N(0,sqrt2)}(0).
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'd = kchain(M, K)\n' +
    'lp = logdensityof(d, 0.0)\n', 'lp', 9000);
  const got = m.samples[0];                       // broadcast scalar
  const want = normLogpdf(0, 0, Math.sqrt(2));     // ≈ -1.2655
  assert.ok(Math.abs(got - want) < 0.06,
    `kchain marginal logp ${got} vs analytic ${want}`);
});

test('density 2c: labelled jointchain joint logdensity = p(a)·p(b|a)', async () => {
  // Exact (no MC): logp = logN(0.5;0,1) + logN(1.0;0.5,1).
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'jc = jointchain(p = M, q = K)\n' +
    'lp = logdensityof(jc, record(p = 0.5, q = 1.0))\n', 'lp', 2000);
  const got = m.samples[0];
  const want = normLogpdf(0.5, 0, 1) + normLogpdf(1.0, 0.5, 1); // ≈ -2.0879
  assert.ok(Math.abs(got - want) < 1e-6,
    `jointchain joint logp ${got} vs analytic ${want}`);
});

test('density 2b-ext: N-ary labelled jointchain joint logdensity (3-step)', async () => {
  // logp = logN(0.3;0,1) + logN(0.5;0.3,1) + logN(0.9;0.5,1). Exact
  // (no MC) — the 3rd kernel takes cat[p,q] and uses q = get(_,2).
  const m = await materialise(
    'M  = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K1 = fn(Normal(mu = _, sigma = 1.0))\n' +
    'K2 = fn(Normal(mu = get(_, 2), sigma = 1.0))\n' +
    'jc = jointchain(p = M, q = K1, r = K2)\n' +
    'lp = logdensityof(jc, record(p = 0.3, q = 0.5, r = 0.9))\n',
    'lp', 2000);
  const got = m.samples[0];
  const want = normLogpdf(0.3, 0, 1) + normLogpdf(0.5, 0.3, 1)
             + normLogpdf(0.9, 0.5, 1);
  assert.ok(Math.abs(got - want) < 1e-6,
    `N-ary jointchain joint logp ${got} vs analytic ${want}`);
});

test('density 2b-ext: positional jointchain joint logdensity (dependent)', async () => {
  // Dependent-positional density now works: jointchain(M,K) variate
  // [a,b]; logp([0.5,1.0]) = logN(0.5;0,1)+logN(1.0;0.5,1), exact —
  // walkJoint positional `args` threads the consumed a under `s0` so
  // the kernel body's rewired ref(s0) scores against the OBSERVED a.
  const m = await materialise(
    'M = Normal(mu = 0.0, sigma = 1.0)\n' +
    'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
    'jc = jointchain(M, K)\n' +
    'lp = logdensityof(jc, [0.5, 1.0])\n', 'lp', 2000);
  const got = m.samples[0];
  const want = normLogpdf(0.5, 0, 1) + normLogpdf(1.0, 0.5, 1);
  assert.ok(Math.abs(got - want) < 1e-6,
    `positional jointchain joint logp ${got} vs analytic ${want}`);
});

test('classifyJointchain: non-kernel later arg → null (parity fallback)', () => {
  // K_i must be a kernel; a measure in kernel position isn't covered
  // by the first-class path → null so the dual-path falls back.
  const cj = orchestrator._internal.classifyJointchain;
  const rhsIR = { kind: 'call', op: 'jointchain',
    args: [{ kind: 'ref', ns: 'self', name: 'M' },
           { kind: 'ref', ns: 'self', name: 'M' }] };
  const ast = { type: 'CallExpr',
    callee: { type: 'Identifier', name: 'jointchain' },
    args: [{ type: 'Identifier', name: 'M' },
           { type: 'Identifier', name: 'M' }] };
  const bindings = processSource(KDEF).bindings;
  assert.equal(cj(rhsIR, ast, bindings), null);
});
