'use strict';

// §04 always-splat enforcement. Spec §04 "Calling conventions":
//
//   "A sole positional record or table therefore always splats: whether
//    its field or column names match the callable's argument names
//    decides only whether the call is valid, never whether the splat
//    occurs. Passing a record or table as one ordinary argument requires
//    the keyword spelling, as in `f(pars = record(...))`."
//
// and, in the same bullet, "A call with field or column names that do not
// match the callable's argument names is a static error."
//
// Before enforcement the engine splatted only when the record's fields
// covered the callee's inputs and otherwise bound the whole record to
// input 0 with no name check — the name-conditioned reading §04 rules
// out. That fallback made `generator(pars)` against a one-input
// `generator` score a value, and made `Poisson(record(zzz = 0.5))` score
// NaN with no error at all.
//
// INDEPENDENT ORACLE — closed form, NOT an engine output:
//   Normal(mu = 1.1, sigma = 0.2) logpdf at x = 1.1 (the mean) is
//     -log(0.2 · √(2π)) = -log(0.2) - ½log(2π)
//                       = 1.6094379124341003 - 0.9189385332046727
//                       = 0.6904993792294276

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const NORMAL_AT_MEAN = 0.6904993792294276;   // closed form; NOT an engine output

// A kernel with ONE input that is itself a record. Under always-splat the
// only valid call spelling is `generator(pars = <record>)`.
function transportSrc(callSpelling: string) {
  return `sigma = 0.2
pars = elementof(cartprod(a = reals, mu = reals))
x ~ Normal(pars.mu, sigma)
generator = kernelof(x, pars = pars)
gp = record(a = 0.1, mu = 1.1)
m = ${callSpelling}
ld = logdensityof(m, 1.1)`;
}

function errorsOf(src: string) {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function buildCtx(src: string, N: number, seed: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

test('always-splat: a sole positional record against a one-input callable is a §04 error', () => {
  const errs = errorsOf(transportSrc('generator(gp)'));
  const splat = errs.filter((m: string) => /no argument named/.test(m));
  assert.equal(splat.length, 1, 'expected one §04 splat-name error; got: ' + errs.join(' | '));
  // Both fields are reported, and the message names the keyword fix.
  assert.match(splat[0], /"a", "mu"/);
  assert.match(splat[0], /generator\(pars = …\)/);
});

test('always-splat: the invalid positional spelling yields NO value (fallback is gone)', async () => {
  // The whole-record positional fallback used to score this model
  // successfully, which is what hid the spec violation.
  const ctx = buildCtx(transportSrc('generator(gp)'), 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')), /no derivation/);
});

test('always-splat: the keyword spelling scores, matching the closed form', async () => {
  const src = transportSrc('generator(pars = gp)');
  assert.deepEqual(errorsOf(src), [], 'keyword spelling must be diagnostic-free');
  const ctx = buildCtx(src, 1, 1);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - NORMAL_AT_MEAN) < 1e-12,
    `logdensity ${got} ≈ closed form ${NORMAL_AT_MEAN}`);
});

test('always-splat: an inline record literal splats into a two-input kernel', async () => {
  // The other splat source at the call site: `record(...)` written inline
  // rather than referenced through a binding. Same closed-form oracle —
  // splatting a = 0.1 (unused by the body) and mu = 1.1.
  const src = `sigma = 0.2
a = elementof(reals)
mu = elementof(reals)
x ~ Normal(mu, sigma)
k = kernelof(x, a = a, mu = mu)
m = k(record(a = 0.1, mu = 1.1))
ld = logdensityof(m, 1.1)`;
  assert.deepEqual(errorsOf(src), []);
  const got = (await buildCtx(src, 1, 1).getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - NORMAL_AT_MEAN) < 1e-12,
    `logdensity ${got} ≈ closed form ${NORMAL_AT_MEAN}`);
});

test('always-splat: a record whose fields match the inputs still splats', () => {
  // The splat itself is unchanged for a matching record — enforcement
  // removes the *precondition*, not the feature.
  assert.deepEqual(errorsOf(`f = (a, b) -> a + b
r = record(a = 1.0, b = 2.0)
y = f(r)`), []);
});

test('always-splat: a surplus field is a §04 error, not a whole-record bind', () => {
  // Previously the surplus `c` failed the cover test, so the record bound
  // positionally to `a` and the user saw only a misleading
  // 'missing argument "b"'.
  const errs = errorsOf(`f = (a, b) -> a + b
r = record(a = 1.0, b = 2.0, c = 3.0)
y = f(r)`);
  const splat = errs.filter((m: string) => /no argument named/.test(m));
  assert.equal(splat.length, 1, 'got: ' + errs.join(' | '));
  assert.match(splat[0], /"c"/);
});

test('always-splat: a keyword-bound record is an ordinary value, never splatted', () => {
  // §04 excludes it explicitly, so a record bound by keyword to a
  // one-input callable must NOT be name-checked against its fields.
  assert.deepEqual(errorsOf(transportSrc('generator(pars = gp)')), []);
});

// ── §04's single-input carve-out (flatppl-design#78) ───────────────────────
//
//   "A callable with exactly one input whose documented domain admits records
//    or tables is exempt and receives a sole positional record or table whole,
//    so that `sum(t)` and `lengthof(t)` reduce over the table rather than
//    splatting."
//
// BOTH halves of the test matter, and it is decidable from the callable's
// signature alone — never from the caller's field names, which is what #74
// rejected.
//
// The exemption keys on the DOCUMENTED domain, so it reaches §07 builtins only.
// A user callable with one record-domain input — `generator = kernelof(x, pars =
// pars)` over `pars = elementof(cartprod(a = reals, mu = reals))` — is NOT
// exempt and still splats (ruled 2026-08-10, narrow reading; a clarifying
// sentence is going into #78). The resulting builtin/user asymmetry is accepted
// as the cost: the same one-input-one-record-domain shape splats for a user
// callable and does not for `sum`. Both engines take the narrow reading. The
// first test in this file pins the user-callable half.
//
// The exempt set, nine callables, agreed with flatppl-rust:
//   sum, mean, var, std   — table domain from §07's "Table reductions"
//                           paragraph, not their Domains cells (`std` added by
//                           owner ruling, spec commit onto design#77)
//   lengthof, reverse     — Domains cells `vectors, tables`
//   indicesof, indicesof0 — Domains cells `vectors, arrays, tables`
//   identity              — Domains `any`, which admits both
// NOT exempt per §07: `prod`, `sizeof`, `cumsum`, `joinblocks` (arrays only),
// every scalar math builtin, every distribution constructor
// (`Exponential(rate)`'s domain is `reals`, so one input is not enough), and the
// single-input §09 module functions (`lu`, `qr`, `svd`, … take matrices and
// RETURN records). `record` and `table` have variadic NAMED inputs, so "exactly
// one input" excludes them from the carve-out either way.
//
// CAREFUL: that list is what §07 SANCTIONS, not what this engine accepts. The
// engine's table-capable reduction set is SEVEN — `sum`, `mean`, `var`, `std`
// plus `prod` (→ `{mass: 6, pt: 120}`), `minimum` (→ `{mass: 1, pt: 4}`) and
// `maximum` (→ `{mass: 3, pt: 6}`) all reduce column-wise — against the FOUR
// §07 sanctions with design#77. `sizeof(t)` is accepted too and silently yields
// an empty vector though its Domains cell is `vectors, arrays`. The values are
// right for what they compute; the set is over-permissive. So a later carve-out
// implementation must read the exemption set off §07, NEVER off what the engine
// happens to accept, or it silently exempts prod/minimum/maximum/sizeof.
//
// `reverse(t)`, `indicesof(t)`/`indicesof0(t)`, `sizeof(t)`, and §03:153-155's
// `table(r)`/`record(t)` round trip are all covered in
// `test/table-conformance.test.ts` instead of here — none is a splat bug.
//
// None of the exempt builtins reaches this branch's enforcement sites: the
// static check fires only for callables carrying `inputs` (user
// functionof/kernelof/lambdas), and §07 builtins are typed by dedicated
// inference. These tests pin that, so extending the check to builtins later
// cannot silently kill `sum(t)`.
function fixedOf(src: string, name: string) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  return built.fixedValues && built.fixedValues.get(name);
}

test('carve-out: sum(t) receives a two-column table whole and reduces column-wise', () => {
  // §07 "Table reductions": returns a record of per-column reductions. Splatting
  // would instead bind `mass`/`pt` to `sum`'s only input `xs` and fail.
  // Oracle: 1+2+3 = 6, 4+5+6 = 15.
  const src = `t = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
s = sum(t)`;
  assert.deepEqual(errorsOf(src), []);
  assert.deepEqual(fixedOf(src, 's'), { mass: 6, pt: 15 });
});

test('carve-out: mean(t) and var(t) reduce column-wise too', () => {
  // mean [1,2,3] = 2, [4,5,6] = 5. var (n-1) of each = 1.
  const src = `t = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
m = mean(t)
v = var(t)`;
  assert.deepEqual(errorsOf(src), []);
  assert.deepEqual(fixedOf(src, 'm'), { mass: 2, pt: 5 });
  assert.deepEqual(fixedOf(src, 'v'), { mass: 1, pt: 1 });
});

test('carve-out: std(t) reduces column-wise (owner ruling, design#77)', () => {
  // std joins §07's Table reductions by owner ruling. Deliberately NOT the
  // fixture above: with [1,2,3] the variance is 1 and √1 = 1, so var and std
  // are indistinguishable and the test could not tell std from a mis-wire to
  // var. Here mass = [1,3,5] has var 4 / std 2 and pt = [10,20,30] has var 100
  // / std 10, so the two reductions are separated.
  const src = `t = table(mass = [1.0, 3.0, 5.0], pt = [10.0, 20.0, 30.0])
v = var(t)
d = std(t)`;
  assert.deepEqual(errorsOf(src), []);
  assert.deepEqual(fixedOf(src, 'v'), { mass: 4, pt: 100 });
  assert.deepEqual(fixedOf(src, 'd'), { mass: 2, pt: 10 });
});

test('carve-out: lengthof(t) counts rows rather than splatting', () => {
  // §07 Domains cell names tables outright; §03: "lengthof(t) returns the
  // number of table rows."
  const src = `t = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
n = lengthof(t)`;
  assert.deepEqual(errorsOf(src), []);
  assert.equal(fixedOf(src, 'n'), 3);
});

test('carve-out: identity(t) receives the table whole (Domains `any`)', () => {
  const src = `t = table(mass = [1.0, 2.0], pt = [3.0, 4.0])
u = identity(t)`;
  assert.deepEqual(errorsOf(src), []);
  const u: any = fixedOf(src, 'u');
  assert.equal(u && u.__table__, true, 'identity must yield the table itself');
  assert.deepEqual(Object.keys(u.columns), ['mass', 'pt']);
});

test('carve-out: a single-input constructor whose domain is reals STILL splats', async () => {
  // #78's own row: `Exponential(record(rate = 1.0))` has one input, but its
  // domain admits no records, so the exemption does not reach it. The splat
  // fires and binds `rate` by name.
  // ORACLE, closed form: Exponential(rate = 2) logpdf at 1 = log 2 - 2·1
  //   = 0.6931471805599453 - 2 = -1.3068528194400546
  const ORACLE = Math.log(2) - 2;
  const got = (await buildCtx(
    `ld = logdensityof(Exponential(record(rate = 2.0)), 1.0)`, 1, 1).getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - ORACLE) < 1e-12, `${got} ≈ ${ORACLE}`);
});

test('always-splat: a record alongside another argument is an ordinary value', () => {
  // §04 excludes this too — the record is not the call's SOLE argument,
  // so `restrict(M, x)`'s observation record is never splatted.
  assert.deepEqual(errorsOf(`mu = joint(a = Normal(0.0, 1.0), b = Normal(0.0, 1.0))
x = record(a = 0.5)
nu = restrict(mu, x)`), []);
});

// ── §04 "of records and table columns" — tables splat too ──────────────────
//
// Every splat site above was written and tested against records only; §04's
// own wording ("Auto-splatting (of records and table columns)") makes a
// table's columns splat identically. A table column is a VECTOR (spec §03),
// so the natural target is a callable whose inputs are themselves array-
// typed — `f = (a, b) -> a + b` below adds elementwise, unlike the scalar
// kernels/distributions the record tests above use.

const registry = require(ENG + 'sampler-registry.ts');
const asArr = (v: any) => Array.from(v && v.data !== undefined ? v.data : v);

test('tables splat: an inline table(...) literal splats into a two-input lambda', () => {
  const fv = orchestrator.buildDerivations(processSource(`f = (a, b) -> a + b
y = f(table(a = [1.0, 2.0], b = [3.0, 4.0]))`).bindings).fixedValues;
  assert.deepEqual(asArr(fv.get('y')), [4, 6]);
});

test('tables splat: a table-typed identifier ref splats the same way', () => {
  const src = `f = (a, b) -> a + b
t = table(a = [1.0, 2.0], b = [3.0, 4.0])
y = f(t)`;
  assert.deepEqual(errorsOf(src), []);
  const fv = orchestrator.buildDerivations(processSource(src).bindings).fixedValues;
  assert.deepEqual(asArr(fv.get('y')), [4, 6]);
});

test('tables splat: a column name with no matching input is a §04 error, not a whole-table bind', () => {
  // Mirrors the record surplus-field test above, with a table column instead.
  const errs = errorsOf(`f = (a, b) -> a + b
t = table(a = [1.0, 2.0], c = [3.0, 4.0])
y = f(t)`);
  const splat = errs.filter((m: string) => /no argument named/.test(m));
  assert.equal(splat.length, 1, 'got: ' + errs.join(' | '));
  assert.match(splat[0], /"c"/);
  assert.match(splat[0], /positional table/);
});

test('tables splat: resolveParams splats an inline table(...) into a distribution, like record(...)', () => {
  // Unit-level: exercises sampler-registry.resolveParams directly (the third
  // of the three splat sites), sidestepping the arity-1 whole-variate feed
  // that a single-parameter array-typed distribution (e.g. Dirichlet) hits
  // before reaching resolveParams — tracked separately in TODO-flatppl-js.md
  // as the deliberate C2/C3 arity-1 design, not a splat gap.
  const lit = (v: number) => ({ kind: 'lit', value: v });
  const measureIR = {
    kind: 'call', op: 'Normal',
    args: [{ kind: 'call', op: 'table', fields: [
      { name: 'mu', value: lit(1.1) },
      { name: 'sigma', value: lit(0.2) },
    ] }],
    kwargs: {},
  };
  const out = registry.resolveParams(measureIR, registry.REGISTRY.Normal, {});
  assert.deepEqual(out, [1.1, 0.2]);
});

test('tables splat: resolveParams rejects a table column with no matching parameter', () => {
  const lit = (v: number) => ({ kind: 'lit', value: v });
  const measureIR = {
    kind: 'call', op: 'Normal',
    args: [{ kind: 'call', op: 'table', fields: [
      { name: 'mu', value: lit(1.1) },
      { name: 'sigma', value: lit(0.2) },
      { name: 'zz', value: lit(9.0) },
    ] }],
    kwargs: {},
  };
  assert.throws(() => registry.resolveParams(measureIR, registry.REGISTRY.Normal, {}),
    /'Normal' has no parameter 'zz'/);
});
