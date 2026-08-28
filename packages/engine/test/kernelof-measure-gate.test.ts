'use strict';

// =====================================================================
// kernelof-measure-gate.test.ts — `kernelof` refuses a measure (§04)
// =====================================================================
//
// What grounds it. `flatppl-design` `docs/04-design.md`, "Kernels and
// `kernelof`":
//
//   "`kernelof(x, kwargs...)` reifies (typically stochastic) value nodes to
//    Markov kernels. `x` must not be a measure."
//
// The refusal costs no expressiveness. §04 "Reifying measure-valued
// expressions to kernels" already spells the measure case: "If `functionof`
// is applied to a measure node, it generates a transition kernel", with
// `functionof(m, …)` the conditional kernel and `functionof(lawof(m), …)`
// the marginal for a stochastic-phase `m`. So `kernelof` over a measure is
// a second spelling of something `functionof` says, and §04 keeps one.
//
// Why the check is explicit rather than emergent. `kernelof(x, kw)` lowers
// to `functionof(lawof(x), kw)`, and §04 "Reification to measures" gives
// `lawof` its own identity law on a measure argument — "`lawof` also accepts
// a measure argument: `lawof(m)` is `lawof(draw(m))`". `inferLawof`
// implements that, so the lowered form typed CLEAN and the illegal spelling
// was silently accepted. `inferReification` therefore reads the
// pre-lowering argument off `wasKernelof`.
//
// The wording matches flatppl-rust's `crates/infer/src/ops.rs` refusal, so
// the two engines report the same diagnostic on the same surface (both at
// inference; neither parser rejects it).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

const H = 'flatppl_compat = "0.1"\n';

function errorsOf(src: string): any[] {
  return (processSource(H + src + '\n').diagnostics || [])
    .filter((d: any) => d.severity === 'error');
}

function kernelofErrors(src: string): string[] {
  return errorsOf(src)
    .map((d: any) => d.message)
    .filter((m: string) => m.startsWith('`kernelof` reifies value nodes'));
}

// ---------------------------------------------------------------------
// Refused: a measure-layer argument
// ---------------------------------------------------------------------

test('§04: kernelof of an inline distribution call is refused', () => {
  const errs = kernelofErrors(
    'K = kernelof(Normal(mu = _p_, sigma = 1.0), p = _p_)');
  assert.equal(errs.length, 1);
  assert.match(errs[0], /this argument is a measure/);
  assert.match(errs[0], /use `functionof` to reify a measure node/);
});

test('§04: kernelof of a measure BINDING is refused', () => {
  const src = 'mu = elementof(reals)\n'
    + `m = Normal(mu = mu, sigma = 1.0)\n`
    + 'K = kernelof(m, mu = mu)';
  assert.equal(kernelofErrors(src).length, 1);
});

test('§04: the refusal does not need a boundary specification', () => {
  const src = 'mu = elementof(reals)\n'
    + 'm = Normal(mu = mu, sigma = 1.0)\n'
    + 'K = kernelof(m)';
  assert.equal(kernelofErrors(src).length, 1);
});

test('§04: a composite measure body is refused too', () => {
  const src = 'mu = elementof(reals)\n'
    + 'K = kernelof(iid(Normal(mu = mu, sigma = 1.0), 5), mu = mu)';
  assert.equal(kernelofErrors(src).length, 1);
});

test('§04: the diagnostic locates the offending argument', () => {
  const errs = errorsOf('K = kernelof(Normal(mu = _p_, sigma = 1.0), p = _p_)');
  assert.equal(errs.length, 1);
  assert.ok(errs[0].loc && errs[0].loc.start,
    'the refusal must carry the argument location');
});

// A KERNEL argument fails the same clause one step earlier. It gets its own
// wording so the diagnostic does not tell a user their kernel is a measure.
test('§04: a kernel argument is refused, and is not called a measure', () => {
  const src = 'mu = elementof(reals)\n'
    + 'Kin = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
    + 'K = kernelof(Kin, mu = mu)';
  const errs = kernelofErrors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /this argument is a kernel/);
  assert.match(errs[0], /a kernel is already a reified law/);
});

// ---------------------------------------------------------------------
// Accepted: a value argument — the only thing §04 admits
// ---------------------------------------------------------------------

test('§04: kernelof of a VARIATE stays accepted', () => {
  const src = 'mu = elementof(reals)\n'
    + 'x ~ Normal(mu = mu, sigma = 1.0)\n'
    + 'K = kernelof(x, mu = mu)';
  assert.deepEqual(kernelofErrors(src), []);
});

test('§04: kernelof of a RECORD of variates stays accepted', () => {
  // The overwhelmingly common corpus spelling (`kernelof(record(obs = obs),
  // theta = theta)`), and the one §04's Bayesian example uses.
  const src = 'theta = elementof(reals)\n'
    + 'obs ~ Normal(mu = theta, sigma = 1.0)\n'
    + 'K = kernelof(record(obs = obs), theta = theta)';
  assert.deepEqual(kernelofErrors(src), []);
});

// ---------------------------------------------------------------------
// Refused: a BROADCAST of a measure-producing head
// ---------------------------------------------------------------------
//
// §04 "Broadcasting" makes broadcast dual: "`broadcast(function, ...)`
// returns an array value" while "`broadcast(kernel, ...)` returns an
// array-valued measure". The measure half is a measure like any other, so
// §sec:kernelof refuses it — but the refusal reads the ARGUMENT's type, and
// the keyword spelling had no type rule at all, so this shape slipped past
// the gate while the positional spelling was refused. One fixture
// (`nested-broadcast-mvnormal-inner.flatppl`) was written that way.

test('§04: kernelof of a keyword-bound distribution broadcast is refused', () => {
  const src = 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'K = kernelof(broadcast(Normal, mu = A, sigma = mu), mu = mu)';
  const errs = kernelofErrors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /this argument is a measure/);
});

test('§04: the dot spelling of the same broadcast is refused', () => {
  const src = 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'K = kernelof(Normal.(mu = A, sigma = mu), mu = mu)';
  assert.equal(kernelofErrors(src).length, 1);
});

// A broadcast whose head is measure-producing is an array-valued measure by
// §04 whatever its data arguments do, so the refusal cannot depend on the
// type resolving. Here the keyword names no parameter of `Kin` (whose only
// input is the placeholder's `arg1`), which leaves the broadcast untyped.
test('§04: an UNTYPED broadcast of a kernel head is refused too', () => {
  const src = 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'Kin = fn(Normal(mu = _, sigma = 0.1))\n'
    + 'K = kernelof(broadcast(Kin, mu = A), mu = mu)';
  const errs = kernelofErrors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /this argument is a measure/);
});

test('§04: an INLINE lambda head over a distribution is refused', () => {
  // The head is an inline reification rather than a ref, and the record
  // argument (§04 "Disallowed inputs") leaves the broadcast with no type at
  // all — the head still decides.
  const src = 'mu = elementof(reals)\n'
    + 'r = record(a = 1.0)\n'
    + 'K = kernelof(fn(Normal(mu = _, sigma = 0.1)).(r), mu = mu)';
  assert.equal(kernelofErrors(src).length, 1);
});

test('§04: a BARE distribution head is refused where the type declines', () => {
  // The record argument (§04 "Disallowed inputs") leaves the broadcast with no
  // type, and the head is a bare builtin with no module binding to read a
  // kind off. §04's dual still answers: a distribution constructor is a
  // kernel, so the broadcast is a measure.
  const src = 'mu = elementof(reals)\n'
    + 'r = record(a = 1.0)\n'
    + 'K = kernelof(broadcast(Normal, mu = r, sigma = mu), mu = mu)';
  const errs = kernelofErrors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /this argument is a measure/);
});

test('§04: a MvNormal broadcast under kernelof is refused', () => {
  // The fixture shape. A vector-output head over a rank-2 argument types as a
  // measure here — the same answer the positional spelling already gave — so
  // the refusal comes from the type, not from the structural fallback.
  const src = 'mu = elementof(reals)\n'
    + 'mus = rowstack([[0.0, 0.0], [1.0, -1.0]])\n'
    + 'cov = rowstack([[1.0, 0.3], [0.3, 0.5]])\n'
    + 'K = kernelof(broadcast(MvNormal, mu = mus, cov = cov), mu = mu)';
  assert.equal(kernelofErrors(src).length, 1);
});

// ---------------------------------------------------------------------
// Accepted: a VALUE broadcast, and the functionof replacement
// ---------------------------------------------------------------------

test('§04: kernelof of a value broadcast stays accepted', () => {
  // `broadcast(<function>, …)` is an array VALUE — exactly what §sec:kernelof
  // reifies. The dual must not over-refuse.
  const src = 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'x ~ Normal(mu = mu, sigma = 1.0)\n'
    + 'K = kernelof(broadcast(add, A, x), mu = mu)';
  assert.deepEqual(kernelofErrors(src), []);
});

test('§04: kernelof of a value broadcast with a non-ref head stays accepted', () => {
  // Neither head produces a measure, so neither reaches the refusal: a
  // callable-valued CALL (`fchain`) and a cross-module function.
  const chained = 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'f = functionof(2.0 * _a_, a = _a_)\n'
    + 'g = functionof(_b_ + 1.0, b = _b_)\n'
    + 'K = kernelof(broadcast(fchain(f, g), a = A), mu = mu)';
  assert.deepEqual(kernelofErrors(chained), []);
  const crossModule = 'mu = elementof(reals)\n'
    + 'poly = standard_module("polynomials", "0.1")\n'
    + 'xs = [0.1, 0.2, 0.3]\n'
    + 'K = kernelof(broadcast(poly.legendre, n = 3, x = xs), mu = mu)';
  assert.deepEqual(kernelofErrors(crossModule), []);
});

test('§04: functionof over the same broadcast is accepted, and is a kernel', () => {
  const { processSource: ps } = require('../index.ts');
  const src = H + 'mu = elementof(reals)\n'
    + 'A = [1.0, 2.0, 3.0]\n'
    + 'K = functionof(broadcast(Normal, mu = A, sigma = mu), mu = mu)\n';
  const { bindings, diagnostics } = ps(src);
  assert.deepEqual((diagnostics || [])
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  assert.equal(bindings.get('K').inferredType.kind, 'kernel');
});

// ---------------------------------------------------------------------
// The spec-legal replacement keeps the composite recognisers
// ---------------------------------------------------------------------
//
// The recognisers used to require the `lawof` wrapper that only a `kernelof`
// produces, so migrating a fixture to `functionof` dropped it off the fusion
// path onto a per-atom `evaluate`. Both now peel an OPTIONAL `lawof`.

test('§04: functionof over a dist body still fuses as a kernel-broadcast', () => {
  const { _tryDissolveKernelBroadcast } = require('../dissolver.ts');
  const kIR = {
    kind: 'call', op: 'functionof',
    params: ['m'], paramKwargs: ['m'],
    body: { kind: 'call', op: 'Normal',
      kwargs: { mu: { kind: 'ref', ns: '%local', name: 'm' },
        sigma: { kind: 'lit', value: 1.0 } } },
  };
  const bindings = new Map([['nk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: 'nk' }],
    kwargs: { m: { kind: 'ref', ns: 'self', name: 'M' } },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.ok(r, 'the bare-measure body must fuse like the lawof-wrapped one');
  assert.equal(r.op, 'broadcast');
  assert.equal(r.args[0].name, 'Normal');
  assert.equal(r.kwargs.mu.name, 'M');
  assert.equal(r.kwargs.sigma.value, 1.0);
});
