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
