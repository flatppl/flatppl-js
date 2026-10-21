'use strict';

// Static diagnostics for three malformed shapes, each located on the
// offending binding:
//   - a binding named `self` or `base`;
//   - a callback whose input count differs from the fixed call shape;
//   - a lambda or reification whose body reaches an `elementof` its
//     boundary does not list.
// Each refusing case has a well-formed control that analyses with no error and
// scores against a closed form worked by hand.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function errors(src: string): any[] {
  return (processSource(src).diagnostics || []).filter((d: any) => d.severity === 'error');
}

async function scoreScalar(ctx: any, name: string): Promise<number> {
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}

// §04 "Name resolution": "`self` and `base` themselves are reserved names and
// cannot be bound to something else."
test('self and base cannot be bound, with a diagnostic on the binding', () => {
  const cases: Array<[string, string, number]> = [
    ['self = 1.0', 'self', 0],
    ['base = 1.0', 'base', 0],
    ['a = 2.0\nself ~ Normal(0.0, 1.0)', 'self', 1],
    ['a = 2.0\nbase(x) = x', 'base', 1],
  ];
  for (const [src, name, line] of cases) {
    const hit = errors(src).find((d: any) => /reserved name/.test(d.message));
    assert.ok(hit, `${JSON.stringify(src)} is refused`);
    assert.ok(hit.message.includes(`'${name}'`), hit.message);
    assert.equal(hit.loc.start.line, line, `${JSON.stringify(src)} locates the binding`);
    assert.equal(hit.loc.start.col, 0);
  }
});

test('self and base stay usable as module references; look-alike names bind', () => {
  assert.deepEqual(errors('selfish = 1.0\nbase_measure = Normal(0.0, 1.0)\nb = self.selfish'), []);
});

// §04 "Explicit callback parameters": "Without `p`, that argument is omitted."
// The table gives `f(x, p)` for the whole variate `x` of `weighted(f, M, p)`,
// `logweighted(f, M, p)` and `pushfwd(f, M, p)`.
test('a two-input callback over a whole variate is refused statically', () => {
  const M = 'M = Uniform(interval(0.0, 1.0))\nw2 = (x, y) -> x * y\n';
  for (const op of ['weighted', 'logweighted', 'pushfwd']) {
    const errs = errors(`${M}m2 = ${op}(w2, M)`);
    assert.equal(errs.length, 1, `${op}: ${errs.map((e: any) => e.message).join('; ')}`);
    assert.match(errs[0].message,
      new RegExp(`${op}: callback receives 1 whole inputs, but declares 2`));
    assert.equal(errs[0].loc.start.line, 2, `${op} is located on the m2 binding`);
  }
});

// The same table: `reduce(f, xs, p)` calls `f(acc, next, p)` and
// `kscan(kernel, init, xs, p)` calls `kernel(state, xs[i], p)`.
test('collection and chain callbacks of the wrong input count are refused', () => {
  const cases: Array<[string, RegExp]> = [
    ['f = (a, b, c) -> a + b\nr = reduce(f, [1.0, 2.0, 3.0])',
      /reduce: callback receives 2 whole inputs, but declares 3/],
    ['k = x -> Normal(x, 1.0)\nc = kscan(k, 0.0, [1.0, 2.0])',
      /kscan: callback receives 2 whole inputs, but declares 1/],
    ['k = (x, t) -> Normal(x, 1.0)\nc = markovchain(k, 0.0, 3)',
      /markovchain: callback receives 1 whole inputs, but declares 2/],
    ['f = (a, b) -> a + b\nr = filter(f, [1.0, 2.0])',
      /filter: callback receives 1 whole inputs, but declares 2/],
  ];
  for (const [src, want] of cases) {
    const errs = errors(src);
    assert.equal(errs.length, 1, `${src}: ${errs.map((e: any) => e.message).join('; ')}`);
    assert.match(errs[0].message, want);
    assert.equal(errs[0].loc.start.line, 1, 'located on the call binding');
  }
});

test('the matching call shapes analyse and score the closed form', async () => {
  // Uniform(0, 1) has density 1, so the weighted density is the weight itself.
  const { ctx } = makeMatCtx(`
M = Uniform(interval(0.0, 1.0))
w1 = x -> x * 3.0
lp_whole = logdensityof(weighted(w1, M), 0.5)
wp = (x, p) -> x * p
lp_param = logdensityof(weighted(wp, M, 3.0), 0.5)
m_push = pushfwd(x -> 2.0 * x, M)
lp_push = logdensityof(m_push, 1.0)
`, { sampleCount: 8 });
  // log(0.5 * 3) for both weights; pushfwd of Uniform(0, 1) by 2x has density 1/2.
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp_whole') - Math.log(1.5)) < 1e-12);
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp_param') - Math.log(1.5)) < 1e-12);
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp_push') + Math.log(2)) < 1e-12);
});

// §04 "Reification to functions and kernels": "With boundary specifications,
// the boundary inputs must be the only `elementof` leaves of the subgraph."
// A lambda is `functionof` with its arguments as the boundary.
test('a lambda that reaches an unlisted elementof is refused on its binding', () => {
  const src = `D = elementof(posreals)
f_step = (x, dt) -> Normal(x, sqrt(2.0 * D * dt))`;
  const errs = errors(src);
  assert.equal(errs.length, 1, errs.map((e: any) => e.message).join('; '));
  assert.match(errs[0].message, /Explicit reification boundaries leave an elementof input unbound/);
  assert.equal(errs[0].loc.start.line, 1, 'located on the f_step binding');
});

test('functionof with an incomplete boundary is refused the same way', () => {
  const errs = errors(`D = elementof(posreals)
a = elementof(reals)
g = functionof(sqrt(D * a), x = a)`);
  assert.equal(errs.length, 1, errs.map((e: any) => e.message).join('; '));
  assert.match(errs[0].message, /elementof input unbound/);
  assert.equal(errs[0].loc.start.line, 2);
});

test('a lambda that lists every input analyses and scores the closed form', async () => {
  // The unused top-level `D` is not reached by the lambda, so it is legal.
  const { ctx } = makeMatCtx(`
D = elementof(posreals)
f_step = (x, dt, s) -> Normal(x, sqrt(2.0 * s * dt))
M = f_step(0.1, 0.02, 0.5)
lp = logdensityof(M, 0.3)
`, { sampleCount: 8 });
  // Normal(mu = 0.1, sigma^2 = 2 * 0.5 * 0.02 = 0.02) at 0.3.
  const var2 = 0.02;
  const want = -0.5 * Math.log(2 * Math.PI * var2) - (0.3 - 0.1) ** 2 / (2 * var2);
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp') - want) < 1e-12);
});
