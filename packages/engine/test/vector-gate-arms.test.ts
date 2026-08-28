'use strict';

// The two determiniser gate arms that appear ONLY over a vector variate, scored
// end-to-end from the FlatPDL the determiniser actually emits:
//
//   * the discrete lattice snap — `iszero(sum(abs.(...)))` over the per-cell
//     differences, with the preimage snapped by `real(round.(v))`
//     (flatppl-rust determinizer/src/density.rs::snap_to_lattice, lattice_test);
//   * the `cartpow` image gate — `y in cartpow(posreals, n)`, the vector form of
//     the scalar `y in S` (determinizer/src/invert.rs::Image::vector_condition).
//
// Both were unscoreable: `real` rejected an integer array and `in` had no
// `cartpow` branch, so no vector discrete pushforward and no vector set-valued
// image could be evaluated at all.
//
// The sources below are the emitted text with `abs` and `exp` DOTTED. §07's
// elementary functions are scalar-only, so the un-dotted array spellings
// flatppl-rust currently emits (`sum(abs.(v - …))` in lattice_test, `exp(…)` in
// the round-trip through a non-affine forward) are static errors here. The
// determiniser must emit `abs.` / `exp.`; until it does, these arms are the
// dotted form the spec allows, not the text the determiniser produces. See
// TODO-flatppl-js.md.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function scoreValue(src: string, name: string): Promise<any> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure(name);
  if (m && m.value && m.value.data) return m.value.data[0];
  if (m && m.samples && m.samples.length) return m.samples[0];
  throw new Error(`no value for binding '${name}'`);
}

// `flatppl determinize` of
//   m = pushfwd(neg, Multinomial(n = 5, p = [0.2, 0.3, 0.5]))
//   lp = logdensityof(m, [-1.0, -2.0, -2.0])
const LATTICE_SNAP_ARM = `
m = 0.0
lp = ifelse(
  land(
    maximum([-1.0, -2.0, -2.0]) <= 0.0,
    iszero(sum(abs.([-1.0, -2.0, -2.0] - -real(round.(.-[-1.0, -2.0, -2.0]))))),
  ),
  builtin_logdensityof(
    Multinomial,
    record(n = 5, p = [0.2, 0.3, 0.5]),
    real(
      round.(
        neg.(
          ifelse(
            land(
              maximum([-1.0, -2.0, -2.0]) <= 0.0,
              iszero(sum(abs.([-1.0, -2.0, -2.0] - -real(round.(.-[-1.0, -2.0, -2.0]))))),
            ),
            [-1.0, -2.0, -2.0],
            fill(-1.0, 3),
          ),
        ),
      ),
    ),
  ),
  -inf,
)
`;

test('the vector lattice snap scores the Multinomial pmf at the preimage', async () => {
  // §08 gives Multinomial's density w.r.t. `iid(Counting(integers), k)`, and a
  // bijection does not distort a counting measure, so the pushforward's mass at
  // the image point is the base pmf at the preimage:
  //   log(5! / (1! 2! 2!)) + 1·log 0.2 + 2·log 0.3 + 2·log 0.5.
  // Independently: Distributions.jl `logpdf(Multinomial(5, [0.2,0.3,0.5]),
  // [1,2,2])` = -2.002480500543708.
  const lgamma = (n: number) => {
    let acc = 0;
    for (let i = 2; i < n; i++) acc += Math.log(i);
    return acc;                                   // log((n-1)!) for integer n
  };
  const expect = lgamma(6) - lgamma(2) - lgamma(3) - lgamma(3)
    + Math.log(0.2) + 2 * Math.log(0.3) + 2 * Math.log(0.5);
  assert.ok(Math.abs(expect - (-2.002480500543708)) < 1e-12,
    'closed form disagrees with the Distributions.jl value: ' + expect);
  const got = await scoreValue(LATTICE_SNAP_ARM, 'lp');
  assert.ok(Math.abs(got - expect) < 1e-12, `${got} vs ${expect}`);
});

test('the lattice snap returns -inf off the lattice', async () => {
  // `-1.5` is not the image of any integer under `neg`, so the round-trip
  // through the snap does not reproduce it and the gate must exclude the point.
  const off = LATTICE_SNAP_ARM.replace(/-2\.0, -2\.0/g, '-1.5, -2.0');
  assert.equal(await scoreValue(off, 'lp'), -Infinity);
});

// `flatppl determinize` of
//   m = pushfwd(exp, Multinomial(n = 5, p = [0.2, 0.3, 0.5]))
//   lp = logdensityof(m, [e^1, e^2, e^2])
//
// The same arm through a NON-affine forward, which is where the round-trip test
// earns its keep: `exp.(real(round.(log.(y))))` maps the whole snapped array
// back through the forward before the per-cell difference is reduced.
const LATTICE_SNAP_EXP_ARM = (point: string) => `
m = 0.0
lp = ifelse(
  land(
    minimum(${point}) >= 1.0,
    iszero(sum(abs.(${point} - exp.(real(round.(log.(${point}))))))),
  ),
  builtin_logdensityof(
    Multinomial,
    record(n = 5, p = [0.2, 0.3, 0.5]),
    real(
      round.(
        log.(
          ifelse(
            land(
              minimum(${point}) >= 1.0,
              iszero(sum(abs.(${point} - exp.(real(round.(log.(${point})))))))
            ),
            ${point},
            fill(exp(1.0), 3),
          ),
        ),
      ),
    ),
  ),
  -inf,
)
`;

test('the lattice snap through a non-affine forward scores the same pmf', async () => {
  // A bijection does not distort a counting measure, so the mass at the image
  // point is the pmf at the preimage (1, 2, 2) — the same value the `neg` arm
  // gives. Distributions.jl: -2.002480500543708.
  const point = '[2.718281828459045, 7.38905609893065, 7.38905609893065]';
  const got = await scoreValue(LATTICE_SNAP_EXP_ARM(point), 'lp');
  assert.ok(Math.abs(got - (-2.002480500543708)) < 1e-12, `${got} vs pmf`);
});

test('the lattice snap through a non-affine forward excludes an off-lattice point', async () => {
  // 5.0 is not `exp` of any integer, so the round-trip does not reproduce it.
  // This is the case that read as a valid pmf while `abs` over an array coerced
  // to NaN and `sum` of that read 0 — the gate admitted everything.
  const got = await scoreValue(
    LATTICE_SNAP_EXP_ARM('[2.718281828459045, 5.0, 7.38905609893065]'), 'lp');
  assert.equal(got, -Infinity);
});

// `flatppl determinize` of
//   m = pushfwd(exp, Dirichlet(alpha = [2.0, 3.0, 4.0]))
//   lp = logdensityof(m, [<exp of each cell of (0.2, 0.3, 0.5)>])
//
// The gate is what this test pins. The gated VALUE is the emitted lowering's own
// ambient-Jacobian reading (`- sum(log y)`), and Dirichlet's support is an
// (n-1)-manifold in `cartpow(reals, n)` (§03 `stdsimplex`), for which the
// tangential volume element differs from the ambient one — so the number below
// is "the engine evaluates this expression correctly", NOT "this is the correct
// pushforward density". flatppl-testsuite refuses to derive an oracle for the
// shape for that reason (`sweep/oracle._MANIFOLD_SAFE_FORWARDS`).
const CARTPOW_GATE_ARM = (point: string) => `
m = 0.0
lp = ifelse(
  ${point} in cartpow(posreals, 3),
  builtin_logdensityof(Dirichlet, record(alpha = [2.0, 3.0, 4.0]), log.(${point}))
  - sum((x -> x).(log.(${point}))),
  -inf,
)
`;

test('the cartpow image gate admits an all-positive point and scores it', async () => {
  const point = '[1.2214027581601699, 1.3498588075760032, 1.6487212707001282]';
  const x = [0.2, 0.3, 0.5];                      // log of each cell
  const alpha = [2.0, 3.0, 4.0];
  // §08 Dirichlet: log Γ(Σα) − Σ log Γ(α_i) + Σ (α_i − 1) log x_i.
  const lgammaInt = (n: number) => {
    let acc = 0;
    for (let i = 2; i < n; i++) acc += Math.log(i);
    return acc;
  };
  let expect = lgammaInt(9) - lgammaInt(2) - lgammaInt(3) - lgammaInt(4);
  for (let i = 0; i < 3; i++) expect += (alpha[i] - 1) * Math.log(x[i]);
  expect -= x[0] + x[1] + x[2];                   // the emitted `- sum(log y)`
  const got = await scoreValue(CARTPOW_GATE_ARM(point), 'lp');
  assert.ok(Math.abs(got - expect) < 1e-12, `${got} vs ${expect}`);
});

test('the cartpow image gate excludes a point with a non-positive cell', async () => {
  // `exp` maps onto `posreals` cell-wise, so a zero or negative cell is outside
  // the image and the mass there is zero (log 0 = -inf).
  assert.equal(await scoreValue(CARTPOW_GATE_ARM('[1.2, 0.0, 1.6]'), 'lp'), -Infinity);
  assert.equal(await scoreValue(CARTPOW_GATE_ARM('[1.2, 1.3, -1.6]'), 'lp'), -Infinity);
});
