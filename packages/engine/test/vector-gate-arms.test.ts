'use strict';

// The two determiniser gate arms that appear ONLY over a vector variate, scored
// end-to-end from the FlatPDL the determiniser emits:
//
//   * the discrete lattice snap — `iszero(sum(abs.(...)))` over the per-cell
//     differences, with the preimage snapped by `real.(round.(...))`
//     (flatppl-rust determinizer/src/density.rs::snap_to_lattice, lattice_test);
//   * the `cartpow` image gate — `y in cartpow(posreals, n)`, the vector form of
//     the scalar `y in S` (determinizer/src/invert.rs::Image::vector_condition).
//
// Each source below is the verbatim output of
// `flatppl determinize <model> --keep lp` from flatppl-rust main 7f0f446, with
// the model in the comment above it. §07's elementary functions and scalar
// constructors are scalar-only, and the determiniser dots every elementwise
// use, so the emitted text type-checks here as it stands. A variant point is
// spliced into that text in place of the emitted one.

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

function atPoint(src: string, emitted: string, point: string): string {
  assert.ok(src.includes(emitted), 'the emitted point is in the emitted text');
  return src.split(emitted).join(point);
}

// m = pushfwd(neg, Multinomial(n = 5, p = [0.2, 0.3, 0.5]))
// lp = logdensityof(m, [-1.0, -2.0, -2.0])
const NEG_POINT = '[-1.0, -2.0, -2.0]';
const LATTICE_SNAP_ARM = `
lp = ifelse(
  land(
    maximum([-1.0, -2.0, -2.0]) <= 0.0,
    iszero(sum(abs.([-1.0, -2.0, -2.0] - .-real.(round.(.-[-1.0, -2.0, -2.0]))))),
  ),
  builtin_logdensityof(
    Multinomial,
    record(n = 5, p = [0.2, 0.3, 0.5]),
    real.(
      round.(
        neg.(
          ifelse(
            land(
              maximum([-1.0, -2.0, -2.0]) <= 0.0,
              iszero(sum(abs.([-1.0, -2.0, -2.0] - .-real.(round.(.-[-1.0, -2.0, -2.0]))))),
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
  const off = atPoint(LATTICE_SNAP_ARM, NEG_POINT, '[-1.0, -1.5, -2.0]');
  assert.equal(await scoreValue(off, 'lp'), -Infinity);
});

// m = pushfwd(exp, Multinomial(n = 5, p = [0.2, 0.3, 0.5]))
// lp = logdensityof(m, [2.718281828459045, 7.38905609893065, 7.38905609893065])
//
// The same arm through a NON-affine forward, which is where the round-trip test
// earns its keep: `exp.(real.(round.(log.(y))))` maps the whole snapped array
// back through the forward before the per-cell difference is reduced.
const EXP_POINT = '[2.718281828459045, 7.38905609893065, 7.38905609893065]';
const LATTICE_SNAP_EXP_ARM = `
lp = ifelse(
  land(
    minimum([2.718281828459045, 7.38905609893065, 7.38905609893065]) >= 1.0,
    iszero(
      sum(
        abs.(
          [2.718281828459045, 7.38905609893065, 7.38905609893065]
          - exp.(real.(round.(log.([2.718281828459045, 7.38905609893065, 7.38905609893065])))),
        ),
      ),
    ),
  ),
  builtin_logdensityof(
    Multinomial,
    record(n = 5, p = [0.2, 0.3, 0.5]),
    real.(
      round.(
        log.(
          ifelse(
            land(
              minimum([2.718281828459045, 7.38905609893065, 7.38905609893065]) >= 1.0,
              iszero(
                sum(
                  abs.(
                    [2.718281828459045, 7.38905609893065, 7.38905609893065]
                    - exp.(
                      real.(round.(log.([2.718281828459045, 7.38905609893065, 7.38905609893065]))),
                    ),
                  ),
                ),
              ),
            ),
            [2.718281828459045, 7.38905609893065, 7.38905609893065],
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
  const got = await scoreValue(LATTICE_SNAP_EXP_ARM, 'lp');
  assert.ok(Math.abs(got - (-2.002480500543708)) < 1e-12, `${got} vs pmf`);
});

test('the lattice snap through a non-affine forward excludes an off-lattice point', async () => {
  // 5.0 is not `exp` of any integer, so the round-trip does not reproduce it.
  // This is the case that read as a valid pmf while `abs` over an array coerced
  // to NaN and `sum` of that read 0 — the gate admitted everything.
  const off = atPoint(LATTICE_SNAP_EXP_ARM, EXP_POINT, '[2.718281828459045, 5.0, 7.38905609893065]');
  assert.equal(await scoreValue(off, 'lp'), -Infinity);
});

// m = pushfwd(exp, iid(Poisson(2.0), 2))
// lp = logdensityof(m, [2.718281828459045, 7.38905609893065])
//
// The iid form: the snapped preimage feeds a broadcast of the Poisson pmf.
const POIS_POINT = '[2.718281828459045, 7.38905609893065]';
const LATTICE_SNAP_IID_ARM = `
lp = ifelse(
  land(
    minimum([2.718281828459045, 7.38905609893065]) >= 1.0,
    iszero(
      sum(
        abs.(
          [2.718281828459045, 7.38905609893065]
          - exp.(real.(round.(log.([2.718281828459045, 7.38905609893065])))),
        ),
      ),
    ),
  ),
  sum(
    builtin_logdensityof.(
      Poisson,
      broadcast(record, rate = [2.0]),
      real.(
        round.(
          log.(
            ifelse(
              land(
                minimum([2.718281828459045, 7.38905609893065]) >= 1.0,
                iszero(
                  sum(
                    abs.(
                      [2.718281828459045, 7.38905609893065]
                      - exp.(real.(round.(log.([2.718281828459045, 7.38905609893065])))),
                    ),
                  ),
                ),
              ),
              [2.718281828459045, 7.38905609893065],
              fill(exp(1.0), 2),
            ),
          ),
        ),
      ),
    ),
  ),
  -inf,
)
`;

test('the lattice snap over an iid scores the summed pmf at the preimage', async () => {
  // log Poisson(1; 2) + log Poisson(2; 2), from scipy 1.18.0.
  const got = await scoreValue(LATTICE_SNAP_IID_ARM, 'lp');
  assert.ok(Math.abs(got - (-2.613705638880109)) < 1e-12, `${got} vs pmf`);
  const off = atPoint(LATTICE_SNAP_IID_ARM, POIS_POINT, '[2.718281828459045, 5.0]');
  assert.equal(await scoreValue(off, 'lp'), -Infinity);
});

// m = pushfwd(exp, Dirichlet(alpha = [2.0, 3.0, 4.0]))
// lp = logdensityof(m, [1.2214027581601699, 1.3498588075760032, 1.6487212707001282])
//
// The gate is what this test pins. The gated VALUE is the emitted lowering's own
// ambient-Jacobian reading (`- sum(log y)`), and Dirichlet's support is an
// (n-1)-manifold in `cartpow(reals, n)` (§03 `stdsimplex`), for which the
// tangential volume element differs from the ambient one — so the number below
// is "the engine evaluates this expression correctly", NOT "this is the correct
// pushforward density". flatppl-testsuite refuses to derive an oracle for the
// shape for that reason (`sweep/oracle._MANIFOLD_SAFE_FORWARDS`).
const DIR_POINT = '[1.2214027581601699, 1.3498588075760032, 1.6487212707001282]';
const CARTPOW_GATE_ARM = `
lp = ifelse(
  [1.2214027581601699, 1.3498588075760032, 1.6487212707001282] in cartpow(posreals, 3),
  builtin_logdensityof(
    Dirichlet,
    record(alpha = [2.0, 3.0, 4.0]),
    log.([1.2214027581601699, 1.3498588075760032, 1.6487212707001282]),
  )
  - sum((x -> log(x)).([1.2214027581601699, 1.3498588075760032, 1.6487212707001282])),
  -inf,
)
`;

test('the cartpow image gate admits an all-positive point and scores it', async () => {
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
  const got = await scoreValue(CARTPOW_GATE_ARM, 'lp');
  assert.ok(Math.abs(got - expect) < 1e-12, `${got} vs ${expect}`);
});

test('the cartpow image gate excludes a point with a non-positive cell', async () => {
  // `exp` maps onto `posreals` cell-wise, so a zero or negative cell is outside
  // the image and the mass there is zero (log 0 = -inf).
  for (const point of ['[1.2, 0.0, 1.6]', '[1.2, 1.3, -1.6]']) {
    assert.equal(await scoreValue(atPoint(CARTPOW_GATE_ARM, DIR_POINT, point), 'lp'), -Infinity, point);
  }
});
