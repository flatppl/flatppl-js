# @flatppl/engine

The JavaScript reference engine for [FlatPPL](https://github.com/flatppl/flatppl-design),
the Flat Portable Probabilistic Language.

> Note: The FlatPPL JavaScript engine is in early development and may be
> unstable. It is not yet published to the npm registry; it currently
> ships only as part of the [`flatppl-js`](https://github.com/flatppl/flatppl-js)
> monorepo and is consumed via npm workspace symlinks by sibling packages.

The engine uses `@stdlib` distribution and random-sampling packages at runtime.
The current workspace runs TypeScript sources directly. Package publication
still requires a complete JavaScript build and a consumer installation check.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models. The language is host-language-neutral; this package is
one implementation among several planned across language ecosystems (Rust,
Julia, Python, …). Cross-implementation conformance is anchored by
[`flatppl-design`](https://github.com/flatppl/flatppl-design) (the spec) and
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples) (the shared
example suite).

## Analytic exponential tilting

Affine log weights normalize analytically for `Normal`, `Gamma` (shape/rate),
`Exponential`, `ChiSquared`, `Bernoulli`, `Binomial`, `Poisson`, `Geometric`,
`NegativeBinomial` (alpha/beta), `NegativeBinomial2` (mu/psi), and `MvNormal`.
The resulting law supports density evaluation, materialisation, and explicit
RNG sampling through the ordinary distribution routes. For example:

```flatppl
theta = elementof(reals)
B = Normal(0, 1)
w = fn(theta * _ + 710)
T = normalize(logweighted(w, B))
U = tilt(logweighted(w, Lebesgue(reals)), B)
lp = logdensityof(T, 0.4)
f = functionof(lp, theta = theta)
score = f(0.3)
P = functionof(T, theta = theta)
x, next_state = rand(P(0.3), rnginit(42))
```

Here both `T` and `U` are `Normal(theta, 1)`: a finite additive constant in
its log weight cancels. Parameters and aliases can remain inferred, and
successive tilts compose. For vectors, use `fn(sum(theta .* _))`; this shifts
an `MvNormal` mean by `cov * theta` and retains its covariance. The
`transpose(theta) * _` spelling still has an inherited type-inference gap.

For equivalent `tilt` forms, the source reference must have unit density on
the base's support: `Lebesgue(reals)`, `Lebesgue(nonnegreals)` for the positive
scalar families, `Lebesgue(cartpow(reals, D))` for vectors, or
`Counting(integers)`/`Counting(nonnegintegers)` for counts (`Counting(booleans)`
also works for Bernoulli). Restricted or nonconstant references retain the
generic route. In particular `Lebesgue(posreals)` is not erased, because its
zero density at zero can change point scoring.

Original parameters, the slope and the offset must be finite and valid.
The MGF domain requires `theta < rate` for Gamma/Exponential, `theta < 1/2`
for ChiSquared, `theta < -log1p(-p)` for Geometric, `theta < log1p(beta)` for
NegativeBinomial, and `theta < log1p(psi/mu)` for NegativeBinomial2. Runtime
guards reject invalid laws or divergent tilts before sampling. Transformed
locations/rates that cannot be represented as finite numbers also refuse;
this is not an arbitrary-weight normalization algorithm.

Scoring retains log odds and log parameters when sampler parameters round
to endpoints or underflow. For example a Poisson rate of one tilted by
`theta = -1000` still scores one event as `-1000`. Repeated tilts retain this
information. Sampling uses finite-precision probabilities, so events below
representable resolution can collapse to a zero draw even though their log
scores remain finite. Overflowing NegativeBinomial beta uses its retained
log parameter when sampling rather than changing the shape parameter.

## Modules

- [`tokenizer.ts`](tokenizer.ts) — source text → token stream
- [`ast.ts`](ast.ts) — AST node constructors
- [`parser.ts`](parser.ts) — recursive-descent parser → AST + diagnostics
- [`analyzer.ts`](analyzer.ts) — scope, classification, dependencies, diagnostics
- [`dag.ts`](dag.ts) — ancestor sub-DAG extraction
- [`disintegrate.ts`](disintegrate.ts) — structural disintegration rewriter
- [`builtins.ts`](builtins.ts) — catalog of known FlatPPL names
- [`index.ts`](index.ts) — public API

## Testing

```sh
npm test                          # via the workspace's test script
```

Or run directly from inside this package:

```sh
node --test 'test/**/*.test.ts'
```

These commands require a Node version that runs the repository's TypeScript
sources. The audit checks used Node 26.3.0.

## License

[MIT](LICENSE)
