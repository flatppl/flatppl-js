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

## Loading model data

`load_data(source, valueset)` reads prefetched JSON, CSV or WSV text. Hosts
use the same resolver for models and data:

```javascript
const bundle = await resolveBundle(modelPath, source, readText);
const model = processSource(source, { path: modelPath, bundle });
```

`readText(resolvedPath)` is the host's asynchronous file or URL reader. Data
paths resolve relative to the module that declares them. The resolver discovers
literal paths and caches shared files once; data files are not parsed as modules.

```flatppl
dimensions = load_data("data.json", cartprod(n = posintegers))
n = dimensions.n
data = load_data("data.json", cartprod(y = cartpow(reals, n)))
```

The schema selects scalar values, numeric tensors, nested vectors, records or
tables. Records select named fields; vectors and tables may select a prefix.
Selected values must satisfy the declared shape and domain. JSON records may
contain tensors and arrays of different lengths. CSV/WSV headers name columns;
one column can also be loaded as a vector. Empty selected cells are errors.

`anything` remains an opaque placeholder and errors when accessed. Schemas and
their dimensions must resolve within the declaring module; dimensions supplied
only by a later module input substitution and imported schema references are
not supported yet. Multi-axis arrays with structured or complex elements,
Arrow IPC and Safetensors also report unsupported loading rather than guessing
a value or shape.

## Posterior MCMC with density-only priors

Use `b = elementof(reals)` and `prior = joint(b = Lebesgue(reals))`
for an improper reference prior. Combine proper draws in a `lawof` record, then
join the two measures. Dependent prior blocks use `jointchain` with explicit
kernel boundaries. MH, RAM, slice, emcee, DEMCz and elliptical slice can sample
a proper posterior without prior draws.
When prior atoms are unavailable, they try seeded uniform starting points in
`[-2,2]` for each free coordinate, using the existing support transforms. Each
walker must find a finite posterior score within 100 attempts. The host may
instead supply `initPositions`, one unconstrained vector per walker.

These starting points are not prior draws. Direct prior sampling, prior-based
importance sampling, SMC and AMIS still refuse unavailable prior samples.

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
