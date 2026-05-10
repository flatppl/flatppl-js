# `@flatppl/engine` — Architecture

Deep-dive companion to the root [`AGENTS.md`](../../AGENTS.md). This document covers
the engine package only. For the language semantics it implements, read the FlatPPL
language spec in the **flatppl-design** repository (resolution order in `AGENTS.md`).

> **Status:** Reference implementation, FlatPPL spec v0.1, pre-release. The engine
> covers a working subset of the language — enough to drive the VS Code visualizer
> end-to-end. Many spec features are recognised by the parser and analyzer but
> stop short of full execution; those gaps are noted per file below.

## Pipeline overview

```
┌──────────────┐  source text
│ tokenizer.js │
└──────┬───────┘
       │ Token[]
┌──────▼───┐
│ parser.js │  recursive-descent
└──────┬───┘
       │ AST (ast.js node constructors)
┌──────▼─────┐
│ analyzer.js│  bindings, deps, classification, validation,
└──────┬─────┘  phase analysis, multi-LHS rewrite,
       │        disintegrate detection
       │
       ├── dag.js        ── ancestor sub-DAG / full DAG
       ├── disintegrate ── structural disintegration plans
       │
       │ analyzer.bindings (Map<name, BindingInfo>)
┌──────▼──────┐
│   pir.js    │  LoweredModule = ordered map of LoweredBinding(rhs=PIR-JSON)
└──────┬──────┘
       │
       ├── lower.js      ── per-expression AST → FlatPIR-JSON
       ├── types.js      ── type constructors, unify, signatureOf
       ├── typeinfer.js  ── inferTypes(LoweredModule), per-call meta annotation
       │
       │ LoweredModule + bindings + diagnostics
┌──────▼─────────┐
│ orchestrator.js│  buildSampleChain, buildDerivations, signatureOf-likelihood,
└──────┬─────────┘  profile-plan, scope materialisation, axis enumeration
       │
       │ chain steps  (sent over postMessage)
┌──────▼──────┐
│  worker.js  │  message handler (sampleN, logDensityN, evaluateN, ...)
└──────┬──────┘
       │
       ├── sampler.js    ── stdlib-backed registry, per-distribution sampling
       ├── traceeval.js  ── sample/score walk over measure IR
       ├── empirical.js  ── log-space weighted measures, ESS, resampling
       ├── histogram.js  ── FD-bins, weighted histogram, density estimate
       └── rng.js        ── pure Philox-4x32-10 + state threading
```

Each layer is independently testable. The test suite exercises this:
`tokenizer.test.js`, `parser.test.js`, `lower.test.js`, `types.test.js`,
`typeinfer.test.js`, `orchestrator.test.js`, `sampler.test.js`, `worker.test.js`,
`empirical.test.js`, `histogram.test.js`, `rng.test.js`, etc.

## Public entry: `index.js`

`processSource(source)` runs tokenize → parse → analyze and returns
`{ ast, bindings, loweredModule, symbols, diagnostics }`. This is the
extension-host-side entry point and the one most callers want.

**Important:** `sampler.js` and `worker.js` are deliberately **not** re-exported —
they pull in ~1 MB of stdlib distribution code and are only needed inside the
worker bundle. Code on the extension host or main webview thread that needs to
sample drives the worker via postMessage. See `engine/index.js:20-25`.

---

## Module reference

Each section: responsibility · exports · key invariants · gotchas.

### `tokenizer.js` (250 lines)

**Responsibility.** Source text → Token[] with diagnostics.

**Notable.** Tracks paren/bracket depth so newlines inside `(`/`[` are treated as
whitespace (Python-style implicit line continuation). Distinguishes `_` (HOLE),
`_name_` (PLACEHOLDER), and ordinary identifiers at lex time.

**Token types.** See `T` constant; matches FlatPPL's tiny grammar (no `**`, no
`and`/`or`, no block keywords, etc. — see spec §05).

### `parser.js` (370 lines)

**Responsibility.** Token[] → AST (Program, AssignStatement, expressions).
Recursive-descent precedence climbing.

**Surface form gotchas:**
- `(x)` is parens; `(x, y)` is a tuple; `(x,)` is rejected (spec: tuples ≥ 2 elements).
- `MixedArgs` — a positional arg followed by kwargs — is admitted but the
  parser doesn't reject it; the analyzer flags ops that don't allow it.
- `==`/`!=` produce `BinaryExpr` with op `'=='` / `'!='`; the lowering to
  the function form happens in `lower.js`.

### `ast.js` (120 lines)

**Responsibility.** AST node factories. Plain objects with a `type` discriminator
and a `loc: {start, end}` field. `synthLoc(source)` makes a sentinel location for
engine-synthesized nodes (used by analyzer's multi-LHS rewriter and disintegrate).

### `builtins.js` (190 lines)

**Responsibility.** Single source of truth for which names are FlatPPL built-ins:
constants, sets, special operations, ordinary built-in functions, distributions,
measure-algebra ops.

**Critical invariant.** The catalogs here drive name-resolution in `lower.js`
(builtin call vs. user-defined call) and the unknown-name diagnostic in
`analyzer.js`. Adding a new built-in to FlatPPL requires adding it here AND in
the type-system signatures (`types.js`) AND, if executable, in
`orchestrator.EVALUABLE_OPS` AND `sampler.ARITH_OPS` (or `sampler.REGISTRY` for
distributions). See "Cross-file invariants" below.

**Engine-internal additions.** `tuple_get` is not a spec built-in — it's emitted by
the analyzer's multi-LHS rewriter. Listed here so `lower.js` treats it as a
built-in op rather than a user-defined function call.

### `analyzer.js` (1583 lines — large; subdivide mentally)

**Responsibility.** AST → analyzed `bindings: Map<name, BindingInfo>`. Top-level
entry is `analyze(ast, source)`; everything else supports it.

**What `analyze` does, in order:**
1. Pass 1: collect all defined names (catches duplicates).
2. Pass 2: classify each binding (`classifyStatement`), validate special-op
   shapes (`validateSpecialOperation`), validate holes/placeholders/indexing,
   collect deps, build `BindingInfo`.
3. Pass 3: detect disintegrate-decompositions, build `Plan`s via
   `disintegrate.disintegratePlan`, attach `effectiveValue` for synthesized
   plans / `attachDelegate` for delegate plans.
4. Pass 4: multi-LHS rewrite — `a, b, ... = call` becomes `a = tuple_get(syn, 0)`,
   `b = tuple_get(syn, 1)`, ... where `syn` is a synthetic `%mlhs:line:col`
   binding holding the original RHS.
5. Pass 5: `computePhases` — fixed/parameterized/stochastic by ancestor analysis,
   with the spec's lawof-absorbs-stochasticity rule and degenerate-Dirac
   sharpening for `draw(Dirac(value=e))`.
6. Pass 6: `pir.lowerToModule(bindings)` — produces the LoweredModule.
7. Pass 7: `typeinfer.inferTypes(loweredModule)` — mutates each LoweredBinding
   with `inferredType`, mirrored back onto analyzer-level bindings for legacy
   consumers.
8. Pass 8: bin diagnostics back onto their bindings (so the DAG view can answer
   "does this binding have an error?" without re-walking the global list).

**`BindingInfo` shape** (mutable; passes 3-8 all write to it):
```js
{
  name, names,         // names is the multi-LHS group; `name` is this entry
  line, rhs,           // line: source line; rhs: source-text slice
  type,                // 'draw' | 'input' | 'lawof' | 'functionof' | ...
                       //   (see classifyStatement)
  deps, callDeps,      // arrays of upstream binding names
  node,                // the AssignStatement AST
  nameLoc,             // the binding name's source location
  effectiveValue,      // optional: AST replacement (set by disintegrate / mlhs)
  effectiveDeps,       // recomputed deps for effectiveValue
  effectiveCallDeps,
  synthetic,           // true for engine-emitted bindings (e.g. `%mlhs:...`)
  phase,               // 'fixed' | 'parameterized' | 'stochastic'  (set by pass 5)
  inferredType,        // FlatPIR type from typeinfer (set by pass 7)
  diagnostics,         // diagnostics localised to this binding (set by pass 8)
  disintegrateRole,    // optional: { kind: 'kernel'|'prior', ...info }
  disintegratePlan,    // optional: synthesised | delegate | unsupported
}
```

**Phase analysis** — `computePhases`. Per spec §04 phases:
- `draw(...)` self → `'stochastic'` (with degenerate-Dirac sharpening to phase(e))
- `elementof(...)` self → `'parameterized'`
- `external(...)` self → `'fixed'`
- `lawof(...)` self → absorbs stochasticity (max over `absorbedPhaseOf(deps)`,
  which collapses stochastic to fixed)
- `functionof`/`kernelof`/`fn` self → `'fixed'` (the function value itself doesn't
  vary with inputs; per-call body phase is in `computePhasesForScope`, used by
  the DAG renderer for scope-local coloring)
- otherwise → max over `phaseOf(deps)`
- inline draws via `rhsContainsInlineDraw` catch `s = 2 * draw(m)` shapes the
  callee-name check would miss

**`isMeasureExpr(node, bindings)`** — measure-typed-expression predicate. Used by
disintegrate.js, dag.js, and orchestrator.js. Could/should consolidate with the
type system's `isMeasure`; right now both exist for historical reasons.

### `dag.js` (771 lines)

**Responsibility.** `computeSubDAG(bindings, target)` and `computeFullDAG(bindings)`.
Walks ancestors, materialises reified-callable scopes (each `functionof`/`kernelof`
boundary becomes a "bubble" with scope-local phase coloring), produces nodes/edges
for the cytoscape renderer.

**Key invariant.** Synthetic node IDs use `:` as separator (e.g. boundary nodes
get `id = bindingName + ':' + boundaryName`). The renderer detects scope-local
nodes via `id.indexOf(':') !== -1`. Fragile to refactor.

### `disintegrate.js` (694 lines)

**Responsibility.** Structural disintegration of joint measures into
`(kernel, prior)` pairs. Returns one of:
- `Synthesized(kernel, prior)` — the analyzer attaches synthesized AST replacements
  via `effectiveValue`.
- `Delegate(kernel: {binding}, prior: {binding})` — points at existing user
  bindings that already are the kernel and prior.
- `Unsupported(reason)` — surfaced in the DAG view as a warning.

Operates structurally on the joint's RHS AST; supports `lawof(record(...))`,
keyword-form `joint(...)`, and `jointchain(...)` shapes.

### `lower.js` (630 lines)

**Responsibility.** AST expression → FlatPIR-JSON. Pure; no bindings map needed.

**FlatPIR-JSON shapes:**
```js
// Literals
{ kind: 'lit', value, numType?, loc }      // numType: 'integer' | 'real'
// Built-in symbols (constants pi/inf/im, sets reals/posreals/..., the slice marker `all`)
{ kind: 'const', name, loc }
// References
{ kind: 'ref', ns: 'self' | '%local' | <module-alias>, name, loc }
// Hole inside fn(...)
{ kind: 'hole', loc }
// Built-in call
{ kind: 'call', op, args?, kwargs?, fields?, loc, meta? }
// User-defined call
{ kind: 'call', target: { ns, name }, args?, kwargs?, loc, meta? }
// Reified callable (functionof / kernelof / fn — kernelof and fn lowered to functionof here)
{ kind: 'call', op: 'functionof', params, paramKwargs, paramSources, body, loc }
// Module load
{ kind: 'call', op: 'load_module' | 'standard_module', args?, assigns?, loc }
```

**Key conventions:**
- `kernelof(x, kwargs)` is rewritten to `functionof(lawof(x), kwargs)` here (per
  spec §sec:kernelof identity). Downstream IR consumers only ever see `functionof`.
- `fn(body-with-holes)` is desugared to `functionof(body-with-_argN_-placeholders,
  arg1=_arg1_, ...)` — left-to-right hole numbering.
- Forms with ordered named entries (`record`, `joint`, `jointchain`, `cartprod`,
  `table`, `preset`) use `fields: [{name, value}, ...]` because order is part of
  the structure (mirrors FlatPIR `%field`). Other kwargs use `kwargs: {name: value}`.
- Operators desugar to function calls: `+` → `add`, `-` → `sub`, `*` → `mul`,
  `/` → `div`, comparisons → `lt`/`le`/`gt`/`ge`/`eq`/`ne`, etc.
  See `BIN_OP_MAP` and `UN_OP_MAP`.

### `pir.js` (190 lines)

**Responsibility.** Module-level container (`LoweredModule`) and lowering driver
(`lowerToModule`). The LoweredModule is the single source of truth for the
program's executable form; type inference, derivation building, and the
orchestrator all consume it.

**`LoweredModule` shape:**
```js
{
  bindings:  Map<name, LoweredBinding>,   // insertion-ordered
  publicSet: Set<name>,                    // names not starting with _ or %
  source:    ParsedModule | null,          // the ParsedModule we lowered from
}
```

**`LoweredBinding` shape:**
```js
{
  name, rhs,                  // rhs: PIR-JSON expression
  originLoc, originName,      // back-refs for diagnostics / DAG display
  synthetic,                  // true for engine-emitted bindings
  inferredType,               // set by typeinfer
  phase,                      // optional — analyzer mirrors here
}
```

`walkCalls(expr, visit)` is the standard post-order call visitor used by
type/phase inference. Always walks `args`, `kwargs`, `fields`, and `body`.

### `types.js` (765 lines)

**Responsibility.** FlatPIR type constructors, unification with type variables,
and the built-in signature registry.

**Type constructors** (plain `{kind, ...}` objects):
- `deferred()` — "we'll fill this in later"
- `failed(reason)` — diagnostic marker; never unifies
- `any()` — no constraint imposed (counterpart of FlatPPL's `anything` set)
- `scalar(prim)` — `prim ∈ {real, integer, boolean, complex, string}`. Constants
  `REAL`, `INTEGER`, `BOOLEAN`, `COMPLEX`, `STRING` are pre-allocated.
- `array(rank, shape, elem)` — shape entries are positive ints or `'%dynamic'`
- `record(fields)` — `fields: {name: Type}` (ordered, per spec)
- `tuple(elems)` — length ≥ 2
- `measure(domain)` — closed measure
- `tvar(id)` — type variable for polymorphic signatures
- `funcType(inputs, result)`, `kernelType(inputs, result)` — `inputs` is array of
  `{name, type}`
- `rngstate()` — opaque, structural-by-kind only

**Unification.** `unify(a, b, subst)` returns a new substitution Map or null on
failure. `%deferred` and `%any` unify with anything; `%failed` never unifies
(propagates errors). Scalar promotion handles `booleans ⊂ integers ⊂ reals →
complexes` per spec.

`unifyArith` handles numeric arithmetic with broadcasting (scalar+scalar,
matching-shape arrays, scalar+array → array). Used by `add`/`sub`/`mul`/`div` and
the comparisons.

**Signature registry.** `signatureOf(opName)` returns a freshly-instantiated
signature (variables re-keyed per call site) or null for unknown ops. Built-in
signatures are stored as factory functions in `SIGNATURE_FACTORIES` so each call
site gets fresh type variables.

> **Coverage gap:** the registry covers ~50 ops — well-typed for common
> distributions and the core measure algebra, but most multivariate distributions,
> the entire array/table generation suite, linear algebra, and measure-algebra
> ops outside `weighted`/`normalize`/`superpose`/`joint`/`iid` are missing.
> Unknown ops fall through to `inferGenericCall` → `signatureOf` returns null →
> result is `deferred()`. See `TODO.md` for the planned coverage roadmap.

### `typeinfer.js` (870 lines)

**Responsibility.** Inference over a LoweredModule. Mutates each binding to set
`inferredType`, writes per-call `meta.type` annotations.

**Public entry points:**
- `inferTypes(loweredModule)` — module-level pass; returns diagnostics array.
- `inferExprInScope(loweredModule, expr, paramTypes)` — on-demand inference at a
  synthetic call site (used by the viewer's profile-plot dispatcher).

**Special-cased ops** (don't fit the generic signature table):
`elementof`, `lawof`, `record`, `preset`, `joint`, `tuple`, `tuple_get`,
`get_field`, `vector`, `iid`, `functionof` (covers `kernelof`/`fn` after lowering).

**Reification handling** (`inferReification`). Walks the body in an extended
scope where each parameter is bound to the type of its boundary expression. If
the body is a measure, the result is a `kernelType`; if a value, a `funcType`.

**User-defined call handling** (`inferUserCall`). Currently
"monomorphic-at-definition": uses the callee's stored `result` type directly,
without re-traversing the body with call-site argument types. Polymorphic flow is
in the FlatPIR spec but unused in practice for the visualizer's current scope.

Auto-splatting (single positional record arg whose fields are a subset of the
callee's input names) is detected and routed through the kwarg path so type
checks fire correctly.

### `orchestrator.js` (3445 lines — large; mentally split)

**Responsibility.** Builds executable artifacts on top of analyzer + lowered IR:
- `buildSampleChain(target, bindings)` — topological list of sample/evaluate
  steps for the worker.
- `buildDerivations(bindings)` — measure-algebra "derivations" for the DAG view's
  measure-detail panels. Returns `{ derivations, discrete, bindings, fixedValues }`.
  The `fixedValues` map holds pre-evaluated values for fixed-phase bindings (see
  "Fixed-phase pre-eval" below).
- `signatureOf(name, bindings)` and `signatureOfLikelihood(b, bindings)` —
  call-site signature reconstruction for the plot UI.
- Profile-plot range derivation (`fourSigmaQuantileRange`, `findMatchingPresets`,
  `inlineForProfile`).
- Scope materialisation for reified callables.

**Internal subdivision** (informally — all in one file):
- ~64-450: `buildSampleChain` and supporting predicates (`normalizeMeasureIR`,
  `classifyForChain`, `resolveMeasure`).
- ~480-1640: `liftInlineSubexpressions` and the inline-subexpression machinery
  (handles `mu = 2 * draw(...)` style forms by lifting the inner draw).
- ~1685-2095: `buildDerivations` and `classifyDerivation` — the measure-algebra
  classifier dispatch.
- ~2110-2250: per-op derivation classifiers (`classifyWeighted`,
  `classifyNormalize`, ...). Driven by `MEASURE_OP_CLASSIFIERS`.
- ~2250-2630: derivation expansion and resolution (`derivationRefsValid`,
  `expandMeasureIR`, `expandMeasureRefsInIR`).
- ~2630-2820: `bayesupdate` classification, `resolveIRToValue`, `collectSelfRefs`.
- ~2830-3210: `signatureOf` for callables, `signatureOfLikelihood`, axis
  enumeration (`distributeAxes`, `walkType`, `enumerateOutputLeaves`).
- ~3243-3420: profile-plot helpers.

**Key static gates** (must agree across files — see "Cross-file invariants"):
- `SAMPLEABLE_DISTRIBUTIONS` — names the worker's `sampler.REGISTRY` implements.
- `DISCRETE_DISTRIBUTIONS` — subset whose density is over the counting reference.
- `EVALUABLE_OPS` — built-ins the worker's `evaluateExpr` knows how to compute;
  must mirror `sampler.ARITH_OPS` plus a few hand-listed extras.

### `worker.js` (418 lines) and `worker-entry.js` (89 lines)

**Responsibility.** Stateless message handler that drives `sampler.js` and
`traceeval.js`. `worker-entry.js` is the transport shim — wires the handler to
either Web Worker `postMessage` or Node `worker_threads` based on environment
sniff.

**Message types** (handler entry ~line 103): `init`, `sample`, `density`,
`evaluate`, `sampleN`, `evaluateN`, `logDensityN`, `profileN`, `dispose`. Each
returns a transferable-aware reply (`transferablesOf` lists Float64Array buffers
for zero-copy postMessage).

**`sampleN` has two paths:**
- Static-params path (`makeSampler`): build the stdlib factory once, draw N
  times. Used when params don't depend on per-atom upstreams.
- Parametric path (`makeParametricSampler`): build the factory once with only
  the prng bound, then resolve params per draw. Used for per-i refArrays.

The parametric path is critical for the orchestrator's per-atom-params model:
naively rebuilding the factory per draw makes setup cost dominate (~10× the
actual sampling cost on small distributions).

### `sampler.js` (911 lines)

**Responsibility.** stdlib-backed registry of sampleable distributions, plus
analytical `density()` for visualization, plus `evaluateExpr` for deterministic
sub-expressions.

**`REGISTRY` shape** (per distribution):
```js
{
  params:   ['mu', 'sigma'],   // ordered FlatPPL spec param names
  aliases:  {},                 // FlatPPL-name → other-name (currently empty)
  discrete: false,              // counting vs. Lebesgue reference
  Ctor:     stdlibCtor,         // analytical methods (.pdf, .cdf, .quantile)
  randFn:   stdlibRandFn,       // .factory(...params, opts) → closure
  logpdfFn: stdlibLogpdfFn,     // (x, ...params) → log p(x)
}
```

**Currently sampleable:** Normal, Exponential, LogNormal, Beta, Gamma, Cauchy,
StudentT, Bernoulli, Binomial, Poisson, Dirac. The other 12 spec distributions
are recognised by the parser/analyzer but the orchestrator returns `unsupported`
when they reach `buildSampleChain`.

**`evaluateExpr(ir, env)`** evaluates deterministic IR — literals, constants,
refs (resolved via `env`), and a fixed catalog of calls (`ARITH_OPS` table plus
`tuple`/`tuple_get`/`get_field`/`record`/`rnginit`/`rngstate`/`rand`).

**RNG bridge.** stdlib's distribution samplers expect a `() → [0,1)` PRNG closure
via `opts.prng`. We bridge our pure-functional Philox (state in, value out, new
state out) via a stateful adapter (`makePhiloxPrngAdapter`) that mutates an
internal copy of the state and exposes `getState()` so the caller can read the
trailing state when sampling completes.

### `traceeval.js` (344 lines)

**Responsibility.** Unified trace evaluator for measure expressions. One walk
handles both generative mode (sampling) and scoring mode (log-density at observed
values), driven by a `tally` argument.

**Dispatch** is via a `MEASURE_OP_WALKERS` table (line ~336). To add a new
measure-algebra op, add a handler function and register it here — no edits to the
core `walkInner`.

### `empirical.js` (666 lines)

**Responsibility.** Weighted empirical measure utilities: log-space arithmetic,
effective sample size, resampling (systematic + multinomial), normalization.

**Null-uniform-weight protocol.** A `null` weight array means "uniform weights"
without allocation. Most ops respect this; `materialiseUniform` forces explicit
allocation when an op needs it.

**Key invariant.** `logSumExp` and friends use max-subtraction stabilization;
`totalLogMass` returns 0 for null weights (treated as a probability measure).
These conventions are documented inline but are easy to violate when extending.

### `histogram.js` (324 lines)

**Responsibility.** Binning strategies (Freedman-Diaconis equal-width, integer
atoms for discrete distributions). Weighted histogram, weighted quantile,
quantile-trimmed plot range.

### `rng.js` (378 lines)

**Responsibility.** Pure Philox-4x32-10 counter-based PRNG.

**Why Philox.** Cipher-style: `(key, counter) → 4-tuple` with no internal
sequential state. Three properties this gives us: trivial reproducibility (cache
hit ≡ same value), embarrassingly parallel (workers compute disjoint counter
ranges), cross-implementation parity (cuRAND, NumPy, PyTorch all ship Philox).

**Public API.** `seedFromBytes(bytes)`, `stateFromKey(k0, k1)`,
`nextUint32(state)`, `nextUniform(state)`, `incrementCounter(c)`. The 32×32→64
multiply uses 16-bit decomposition to stay within JS safe-integer range; avoids
BigInt overhead on the hot path.

---

## Cross-file invariants

These lists must agree across multiple files. None is currently enforced by a
test — drift produces silent runtime failures. Watch for them when extending.

### Adding a new built-in distribution (e.g. `Foo`)

| File | What to add | Reason |
|---|---|---|
| `builtins.js` | `'Foo'` in `DISTRIBUTIONS` (and so in `ALL_KNOWN`) | parser/analyzer recognise the name |
| `builtins.js` | (keep `MEASURE_PRODUCING` in sync) | typeinfer / orchestrator measure-classification |
| `types.js` | `Foo: () => realDistKwargs({...})` in `SIGNATURE_FACTORIES` | typeinfer kwargs/result shape |
| `orchestrator.js` | `'Foo'` in `SAMPLEABLE_DISTRIBUTIONS` (and `DISCRETE_DISTRIBUTIONS` if applicable) | chain builder admits it |
| `sampler.js` | `Foo: { params, aliases, discrete, Ctor, randFn, logpdfFn }` in `REGISTRY` | runtime sampling + density |
| `sampler.js` | stdlib package `require()`s at top of file | Ctor / randFn / logpdfFn |
| `engine/package.json` | `@stdlib/...` deps for the new packages | npm install |
| `test/sampler.test.js` and friends | regression tests | catch param-name drift |

The kwargs in `types.js` MUST equal the `params` array entries in
`sampler.REGISTRY[Foo]` (and both should match the spec). The cross-file
invariant test `test/invariants.test.js` enforces this — if you add a
distribution with mismatched param names it'll fail loudly.

### Adding a new evaluable built-in function (e.g. `bar`)

| File | What to add |
|---|---|
| `builtins.js` | `'bar'` in `BUILTIN_FUNCTIONS` |
| `types.js` | `bar: () => ({ args: [REAL], kwargs: {}, result: REAL })` in `SIGNATURE_FACTORIES` |
| `lower.js` | only if `bar` has special syntax handling (most don't) |
| `orchestrator.js` | `'bar'` in `EVALUABLE_OPS` |
| `sampler.js` | `bar: a => Math.bar(a)` in `ARITH_OPS` |
| `test/sampler.test.js` | regression test for `bar` evaluation |

`orchestrator.EVALUABLE_OPS` and `sampler.ARITH_OPS` must contain the same op
names (the orchestrator's static gate is the runtime's gate). The orchestrator
header at line ~90 acknowledges this dependency in a comment but no test enforces
it.

### Adding a new measure-algebra op (e.g. `baz`)

| File | What to add |
|---|---|
| `builtins.js` | `'baz'` in `MEASURE_OPS`; possibly in `MEASURE_PRODUCING` |
| `types.js` | signature in `SIGNATURE_FACTORIES`, possibly `special: 'baz'` if structurally unusual |
| `typeinfer.js` | a special-case handler in `inferCall` if `special` is set |
| `lower.js` | `'baz'` in `FIELD_FORMS` if it has ordered named entries |
| `orchestrator.js` | `classifyBaz` derivation classifier; entry in `MEASURE_OP_CLASSIFIERS` |
| `traceeval.js` | walker function; entry in `MEASURE_OP_WALKERS` |
| `disintegrate.js` | dispatch entry if it can appear in joint-measure RHS |
| `test/measure-algebra.test.js` | regression test |

### Catalog cross-references summary

| Catalog | Source of truth | Mirrored in |
|---|---|---|
| Built-in name set | `builtins.ALL_KNOWN` | (drives lower's user-vs-builtin dispatch) |
| Sampleable distributions | `sampler.REGISTRY` | `orchestrator.SAMPLEABLE_DISTRIBUTIONS` |
| Discrete distributions | (subset of above) | `orchestrator.DISCRETE_DISTRIBUTIONS` |
| Evaluable functions | `sampler.ARITH_OPS` | `orchestrator.EVALUABLE_OPS`, `typeinfer` op handlers |
| Type signatures | `types.SIGNATURE_FACTORIES` | (only place) |
| Measure-producing ops | `builtins.MEASURE_PRODUCING` | `analyzer.isMeasureExpr`, `typeinfer.isMeasure`, `orchestrator` |

---

## Engine-internal vs. spec FlatPIR

Some shapes in the JS engine are engine extensions, not in the FlatPIR spec.
This is fine — engines are allowed to add internal ops — but be aware:

- **`tuple_get`**: not a spec op. Emitted by the analyzer's multi-LHS rewriter to
  project `a, b = rand(...)` into per-name bindings. Listed in
  `builtins.SPECIAL_OPERATIONS` so `lower.js` treats it as a built-in.
- **`get_field`**: lowered from surface `obj.field`. The spec's only access op
  is `get`; we emit `get_field` because it lets `typeinfer` dispatch on the
  literal-string-name shape directly. Convertible to `get` for export to
  spec-canonical FlatPIR.
- **`%mlhs:line:col` synthetic bindings**: hold the shared RHS for multi-LHS
  groups. The `%` prefix is not legal in FlatPPL surface syntax, so it can't
  collide with user names. Synthetic flag: `binding.synthetic === true`.
- **`paramSources` on reified callables**: tracks whether each boundary kwarg
  was supplied as an `Identifier` (binding ref) or `Placeholder` (auto-bound
  parameter). Used by the viewer to compute auto plot ranges; not part of
  FlatPIR.
- **`numType` on `lit` nodes**: distinguishes `1` (integer) from `1.0` (real)
  even though both round-trip to JS Number `1`. Set from the source text's
  `raw` field; preserves the spec's lexical-form distinction.
- **`originLoc` / `originName` on LoweredBindings**: back-refs from PIR-JSON to
  source AST positions, for diagnostics and DAG display. Not in spec FlatPIR;
  metadata-only.

---

## Phase analysis details

Per spec §04. Three phases: `'fixed'`, `'parameterized'`, `'stochastic'`.
Determined by ancestor analysis on the binding graph.

**Standard rule:** a binding's phase is the max of its dependencies' phases,
with `stochastic > parameterized > fixed`.

**Special rules:**
- `draw(...)` self → `'stochastic'` (overridden when degenerate, see below).
- `elementof(...)` self → `'parameterized'`.
- `external(...)` self → `'fixed'`.
- `lawof(...)` self → absorbs stochasticity. Walks deps via `absorbedPhaseOf`,
  which collapses any stochastic verdict to the underlying parameterized/fixed
  phase. Per spec §sec:lawof: "lawof absorbs stochasticity into the reified law
  rather than propagating it outward."
- `functionof`/`kernelof`/`fn` self → `'fixed'` (the function value itself doesn't
  vary). Per-call body phase is computed by `computePhasesForScope` for the DAG
  view's scope-local coloring.
- **Degenerate-Dirac sharpening:** `draw(Dirac(value=e))` and `draw(lawof(e))`
  with value-typed `e` → phase(e), not `'stochastic'`. Implements the spec
  identity that a draw from a point mass is just the point.
- **Inline draws:** `s = 2 * draw(m)` — the top-level callee isn't `draw` so the
  callee-name check would miss it; `rhsContainsInlineDraw` catches these. (Doesn't
  walk into reification bodies — those draws live in a different scope.)

Implemented in `analyzer.js:607-715` (`computePhases`) plus the two helpers
`phaseOfDegenerateDraw` / `degenerateMeasurePhase` / `phaseOfAstExpr`. Memoized
via the `phases` and `absorbedCache` Maps; cycle-guarded.

---

## Fixed-phase pre-eval and `fixedValues`

Per spec §04, fixed-phase bindings are compile-time-determinate. The orchestrator
exploits this: `buildDerivations` runs an iterative pre-evaluation pass that
walks fixed-phase bindings in topological order and computes each one's JS value
via `sampler.evaluateExpr`. The result is exposed as `fixedValues: Map<name,
JSValue>` alongside the derivation map.

This matters whenever a fixed-phase value isn't representable as a per-atom
`Float64Array` slice — typically:

- **Numeric arrays of compile-time-known length** (e.g. `random_data` from
  `rand(rstate, iid(Normal, 10))`): the array has length 10, not `SAMPLE_COUNT`.
  The per-atom evaluator would try to broadcast it into `Float64Array(N)` and
  produce undefined slices for `i ≥ 10`.
- **Records / tuples** (e.g. `rand_pars` from `rp, rs2 = rand(rs, prior)` where
  `prior = lawof(record(...))`): the value is a JS object with named fields,
  not a flat scalar.
- **Opaque values** (rngstate): no useful per-atom representation.

**Pre-eval semantics.** For each fixed-phase binding:

1. Walk all self-refs in the IR (recursing through measure-typed bindings via
   `expandMeasureIR`, which produces the canonical sampleable form traceeval
   would walk). This finds value refs nested inside measure subtrees — e.g.
   distribution params like `Normal(mu = ref(rp_field))`.
2. Skip refs to measure-typed bindings (resolved by traceeval at runtime via a
   `resolveMeasureRef` closure) and to bindings already classified as measure
   derivations.
3. For value refs, require the target to be in `fixedValues` (else defer to a
   later iteration). Build an `env` map of resolved values plus the
   `__resolveMeasureRef` callback.
4. Call `samplerLib.evaluateExpr(ir, env)`. Catch any throw and skip — the
   binding will be retried in a subsequent iteration if its deps fill in.
5. Iterate until no progress (fixed point).

**Cascade-prune compatibility.** The validity-cascade pass that drops
derivations with unresolved refs (`derivationRefsValid`) accepts refs that
resolve through `fixedValues`, not just refs that have their own derivations.
This is essential for chains like
`__anon = Normal(mu = get_field(ref(rp), "theta1"), …)`: `rp` doesn't have a
plot-derivation (it's a fixed record, not a numeric measure), but it IS in
`fixedValues` and the worker resolves the ref through session env at sample
time. Cascade-prune runs **after** pre-eval so the `fixedValues` map is
populated when the validity check fires.

**Viewer integration.** The viewer pushes `fixedValues` to the sampler worker
via `setEnv` on every `rebuildDerivations` (worker session env), so the per-
atom paths can resolve refs to fixed bindings naturally — `evaluateN` /
`sampleN` / `logDensityN` / `profileN` all merge session env underneath the
per-atom refArrays. The viewer's `collectRefArrays` drops refs to fixed-phase
bindings so the per-atom path doesn't shadow the session value with an
undefined slice. The viewer's `getMeasure` short-circuits any binding present
in `fixedValues` — synthesizing the SoA measure shape directly (records →
fields-of-Float64Array, scalars → fill, numeric arrays → literal).

**Opaque values are not plottable.** `buildPlotPlan` returns null when
`inferredType.kind === 'rngstate'`. Same convention applies if other opaque
value types arrive in future.

**Source.** `orchestrator.js:1750-1915` (resolveMeasureRef closure, isMeasureBinding,
the iterative loop). `viewer/src/viewer.js`: `fixedValueToMeasure`,
`rebuildDerivations`'s setEnv push, `collectRefArrays`'s skip, `getMeasure`'s
short-circuit.

---

## Type inference details

Per spec §11 (FlatPIR). Implemented in `typeinfer.js`. Operates on a
LoweredModule; mutates each binding's `inferredType` and writes per-call
`meta.type` annotations (FlatPIR `(%meta type phase)`, type slot only).

**Inference is required to succeed on well-formed modules.** When inference can't
resolve a type, the binding's type becomes `failed(reason)`; failure cascades
through downstream unifications via `unify`'s "failed never unifies" rule.

**Polymorphism.** Built-in signatures use type variables (`weighted: (real,
measure<T>) → measure<T>`). Each `signatureOf(opName)` call yields a freshly-
keyed signature so two call sites can't accidentally share a variable.

**User-defined callables.** Currently monomorphic-at-definition: the callee's
stored `result` type is used directly, without re-traversing the body with
call-site argument types. Polymorphic flow exists in the spec but is unused for
the visualizer's current scope — the type-check still verifies argument types
against the declared parameter types.

**Auto-splatting.** A single positional record arg whose fields are a subset of
the callee's input names is detected (`inferUserCall:721-739`) and treated as a
record-of-kwargs splat per spec §sec:calling-convention.

---

## Diagnostics shape

```js
{
  severity: 'error' | 'warning' | 'information',
  message:  string,
  loc:      { start: {line, col}, end: {line, col} },   // 0-based
}
```

Diagnostics flow:
1. Tokenizer → `tokenDiags`.
2. Parser → `parseDiags`.
3. Analyzer → `analyzeDiags` (validation, undefined-name warnings, multi-LHS
   issues, disintegrate problems).
4. Typeinfer → `typeDiagnostics` (type errors).

`processSource` concatenates them. Pass 8 of `analyze` then bins each diagnostic
back onto its source binding (`binding.diagnostics`) so the DAG view can answer
"does this binding have an error?" without re-walking the global list.

---

## Test layout

`packages/engine/test/`:
- Per-module unit tests: `tokenizer.test.js`, `parser.test.js`, `analyzer.test.js`,
  `lower.test.js`, `pir.test.js`, `types.test.js`, `typeinfer.test.js`,
  `dag.test.js`, `disintegrate.test.js`, `disintegrate-plan.test.js`,
  `histogram.test.js`, `empirical.test.js`, `rng.test.js`, `sampler.test.js`,
  `traceeval.test.js`, `worker.test.js`, `orchestrator.test.js`.
- Cross-cutting: `holes.test.js`, `indexing.test.js`, `selection.test.js`,
  `phases.test.js`, `rename.test.js`, `measure-algebra.test.js`.
- Integration: `integration.test.js` runs against `.flatppl` fixtures copied
  from the **flatppl-examples** repo into `test/fixtures/`. Update the copies
  when the originals change (no auto-sync).

Run with `npm test` (in the workspace root or in `packages/engine/`). 671 tests
in ~3.5 s as of writing.

---

## Where to look for X

| Concern | Start here |
|---|---|
| "Why is this name unknown?" | `builtins.js` (catalogs), `analyzer.js:1075` (undefined-name warning) |
| "Why does this expression have type X?" | `typeinfer.js` (inferCall), `types.js` (signatures) |
| "What phase is this binding?" | `analyzer.js:607` (`computePhases`) |
| "Why is the chain unsupported?" | `orchestrator.js:340` (`classifyForChain`) |
| "How is this distribution sampled?" | `sampler.js` (REGISTRY entry) |
| "How is this measure-algebra op handled?" | `orchestrator.js` classifier; `traceeval.js` walker |
| "Why does the DAG render this way?" | `dag.js` |
| "Why did disintegrate produce this Plan?" | `disintegrate.js` |
| "How does the worker thread work?" | `worker.js` (handler), `worker-entry.js` (transport) |
| "Why is the RNG result this number?" | `rng.js` (Philox), `sampler.js` (PRNG adapter) |

---

## Known issues and gaps

A point-in-time architectural review (May 2026) flagged a number of bugs
and gaps. Most of the small consistency bugs have since been fixed (the
`==`/`!=` lowering, the Cauchy/Logistic/Gamma/InverseGamma/Weibull/Uniform/
Categorical/GeneralizedNormal parameter-name divergences, the `Lebesgue`/
`Counting` `support` kwarg, the non-spec `tan` builtin). The remaining
items are larger:

- **Static type system covers ~50 ops.** Many spec ops fall through to
  `deferred()` silently — most multivariate distributions, the entire
  array/table-generation suite (`array`, `fill`, `zeros`, `linspace`, `cat`,
  `partition`, …), linear algebra, approximation functions, and several
  measure-algebra ops (`truncate`, `pushfwd`, `chain`, `relabel`, `bijection`,
  …). When you add a new distribution, also add its signature. See `TODO.md`
  for the planned coverage roadmap.
- **`orchestrator.js` and `viewer/src/viewer.js` are oversized** (3 445 and 5 683
  lines respectively). Both have natural decomposition seams documented in
  this file ("Internal subdivision" in the orchestrator section). They
  haven't been split yet because the inline comments are dense enough that
  intra-file navigation is workable; future contributors looking to split
  should start with the seams flagged in this doc.
- **The cross-file invariants table at the top of this section is enforced
  by `test/invariants.test.js`** for the SAMPLEABLE↔REGISTRY, EVALUABLE↔ARITH_OPS,
  DISCRETE-flag-consistency, BIN_OP_MAP-has-signature, and REGISTRY-params-↔-
  types-kwargs invariants. Adding a new catalog or a new "must-agree" link
  should come with a test in that file.

When making changes, check the review document if it's accessible — some of
those bugs may still be open.
