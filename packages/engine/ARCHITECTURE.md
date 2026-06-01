# `@flatppl/engine` — Architecture

Deep-dive companion to the root [`AGENTS.md`](../../AGENTS.md). This document covers
the engine package only. For the language semantics it implements, read the FlatPPL
language spec in the **flatppl-design** repository (resolution order in `AGENTS.md`).

> **Status (updated 2026-06-01):** Reference implementation of FlatPPL
> v0.1. All sources in TypeScript with `strict: true` (migration
> complete 2026-05). **2975 engine tests pass**; the engine drives the
> VS Code visualizer and the web gallery end-to-end. The measure-
> algebra core is feature-complete for the spec's Bayesian /
> measure-theoretic vocabulary; multivariate distributions (MvNormal,
> Dirichlet, Multinomial, Wishart, InverseWishart, LKJ / LKJCholesky,
> BinnedPoissonProcess) and the FlatPDL measure-eval primitives
> (`builtin_logdensityof` / `builtin_sample` / the four transports)
> all landed in May 2026.
>
> **`metricsum` (metric-aware Einstein summation, spec §04 §sec:metricsum)
> landed 2026-06-01.** Surface shorthand `g: result[.μ^, .ρ_] := expr`
> desugars (in the parser) to `metricsum(g, [...], expr)`; the lift
> pass (`lift.inlineMetricsumLift`) rewrites every metricsum call to
> `aggregate(sum, [stripped_axes], wrapped_body)` with inv(metric) +
> metric factor insertions per the spec lowering rule. Variance markers
> (`.name^` / `.name_`) flow through the AST, lower to FlatPIR's
> `(%uaxis name)` / `(%laxis name)`, and are consumed by lift; no
> downstream consumer sees a raw metricsum call. See
> `test/fixtures/metricsum-tensor.flatppl` for the Minkowski-metric
> tensor-algebra demo + `metricsum.test.ts` for the 24 conformance
> tests (parse + static-checks + lift IR shape + numerical correctness
> + PIR roundtrip).
>
> **Architectural arc in progress: Phase 5.1 §22 multivariate-as-
> derived-measure reframe.** Sessions 1-5e (May-June 2026) landed
> end-to-end registry-driven `pushfwd(affine, iid(Normal, D))` for
> literal-mu / literal-cov `MvNormal`. matMvNormal stays as safety
> fallback. Read `flatppl-dev/flatppl-engine-concepts.md` §22 + the
> "Session resume point" in `flatppl-dev/TODO-flatppl-js.md` to pick
> up. Remaining: Sessions 5f (widen lift gate + atom-dep params), 5g
> (retire matMvNormal + VECTOR_OUTPUT_DISTRIBUTIONS gates), 5h+
> (additional bijection registry entries for MvLogNormal / Dirichlet /
> Wishart / etc.). Outstanding spec features are tracked in
> `flatppl-dev/TODO-flatppl-js.md` — primarily the standard modules
> (particle-physics, GLM family, ext-linear-algebra, special-
> functions), `PoissonProcess`, multi-file `load_module` end-to-end
> wiring, and finer typeinfer coverage.

## Pipeline overview

```
┌──────────────┐  source text
│ tokenizer.ts │
└──────┬───────┘
       │ Token[]
┌──────▼───┐
│ parser.ts │  recursive-descent precedence climbing
└──────┬───┘
       │ AST (ast.ts node constructors)
┌──────▼─────┐
│ analyzer.ts│  bindings, deps, classification, validation,
└──────┬─────┘  phase analysis, multi-LHS rewrite,
       │        disintegrate detection, restrict expansion
       │
       ├── dag.ts                ── ancestor sub-DAG / full DAG (viewer)
       ├── disintegrate.ts +     ── structural disintegration plans
       │   disintegrate-plan.ts     (decompose → partition →
       │                              admissibility → synthesize)
       │
       │ analyzer.bindings (Map<name, BindingInfo>)
┌──────▼──────┐
│   pir.ts    │  LoweredModule = ordered map of LoweredBinding(rhs=PIR-JSON)
└──────┬──────┘
       │
       ├── lower.ts        ── per-expression AST → FlatPIR-JSON
       ├── types.ts        ── type constructors, unify, signatureOf
       ├── typeinfer.ts +  ── inferTypes(LoweredModule), per-call meta
       │   fixed-eval.ts      annotation, demand-driven const-eval for
       │                       shape positions (engine-concepts §17.4)
       ├── pir-sexpr.ts    ── FlatPIR S-expression printer + reader
       │
       │ LoweredModule + bindings + diagnostics + inferredTypes
┌──────▼─────────┐
│ orchestrator.ts│  buildSampleChain, buildDerivations, scope materialisation,
└──────┬─────────┘  axis enumeration; facade over ir-shared / lift /
       │            derivations / signatures / profile-plan
       │
       ├── ir-shared.ts     ── shared IR utilities (constant folding,
       │                        collectSelfRefs, parseSetIR, static gates)
       ├── lift.ts          ── inline-subexpression lifting + isEvaluable
       ├── derivations.ts   ── per-binding derivation classification +
       │                        the unified `expandMeasure` walker
       ├── signatures.ts    ── callable signature reconstruction
       ├── profile-plan.ts  ── profile-plot range / preset derivation
       │
       │ derivations: Map<name, Derivation> + fixedValues
┌──────▼──────────┐
│ materialiser.ts │  per-derivation-kind handlers → EmpiricalMeasure
└──────┬──────────┘  (samples + value + logWeights + n_eff + logTotalmass)
       │             via prepareDensityRefs / measureToRefValue plumbing
       │
       │ worker messages (over postMessage)
┌──────▼──────┐
│ worker.ts   │  thin shell over sampler / density / traceeval
└──────┬──────┘
       │
       ├── sampler.ts       ── stdlib REGISTRY (18 univariate dists),
       │                        evaluateExpr / evaluateExprN, ARITH_OPS,
       │                        AGGREGATE_PATTERNS, _broadcastApply
       ├── density.ts +     ── consume/rest log-density walker, OP_HANDLERS
       │   density-prims.ts    table, multivariate kernel density math
       ├── traceeval.ts     ── sample/score walker for inline measure IR
       ├── value.ts +       ── shape-tagged Value (engine-concepts §2.1),
       │   value-ops.ts        Klein-4 transpose + struct bitmask, ops
       ├── empirical.ts     ── log-space weighted measures, ESS, resample
       ├── histogram.ts     ── FD-bins, weighted histogram, density curve
       ├── dataload.ts      ── JSON / CSV / WSV parsers (load_data)
       ├── standard-modules.ts ── std-module registry skeleton
       └── rng.ts           ── pure Philox-4x32-10 + state threading
```

Each layer is independently testable. The 2187-test suite covers the
pipeline: `tokenizer.test.ts`, `parser.test.ts`, `lower.test.ts`,
`pir.test.ts`, `types.test.ts`, `typeinfer.test.ts`,
`orchestrator.test.ts`, `materialiser.test.ts`, `density.test.ts`,
`sampler.test.ts`, `traceeval.test.ts`, `worker.test.ts`,
`empirical.test.ts`, `histogram.test.ts`, `rng.test.ts`,
`value.test.ts`, plus the broad cross-cutting suites
(`closed-form-measure-algebra.test.ts`, `measure-algebra.test.ts`,
`broadcast-semantics.test.ts`, `kernel-broadcast.test.ts`,
`jointchain-first-class.test.ts`, …).

## Public entry: `index.ts`

`processSource(source, opts?)` runs tokenize → parse → analyze and returns
`{ ast, bindings, loweredModule, symbols, diagnostics, variant, bundle }`. This
is the extension-host-side entry point and the one most callers want.

`opts` accepts:
- `variant` — `'flatppl'` or a variant object. Canonical FlatPPL is the
  only surface form (spec §05); the seam is retained for forward-
  compatibility. Wins over `path`.
- `path` — source path; reserved for future per-file surface detection.
  Currently ignored.
- `bundle` — module-source bundle for multi-file models (spec §04
  `load_module`). Shape: `{ sources: { [path: string]: text: string } }`.
  Pre-resolved `.flatppl` deps. The engine pipeline stays **synchronous**;
  async I/O is the host's responsibility (vscode-extension uses
  `vscode.workspace.fs`; the standalone web host uses `fetch()`).
  Defaults to an empty bundle.

**Important:** `sampler.ts` and `worker.ts` are deliberately **not** re-exported —
they pull in ~1 MB of stdlib distribution code and are only needed inside the
worker bundle. Code on the extension host or main webview thread that needs to
sample drives the worker via postMessage. See `engine/index.ts`.

### Bundle entry points (4)

1. **`engine.min.js`** — main entry (`/engine/index.ts`). Exports
   `processSource`, `variants`, `tokenize`, `parse`, `analyze`,
   `disintegrate`, `lower`, `orchestrator`, `density`, `materialiser`,
   `histogram`, `empirical`, `value`, `types`, `typeinfer`, etc.
   **Excludes:** `sampler`, `worker` (pulled in by the worker bundle).
2. **`sampler-worker.min.js`** — `/engine/worker-entry.ts`. Thin
   transport shim; dispatches to `worker.createWorkerHandler`. Detects
   Web Worker (`self.postMessage`) vs Node `worker_threads`
   (`parentPort`). Bundled with full stdlib for per-step sampling /
   density.
3. **`viewer.js`** — `/viewer/src/index.ts`. Visualization +
   interaction; bundles cytoscape + echarts + viewer logic. Includes
   the engine bundle but NOT `sampler-worker` (external load).
4. **`vscode-extension.js`** — `/vscode-extension/extension.ts`. Editor
   integration; wires engine parsing + views. Bundles engine; loads
   sampler-worker from `lib/sampler-worker.min.js`.

### Multi-file models (architectural decisions locked 2026-05-10)

- **Engine pipeline stays sync.** `processSource(src, { bundle })` takes
  pre-resolved `.flatppl` source text. Async I/O lives in the host layer
  above the engine.
- **Eager bundling for `.flatppl` sources, lazy fetching for data.**
  Module sources are needed by every analysis pass; data values flow
  through a separate `dataResolver(path)` callback (sampler/plot only).
- **Modules don't flatten.** Each loaded module compiles to its own
  `LoweredModule`; cross-module access (`mod.x`) lowers to `(ref mod x)`.
  The viewer navigates between module DAGs (click-to-enter).
- **Standard modules are engine-provided.** `standard_module(name, compat)`
  resolves to a registry of JS-implemented bindings, not `.flatppl` text.

Currently implemented: bundle-shape API plumbing only — `processSource`
accepts and forwards the bundle. The cross-module reference resolution,
`load_module` end-to-end, and standard-module registry are tracked as
separate items in `TODO-flatppl-js.md` under "Multi-file models".

---

## Bundle build gotchas — ESM/CJS mixing

esbuild infers the module format of each engine source file from
its syntax. Engine modules must fall into exactly one of two camps:

- **Pure CJS.** No top-level `export` / `import` keywords. Runtime
  exports happen via a single `module.exports = { … }` at the
  bottom of the file. Most older engine modules are written this
  way.
- **Pure ESM.** Every runtime binding uses `export function` /
  `export const`. No `module.exports` anywhere.

**Failure mode.** A file that declares `export interface Foo { … }`
at the top (types-only) but still ends with `module.exports = { … }`
makes esbuild see the `export` keyword, classify the file as ESM,
and emit one warning per offending file:

```
[WARNING] The CommonJS "module" variable is treated as a global
variable in an ECMAScript module and may not work as expected
[commonjs-variable-in-esm]
```

The build still succeeds and the bundle **loads fine in Node**
(Node's CJS wrapper makes `module` a global) and in a permissive
browser. It then crashes inside the **VS Code webview's strict-CSP
Chromium sandbox**, where `module` is undefined:
`Uncaught ReferenceError: module is not defined`. The engine global
is never initialised, the viewer can't reach `processSource`, and
the visualization stays empty with no obvious engine-side error.

Concretely this bit us in `aggregate-shape.ts`, `axis-stack.ts`,
and `kernel-broadcast-shape.ts` (commit `9c8ee9e`). The broken
pattern:

```ts
export interface AggregateShape { … }       // <-- makes the file ESM
module.exports = { computeAggregateShape }; // <-- now invalid
```

Fix (three characters per file):

```ts
interface AggregateShape { … }              // type-only, no export
module.exports = { computeAggregateShape };
```

**Rule going forward:**

- Types-only `interface` / `type` declarations in a CJS file must
  drop the `export` keyword. They're already visible to TypeScript
  within the file.
- If a type must be importable from another module, the whole file
  becomes pure ESM — convert every `module.exports`'d binding to a
  top-level `export` and remove the trailing `module.exports = …`.
  Don't try to keep both.

**CI / pre-commit guard.** `npm run build` already prints the
`commonjs-variable-in-esm` warning on every offending file. Treat
it as fatal: pass `--log-override:commonjs-variable-in-esm=error`
to esbuild in the build scripts, or wrap the build with a post-
build check that greps the captured output and exits non-zero on
match. Either catches the regression at build time rather than at
webview load time, where the symptom (blank viewer) is far from
the cause.

---

## Module reference

Each section: responsibility · key exports · invariants · gotchas. Line
counts are approximate (rounded to nearest 50) and drift slowly as the
engine grows — treat them as relative size hints.

### `tokenizer.ts` (~300 lines)

**Responsibility.** Source text → Token[] with diagnostics.

**Notable.** Tracks paren/bracket depth so newlines inside `(`/`[` are treated as
whitespace (Python-style implicit line continuation). Distinguishes `_` (HOLE),
`_name_` (PLACEHOLDER), and ordinary identifiers at lex time.

**Token types.** See `T` constant; matches FlatPPL's tiny grammar (no `**`, no
`and`/`or`, no block keywords, etc. — see spec §05).

### `parser.ts` (~750 lines)

**Responsibility.** Token[] → AST (Program, AssignStatement, expressions).
Recursive-descent precedence climbing.

**Surface form gotchas:**
- `(x)` is parens; `(x, y)` is a tuple; `(x,)` is rejected (spec: tuples ≥ 2 elements).
- `MixedArgs` — a positional arg followed by kwargs — is admitted but the
  parser doesn't reject it; the analyzer flags ops that don't allow it.
- `==`/`!=` produce `BinaryExpr` with op `'=='` / `'!='`; the lowering to
  the function form happens in `lower.js`.

### `ast.ts` (~150 lines)

**Responsibility.** AST node factories. Plain objects with a `type` discriminator
and a `loc: {start, end}` field. `synthLoc(source)` makes a sentinel location for
engine-synthesized nodes (used by analyzer's multi-LHS rewriter and disintegrate).

### `builtins.ts` (~300 lines)

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

### `analyzer.ts` (~2500 lines — large; subdivide mentally)

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

### `dag.ts` (~800 lines)

**Responsibility.** `computeSubDAG(bindings, target)` and `computeFullDAG(bindings)`.
Walks ancestors, materialises reified-callable scopes (each `functionof`/`kernelof`
boundary becomes a "bubble" with scope-local phase coloring), produces nodes/edges
for the cytoscape renderer.

**Key invariant.** Synthetic node IDs use `:` as separator (e.g. boundary nodes
get `id = bindingName + ':' + boundaryName`). The renderer detects scope-local
nodes via `id.indexOf(':') !== -1`. Fragile to refactor.

### `disintegrate.ts` + `disintegrate-plan.ts` (~750 + ~500 lines)

**Responsibility.** Structural disintegration of joint measures into
`(kernel, prior)` pairs. Returns one of:
- `Synthesized(kernel, prior)` — the analyzer attaches synthesized AST replacements
  via `effectiveValue`.
- `Delegate(kernel: {binding}, prior: {binding})` — points at existing user
  bindings that already are the kernel and prior.
- `Unsupported(reason)` — surfaced in the DAG view as a warning.

Operates structurally on the joint's RHS AST; supports `lawof(record(...))`,
keyword-form `joint(...)`, and `jointchain(...)` shapes.

**`disintegrate-plan.ts`** factors the dispatch as `decompose → partition →
admissibility → synthesize` (engine-concepts §11, commit `ca48cde`).
Per-shape `decomposeX` / `synthesizeX` pairs (record, joint, jointchain)
share a partition helper + an admissibility check + a delegate-detection
pass.

### `restrict.ts` (~150 lines)

**Responsibility.** Implements `restrict(M, x)` (spec §06) as a pure
structural rewrite on top of `disintegrate` + `bayesupdate` +
`likelihoodof`. No new derivation kind — `restrict` is canonicalized
to its expanded form during analyzer pass 3. Supports kwarg form
(`restrict(M, a=..., b=...)`) via a synthetic `record(...)` bundling.

### `lower.ts` (~800 lines)

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

### `pir.ts` (~250 lines) + `pir-sexpr.ts` (~600 lines)

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

**`pir-sexpr.ts`** provides the FlatPIR S-expression printer / reader for
round-tripping `.flatpir` ↔ LoweredModule (spec §11). Handles calls, refs,
kwargs, fields, assigns, params, axis, `%public`, `%bind`. `%meta`
emission and cross-module `(%ref <mod> <name>)` resolution are documented
follow-ups in TODO-flatppl-js.md §11.

### `types.ts` (~800 lines)

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
- `measure(domain, opts?)` — closed measure. `opts` carries
  optional `{sampleShape, batchShape, eventShape}` — the three-
  shape decomposition (Pyro/TFP convention; P2,
  engine-concepts §18.11 / §20.10.5 item 2). Today populated for
  scalar leaf distributions (`eventShape: []`) and `iid(M, n…)`
  results (prepends n… to inner `sampleShape`). Kernel-broadcast
  `batchShape` population is a future follow-up. The fields are
  PURELY ADDITIVE — `substitute`/`fresh`/etc. preserve them; no
  consumer requires their presence yet.
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

> **Coverage status:** the registry covers ~90 ops. Univariate +
> multivariate distributions (MvNormal, Dirichlet, Multinomial, Wishart /
> InverseWishart, LKJ / LKJCholesky, BinnedPoissonProcess), the core
> measure algebra (`weighted`/`logweighted`/`normalize`/`superpose`/
> `joint`/`jointchain`/`iid`/`truncate`/`pushfwd`/`bayesupdate`/
> `likelihoodof`/`logdensityof`/`totalmass`/`disintegrate`/`restrict`),
> the FlatPDL primitive family (`builtin_logdensityof`/`builtin_sample`/
> the four transports), select/ifelse, and most array / linalg /
> approximation / reduction / norm builtins all have signatures. Gaps:
> some array-construction shape rules still defer (`fill` / `zeros` /
> `ones` / `eye` / `cartpow` / `cartprod` — see TODO §17), `chain` /
> `relabel` / `bijection` first-class typing, and a few new
> stdlib-additions (qr, diag, quadform, tile, splitblocks, blockdiagmat,
> bandedmat, conv, crosscorr) listed in TODO §07.

### `typeinfer.ts` (~1600 lines) + `fixed-eval.ts` (~250 lines)

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

**Demand-driven const-eval at shape positions** (engine-concepts §17.4).
Pure-structural inference is incomplete on FlatPPL by design — shapes
can be computed from fixed-phase values (`iid(M, lengthof(data))`,
`cartpow(reals, n)`, …). `fixed-eval.ts` provides a lazy / call-by-need
resolver invoked only at shape positions, with shape-observer
short-circuits (`length(x)` / `lengthof(x)` / `sizeof(x)` read off the
inferredType without evaluating x). The IR is left intact: resolved
integers live in a side-table side-map; rewrites that fold them into
the IR are a separate later pass with size-aware policy. **The clean
invariant: typeinfer never changes bytes in the IR.**

### `orchestrator.ts` (~460 lines) + its split modules

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

**Module map.** `orchestrator.js` was decomposed into five sibling modules
(facade-preserving: `orchestrator.js` re-binds every moved name via
`const { ... } = require('./<mod>')`, so the public API and
`module.exports` — including `_internal` test hooks — are byte-identical
to the monolith). Dependency order is strictly one-way, no cycles:

```
ir-shared.js  ◄── signatures.js ◄── lift.js ◄── derivations.js
     ▲                ▲                ▲              ▲
     └──────── profile-plan.js         │              │
     └────────────────────────── orchestrator.js (core + 5 facades)
```

- **`ir-shared.js`** — the dependency ROOT / universal leaf. Pure,
  near-dependency-free IR utilities shared by every other split module:
  constant folding (`resolveConstant`), IR→value resolution
  (`resolveIRToValue`, `valueToPlain`), self-ref collection
  (`collectSelfRefs`), set parsing (`parseSetIR`, `NAMED_SETS`),
  measure-IR canonicalisation (`normalizeMeasureIR`,
  `isFixedPhaseValueIR`), and the **static gates**
  (`SAMPLEABLE_DISTRIBUTIONS`, `DISCRETE_DISTRIBUTIONS`,
  `EVALUABLE_OPS`). Requires only `lower` + `analyzer` (+ a lazy
  `sampler` require inside `resolveIRToValue`).
- **`lift.js`** — inline-subexpression lifting
  (`liftInlineSubexpressions` and its arg-typing /
  implicit-boundary-canonicalisation helpers; `isEvaluable`). Handles
  `mu = 2 * draw(...)` style forms by hoisting the inner draw to a
  synthetic binding. Requires `lower`, `signatures`, `ir-shared`.
- **`derivations.ts`** — the measure-algebra heart: `buildDerivations`,
  `classifyDerivation`, the per-op `classify*` family +
  `MEASURE_OP_CLASSIFIERS` (16 ops), derivation expansion / resolution
  via the unified `expandMeasure(input, ctx, visited)` walker
  (engine-concepts §17.4 — replaces the former four-walker maze of
  `expandMeasureIR` + `expandMeasureRefsInIR` + structural fallback +
  `expandMeasurePos`; the two backwards-compat shims `expandMeasureIR`
  and `expandMeasureRefsInIR` survive as one-liners),
  `derivationRefsValid`, `isDiscreteAt`, `leafSampleIR`,
  `classifyBayesupdate`, implicit kernel/function signatures. Leaf
  w.r.t. the orchestrator core (never calls
  `buildSampleChain`/`classifyForChain`/`resolveMeasure`). The 15
  derivation kinds in `_expandByName` cover the spec measure algebra:
  alias / sample / mvnormal / iid / record / tuple / superpose / select
  / jointchain / weighted / normalize / pushfwd / kernelbroadcast (and
  the structural fallback through `binding.ir`).
- **`signatures.js`** — callable introspection for the profile-plot UI:
  `signatureOf`/`signatureOfLikelihood`, axis enumeration
  (`distributeAxes`, `walkType`, `enumerateOutputLeaves`,
  `extractOutputIR`, `substituteLocals`). Requires only `ir-shared`.
- **`profile-plan.js`** — profile-plot range/preset derivation
  (`resolveAxisBaseSet`, `findMatchingPresets`, `findMatchingDomains`,
  `fourSigmaQuantileRange`, `inlineForProfile`). Zero internal callers
  in `orchestrator.js`; reached only via the public API.
- **`orchestrator.js`** itself now keeps just the `buildSampleChain`
  core (`normalizeMeasureIR`-driven `classifyForChain`, `resolveMeasure`,
  the fixed-phase pre-eval), the five facade re-binds, and
  `module.exports`.

**Key static gates** — now defined in `ir-shared.js` (must agree across
files — see "Cross-file invariants"); `orchestrator.js` /
`derivations.js` consult them via the `ir-shared` facade:
- `SAMPLEABLE_DISTRIBUTIONS` — names the worker's `sampler.REGISTRY` implements.
- `DISCRETE_DISTRIBUTIONS` — subset whose density is over the counting reference.
- `EVALUABLE_OPS` — built-ins the worker's `evaluateExpr` knows how to compute;
  must mirror `sampler.ARITH_OPS` plus a few hand-listed extras.

### `materialiser.ts` (~2800 lines)

**Responsibility.** Per-binding-name → `EmpiricalMeasure` evaluator.
Drives the worker via `ctx.sendWorker(msg)`. `materialiseMeasure(name,
ctx)` dispatches to per-kind handlers in `KIND_HANDLERS` (~27 entries:
the 13 measure-algebra derivation kinds + 8 multivariate distribution
kinds + the FlatPDL primitives surface — see engine-concepts §13.6).

**Bijection-binding contract extension** (Phase 5.1 Sessions 5c+5d).
`binding.bijection` gains two optional additive fields forwarded
through `derivations.resolveBijectionMeta` to `ir.bijection`:
  - `registryName: string` (Session 5c, `188ffb5`) — the registry-
    entry key ('affine' etc.).
  - `paramIRs: Record<string, IRNode>` (Session 5d, `eea8e41`) — the
    parameter IRs the registry entry consumes at materialise / density
    time. Required-with: paramIRs MUST be present when registryName is.

**Producer** (Session 5e, `f24d058`):
  - `lift.inlineMvNormalLift` runs in the inlineUserCall fixed-point
    loop; detects kwarg-form `MvNormal(mu, cov)` calls with literal
    (or one-level-ref-to-literal) mu and cov; emits two synthetic
    bindings (`__bij_N` carrying the AST stub
    `bijection(fn(_), fn(_), 0.0)` + the side-channel marker
    `__mvnormalLowering = {muIR, covIR}`; `__iid_N` carrying
    `iid(Normal(mu=0, sigma=1), D)` with the inner Normal pre-hoisted
    to a fn anon); rewrites the original MvNormal call's RHS in place
    to `pushfwd(__bij_N, __iid_N)`. The fn(_) stubs satisfy the
    additive invariant (the AST path remains a valid fallback);
    paramIRs.L = `lower_cholesky(covIR)` evaluates lazily at
    materialise time via the existing `EVALUABLE_OPS` path.
  - `derivations` bijection-construction loop (line ~178) reads the
    `__mvnormalLowering` marker and layers `binding.bijection.
    registryName = 'affine'` + `paramIRs = {L, b}` onto the same
    object the AST-path metadata sits on. Producer / consumer
    loosely coupled by the marker — lift owns AST recognition,
    derivations owns the registry-binding contract.

**Consumer dispatchers** (Session 5d):
  - `mat-transformations.matPushfwd` (`87c37d7`) — fast path activates
    when bijection has registryName + paramIRs AND base measure is
    vector-atom (M.value with outerRank===1, shape=[N,D]). Resolves
    paramIRs via orchestrator.resolveIRToValue, calls
    `bijRegistry.getBijection(name).atomBatchedForward(z, params, N)`,
    preserves outerRank=1 on output, propagates mass/weights/n_eff.
    Falls through to scalar AST path otherwise.
  - `density.walkPushfwd` (`449e8af`) — symmetric density-side fast
    path. Resolves paramIRs via samplerLib.evaluateExpr(baseEnv+overlay),
    discovers D from params.b.shape (affine convention), consumes a
    length-D vector head, calls atomBatchedInverse + logDetJ, recurses
    walkAcc on the base measure with z as a length-D vector, ADDs
    logDetJ to acc per atom (sign convention — registry returns
    inverse Jacobian).

**Invariant: registryName / paramIRs are PURELY ADDITIVE.** `fName` /
`fInvName` / `logVolume` MUST remain present and valid when
registryName is set; the registry path is an OPTIMISATION over the AST
path, never a replacement.

**Bijection registry** (`bijection-registry.ts`, ~330 lines, Phase 5.1
Sessions 1+2) — engine-concepts §22 lands the canonical decomposition
view of multivariate distributions: every practical multivariate is
`pushfwd(known_bijection, iid(scalar, D))` (`MvNormal = pushfwd(
affine(L, mu), iid(Normal, n))`). The registry is the single point of
bijection support — sample-side `atomBatchedForward`, density-side
`atomBatchedInverse` + `logDetJ`, declared shape contract. Session 1
landed the surface + `affine` sample-side; Session 2 added the
density-side and pinned equivalence with `density-prims.MvNormal`;
Session 3 routed the LIVE density-prims path through the registry, so
both sides of MvNormal materialisation + scoring consume the registry
entry concretely. `mat-multivariate.matMvNormal` and
`density-prims.MV_DENSITY_FNS.MvNormal` are now both "thin shortcuts
over the canonical decomposition" per §22.5. Adding a new bijection
(`exp` / `log` / `stick-breaking` / `Cholesky-pack` / `projection`)
is one registry entry; future multivariates lower surface-side to a
one-line pushfwd composition.

**Bare vector-output kernel-broadcast** (`mat-broadcast.
_executeBareVectorOutputBroadcast`, Phase 5.1 Session 4 = `039031a`).
`broadcast(MvNormal, mu = mu_per_group, cov = cov_shared)` —
classifier recognises bare `VECTOR_OUTPUT_DISTRIBUTIONS` heads (set
in `ir-shared.ts` parallel to `SAMPLEABLE_DISTRIBUTIONS`), runtime
dispatches per outer cell through the bijection registry's `affine`
atom-batched forward — same hot path matMvNormal consumes, iterated
over the K cell axis. Output stitched into `[N, K, n]` atom-major.
Session 4 MVP scope: per-cell mu (rank-2 `[K, n]`) or shared mu
(rank-1 `[n]`); SHARED cov (rank-2 `[n, n]`). Per-cell cov / per-atom
stochastic mu / cov defer to Session 5+ alongside the matPushfwd
vector-base extension that gives walker symmetry.

**Nested-broadcast with vector-output inner**
(`mat-broadcast._executeNestedBroadcastComposite` extension, Phase 5.1
Session 5b = `5371a32`).
`detectNestedBroadcastKernelBinding` accepts MvNormal as the inner
head, recording `innerIsVectorOutput` + `innerEventDim`. The executor
dispatches vector-output inner per (outer, inner) cell through
`_sampleVectorOutputAtCell`; scalar inner stays on the worker's
sampleN path. Stitching is per-cell event-dim aware: output
`[N, K_outer, K_inner * eventDim]` atom-major. **Recognition-layer
only in Session 5b** — end-to-end meaningful nested pattern requires
atom-dep MvNormal params (Session 5c+'s matPushfwd vector-base
extension), since the outer kernel param needs to thread into the
inner MvNormal kwarg.

**Joint composite-body with vector-output components**
(`mat-broadcast._executeJointComposite` extension, Phase 5.1 Session
5a = `f7546a9`). `joint(loc = MvNormal(mu, cov), obs = Normal(…))` as
a kernel-broadcast body. `detectJointKernelBinding` records
`isVectorOutput` + `eventDim` per component; the executor dispatches
vector-output components through `_sampleVectorOutputAtCell` (which
consumes the same registry path matMvNormal uses), scalar components
through the existing worker `sampleN` path. Stitching is per-component
event-dim aware: output `[N, K, sum_c(eventDim_c)]` atom-major. D > 1
broadcast args admitted when any component is vector-output — per-cell
vector slicing builds a literal `vector(...)` IR for the substituted
MvNormal kwarg.

**Shared plumbing** (engine-concepts §17.1, commit `38135f2`):
- `prepareDensityRefs(ir, ctx, label)` — one owner of the
  collectSelfRefs → filter (function-like / non-binding /
  non-fixed) → split (per-atom vs fixed) → materialise per-atom →
  return `{ refArrays, fixedEnv, perAtomNames }` plumbing.
- `measureToRefValue(m, name, label)` — uniform parent-measure →
  worker-facing Value dispatch (`m.value` for vector ensembles,
  `batchedScalar(m.samples)` for scalar ensembles).
- `pushFixedEnv(ctx, fixedEnv)` — canonical setEnv merge for
  fixed-phase parents.
- `collectRefArrays(ir, ctx)` — refArrays builder used by
  leaf-sample / iid / multivariate paths. Auto-pushes fixed-phase
  refs into the worker session env via `pushFixedEnv` (single
  owner), so callers never need to remember the setEnv-merge step
  themselves. Returns the per-atom refArrays map; fixed refs are
  delivered out-of-band on the worker. Legacy 3-arg form
  `(ir, fixedValues, getMeasure)` retired 2026-05-29 after every
  in-engine caller migrated to ctx-form.

**`EmpiricalMeasure` shape** (engine-concepts §2 universal value):
```ts
{
  samples:      Float64Array,   // per-atom scalar view (atom-major)
  value?:       Value,          // shape-[N, ...dims] canonical batched
  logWeights:   Float64Array | null,   // null ⇒ uniform 1/N
  logTotalmass: number,         // log-space scalar; 0 ⇒ probability
  n_eff:        number,         // Kish ESS or constructive analogue
  fields?:      { [name]: EmpiricalMeasure },   // record-typed
  elems?:       EmpiricalMeasure[],             // tuple-typed
  dims?:        number[],       // intrinsic shape (vector/matrix atoms)
}
```

Each kind handler returns one of these. Composite measures (record /
tuple / iid / superpose / select / jointchain) compose by recursing
into children via `ctx.getMeasure(name)`. The orchestrator caches per
binding name; the materialiser is called at most once per
binding-per-context.

### `density.ts` (~1400 lines) + `density-prims.ts` (~1100 lines)

**Responsibility.** Single density implementation for the engine.
Every production density caller (matLogdensityof, matBayesupdate,
profileN-logdensity, worker.logDensityN, broadcast(logdensityof, …)
points-batched fast path) goes through `logDensityConsumeN(ir, value,
refArrays, count, opts)` — the foundation that walks the measure IR
once (atom-independent consume/rest splitting) while evaluating
per-leaf logpdf N times (the only per-atom work).

**Public API** (single function, three convenience wrappers):
- `logDensityConsumeN(ir, value, refArrays, count, opts)` →
  `{ logps: Float64Array(count), rest }` — foundation.
- `logDensityN(ir, value, refArrays, count, opts)` → Float64Array —
  strict batched wrapper (asserts empty rest).
- `logDensityConsume(ir, value, env, opts)` → `{ logp, rest }` —
  single-point with rest.
- `logDensity(ir, value, env, opts)` → number — single-point strict.

**Consume/rest dispatch** (engine-concepts §4) — `OP_HANDLERS` table
covers 11 measure ops:
- Structural: `joint` / `record` (unified `walkJointFieldsOrPositional`),
  `iid` (`walkIid`), `jointchain` (stub — canonicalised to record/joint
  by `expandMeasure` before reaching density).
- Reweighting: `weighted` (`walkWeighted`), `logweighted`
  (`walkLogWeighted`), `truncate` (`walkTruncate`),
  `normalize` (`walkNormalize`).
- Transformation: `pushfwd` (`walkPushfwd`) — requires `bijection`
  annotation per spec §06.
- Discrete-selector: `select` (`walkSelect`) — engine-concepts §12
  unified `ifelse` / `superpose` / `mixture` / `xs[i]`. Carries a
  reference-measure guard against spike-and-slab heterogeneity.
- Array-product: `broadcast` (`walkBroadcast`) — closed-form density
  for `broadcast(Dist, args…)` (engine-concepts §17 — added 2026-05-26;
  per-atom × per-element parameter resolution via `evaluateExprN`).
- Multivariate kernels (8): `MvNormal`, `Dirichlet`, `Multinomial`,
  `BinnedPoissonProcess`, `Wishart`, `InverseWishart`, `LKJ`,
  `LKJCholesky` — all route through a generic `walkMultivariate` that
  dispatches into `densityPrims.builtinLogdensityof(name, kw, x)`
  (FlatPDL primitive, see below).
- Univariate leaves: `walkLeaf` — `densityPrims.builtinLogdensityofPositional`.

**Density primitives** (`density-prims.ts`, engine-concepts §13.6) — the
FlatPDL `builtin_logdensityof` ABI surface. Holds the per-kernel
log-density math for the eight multivariate kernels (**MvNormal** —
since Phase 5.1 Session 3 (`8fbce44`), MvNormal's density routes
through the `bijection-registry.affine` entry's `atomBatchedInverse` +
`logDetJ` rather than inline forward-substitute + log-det math, so the
closed-form has ONE owner — engine-concepts §22.5 "thin shortcut over
the canonical decomposition" applies live on the density side just as
matMvNormal applies it on the sample side),
Dirichlet, Multinomial, Wishart, InverseWishart, LKJ, LKJCholesky,
BinnedPoissonProcess), the FlatPDL dispatch helpers
(`builtinLogdensityof(name, kwRecord, x)` and the positional shortcut
`builtinLogdensityofPositional(name, params, x)` walkLeaf uses), and
the multivariate transports `builtin_touniform` / `builtin_fromuniform`
/ `builtin_tonormal` / `builtin_fromnormal` (MvNormal: direct
Cholesky; non-MvNormal: TODO §07 follow-up).

`density-prims.ts` also hosts the type-mode consume/rest static
shape walker `staticConsume` / `staticDensityShapeCheck`
(engine-concepts §17.3): scalar leaves, joint(fields), iid (literal n),
weighted / normalize / truncate passthroughs, multivariate rank check.
Wired into analyser-time diagnostics is a TODO (§17 follow-up).

### `value.ts` (~700 lines) + `value-ops.ts` (~1100 lines)

**Responsibility.** The shape-tagged Value contract (engine-concepts
§2.1) — the single per-cell numeric representation the engine carries
between primitives.

**`Value` shape:**
```ts
{
  shape:   number[],         // logical shape, leading axis = batch
  data:    Float64Array,     // contiguous storage (real part if complex)
  dtype?:  'f64' | 'complex',
  im?:     Float64Array,     // imaginary part (planar; complex dtype)
  t?:      'N' | 'T' | 'A' | 'C',   // Klein-4 transpose/adjoint tag
  struct?: number,           // structured-matrix bitmask
}
```

Klein-4 tag (engine-concepts §2.1) — `transpose` toggles the swapped
bit, `adjoint` toggles both, `conjugate` toggles the conjugate bit.
Matrices read the tag in matmul / matvec / inner / outer via
index-permutation; transposes allocate no storage. Vectors stay
rank-1 (a row vector is a transposed vector, not a single-row
matrix, per spec §07).

`struct` bitmask (engine-concepts §2.2) — `ST_LOWER | ST_DIAG |
ST_UPPER` occupancy + `ST_UNIT | ST_SYM | ST_POSDEF` refinements.
Algebra: occupancy ORs under `+`; refinements AND. Storage: only
`diag` is vector-backed today; other shapes are dense + flag (the
flag's payoff is algorithmic dispatch, not bytes).

**Densify contract.** Exactly one fallback,
`densify(v)`, materialises a structured Value to plain dense. Every op
without a structured fast-path calls it via the nested-array linalg
bridge `_valueToNested`. **Correctness never depends on a fast-path
existing.**

**`value-ops.ts`** holds shape-aware `add` / `sub` / `neg` / `mul` /
the batched siblings (`mulN`, `addN`), structured-matrix Cholesky /
det / inv / triangular solve, plus the complex-arithmetic helpers
(real and imaginary buffers, planar layout). The op set is closed
under the Klein-4 tag — every binary op respects the operand tags via
the dispatcher.

### `worker.ts` (~430 lines) and `worker-entry.ts` (~90 lines)

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

### `sampler.ts` (~2800 lines) + its split modules

**Responsibility.** Single-point deterministic value evaluator
(`evaluateExpr` / `evaluateCall`), the ARITH_OPS scalar table, and
the high-level entry points the worker wraps (`rand` / `makeSampler` /
`makeParametricSampler` / `makeAnalytical` / `density`). The §17.5
sampler split (2026-05-26) factored growth-prone surfaces into five
sibling modules so standard-module work (particle-physics / GLM /
ext-linalg / …) touches the right file directly instead of bloating
the evaluator core.

**Module map** (facade-preserving: sampler.ts re-exports the names so
the public API and `module.exports._internal` are byte-identical to
the pre-split monolith). Sampler.ts went from 6154 → 2791 lines
(≈55% reduction) across the split:

```
sampler.ts (evaluator core + facade)
   │
   ├── sampler-registry.ts       (distribution catalog + REGISTRY + PRNG bridge)
   ├── sampler-complex.ts        (scalar complex arithmetic primitives, ESM)
   ├── sampler-linalg.ts         (LU / Cholesky / linsolve / inv / matmul, ESM)
   ├── sampler-aggregate.ts      (aggregate + AGGREGATE_PATTERNS, CJS)
   └── sampler-eval-batched.ts   (evaluateExprN + ARITH_OPS_N + cx batched, CJS)
```

**`sampler-registry.ts`** (~1100 lines) holds:
- REGISTRY (~25 entries): per-distribution metadata
  (`{params, aliases, discrete, Ctor, randFn, logpdfFn,
  customResolveParams?}`).
- Custom Ctor + logpdf implementations for distributions without
  stdlib backing: Dirac, Logistic, Weibull, GeneralizedNormal,
  InverseGamma, ChiSquared, VonMises, Laplace, Geometric,
  NegativeBinomial, NegativeBinomial2, Categorical, Categorical0.
- Dispatch helpers: `isKnownDistribution`, `listDistributions`,
  `lookupDistribution`, `resolveParams` (per-call lazy-requires
  `sampler.evaluateExpr` for parameter resolution),
  `regionBoundsFromIR`.
- `makePhiloxPrngAdapter` — the Philox PRNG bridge.
- `readSupport` — normalises stdlib distribution `.support` shapes.

**`sampler-complex.ts`** (~70 lines, ESM leaf): `_isComplex` /
`_toComplex` + the `_c{Add,Sub,Neg,Mul,Div,Conj,Abs,Abs2,Exp,Log,
Sqrt,Pow}` family. Pure leaf — no engine-internal deps.

**`sampler-linalg.ts`** (~330 lines, ESM leaf): the two parallel
linear-algebra surfaces — nested-array form (`_luDecomp` / `_detLU` /
`_logAbsDetLU` / `_linsolveLU` / `_invGaussJordan` / `_cholesky` /
`_matmul`) for the original ARITH_OPS path; Value-native form
(`_luDecompValue` / `_detLUValue` / `_logAbsDetLUValue` /
`_linsolveLUValue` / `_invValue` / `_choleskyValue`) for the §2.1
shape contract's row-major Float64Array path.

**`sampler-aggregate.ts`** (~1000 lines, CJS): `aggregate(f,
output_axes, expr)` — the AGGREGATE_PATTERNS specialiser table
(matmul / matvec / outer / batched-matmul / pure-axis-reduction)
plus the broadcast-reduce default lowering and its
lift/broadcast/align helpers (engine-concepts §15 + §16). Lazy-
requires `sampler.evaluateExpr` and `ARITH_OPS` (via a Proxy
forwarding to `_internal.ARITH_OPS`) for the cycle.

**`sampler-eval-batched.ts`** (~790 lines, CJS): the batched IR
evaluator dispatch (`evaluateExprN` / `_evalN` /
`_batchedApproximation` / `_perAtomFallback`), the broadcast helpers
(`broadcast1/2/3`, `isBatch`, `_scalarVal`, `_batchData`,
`_anyValueN`), the ARITH_OPS_N table + shape-aware mul/add/sub/neg
wrappers, and the complex-aware batched dispatchers (`_cxInPlay` /
`_cxArgAccessor` / `_cxRichArgGetter` / `_cxElementwise` /
`_cxIsRich` / `_cxBroadcast` / `_cxOrRealShapeAware` / `_cxOrReal`).
ARITH_OPS_N is exposed as an initially-empty table object and
populated by `initARITHOPSN(ARITH_OPS)`; sampler.ts calls
initARITHOPSN once ARITH_OPS is fully defined. `resolveConst` is
exposed on sampler.ts's `_internal` so the batched evaluator's
`const` case can look it up lazily.

**ESM vs CJS choice**: ESM modules (`sampler-complex`,
`sampler-linalg`) for pure-leaf with no cycle; CJS modules
(`sampler-registry`, `sampler-aggregate`, `sampler-eval-batched`)
for anything that requires() back to sampler.ts. Node forbids
`require()` in ES-module scope, and the cycle-breaking pattern needs
require(). The lazy require is intentionally NOT memoised —
sampler.ts re-requires its sub-modules during load, so the first
snapshot is the partial-exports view (ARITH_OPS or similar not yet
set). Node caches the module object itself, so each
`require('./sampler.ts')` returns the live `module.exports` — cheap,
and the lookup always sees the post-load state at call time.

**`sampler.ts`** itself (~2800 lines) holds: the single-point
deterministic value evaluator (`evaluateExpr` / `evaluateCall` —
the big op switch, 600+ lines covering aggregate dispatch /
broadcast / reduce / scan / filter / FlatPDL primitives / etc.),
the ARITH_OPS scalar table (~700 lines — every primitive op
including the value-aware shape dispatch), `_resolveFn` +
`_broadcastApply` + the `rand` / `makeSampler` /
`makeParametricSampler` / `makeAnalytical` / `density` entry points
the worker wraps, plus the `_internal` exports the split modules
read via the lazy cycle.

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

**Currently sampleable:**
- Univariate continuous: Normal, Uniform, Exponential, LogNormal, Beta,
  Gamma, InverseGamma, Cauchy, StudentT, Logistic, Weibull, ChiSquared,
  GeneralizedNormal, VonMises, Laplace.
- Univariate discrete: Bernoulli, Categorical, Categorical0, Binomial,
  Geometric, NegativeBinomial, NegativeBinomial2, Poisson.
- Multivariate continuous (via dedicated mat handlers, density via
  `walkMultivariate` + `density-prims`): MvNormal, Dirichlet, Wishart,
  InverseWishart, LKJ, LKJCholesky.
- Multivariate discrete: Multinomial.
- Composite: BinnedPoissonProcess.
- Degenerate: Dirac (scalar-only; identity rewrite via classifier).

**Outstanding spec distributions** (parser-recognised, sample/density
not yet implemented): `PoissonProcess` (point process with ragged
per-atom shape — needs a new measure shape class for "ragged vector
per atom"; tracked in TODO §08).

**`evaluateExpr(ir, env)`** / **`evaluateExprN(ir, refArrays, count,
baseEnv, opts)`** — the single deterministic value evaluator (single-point
+ batched form). Single-point is a one-shot if no per-atom refs touch the
expression. Batched dispatches per-op:
- `ARITH_OPS_N` (~50 entries): scalar arith with batching — add / sub / mul
  (with shape-aware routing through `value-ops` for matrices / vectors) /
  div / mod / pow / neg / pos / abs / abs2 / exp / log / sqrt / trig +
  hyperbolic / floor / ceil / round / comparisons / logic.
- `_batchedApproximation`: polynomial / bernstein / stepwise with tight
  per-atom Horner / basis loops.
- `_perAtomFallback`: per-op handlers for non-arith shapes (`tuple` /
  `tuple_get` / `get_field` / `get` / `get0` / `record` / `rnginit` /
  `rngstate` / `rand` / `broadcast` / `reduce` / `scan` / `filter` /
  `aggregate` / shape funcs / binning / `builtin_sample` /
  `builtin_logdensityof` / the four transports).

**`AGGREGATE_PATTERNS`** (engine-concepts §15 + §16) — pattern
specialiser table over the broadcast-reduce default. Landed entries:
matmul (all four transpose variants), matrix-vector, outer product,
batched matmul, pure axis reductions. Gated by the `aggregate`
optimisation toggle (`perf-config.ts`); every value-computing aggregate
test runs in both modes via `inBothModes` so the specialisers and the
default agree by construction.

**RNG bridge.** stdlib's distribution samplers expect a `() → [0,1)` PRNG closure
via `opts.prng`. We bridge our pure-functional Philox (state in, value out, new
state out) via a stateful adapter (`makePhiloxPrngAdapter`) that mutates an
internal copy of the state and exposes `getState()` so the caller can read the
trailing state when sampling completes.

### `ops.ts` (~250 lines) + `ops-declarations.ts` (~650 lines)

**Responsibility.** Unified op-declaration registry + atom-batched
dispatcher (engine-concepts §18). One declaration per non-scalar op
drives the static type signature, the runtime impl, and the
atom-batched dispatch from one source. Atom-batching is the
engine's job (recognised from the input shape's leading axis), not
the op's.

> **Status: Phases 1-5c (complete) landed (2026-05-26).** Twenty
> ops declared via `OpDecl` across four kinds:
>
> - **fixed-rank** (10): `cross`, `self_outer`, `trace`, `diagmat`,
>   `det`, `logabsdet`, `inv`, `lower_cholesky`, `row_gram`,
>   `col_gram`. Atom-batched dispatch auto-detected via leading-axis
>   convention; three have `batched(args, N)` fast-paths (`cross`,
>   `inv`, `lower_cholesky`).
> - **rank-polymorphic** (3): `transpose`, `adjoint`, `linsolve`.
>   Dispatcher hands inputs to `logical` as-is; atom-batched
>   semantics require explicit `broadcast(fn(op(_)), …)` wrapping.
> - **variadic** (2): `vector`, `cat`. Dispatcher forwards all args
>   to `logical` with no atom-batch detection.
> - **higher-order** (5): `reduce`, `scan`, `filter`, `broadcast`,
>   `aggregate`. Use the `dispatchHigherOrder(name, ir, ctx)` entry
>   which threads the engine's env + evaluateExpr + resolveFn into
>   the op's `logical`. `broadcast` delegates per-element iteration
>   to the engine's `_broadcastApply` (single-sourced impl);
>   `aggregate` delegates dispatch to `_evalAggregate` from
>   `sampler-aggregate.ts`. All dedicated `if (op === …)` blocks in
>   `evaluateCall` for these ops retired.
>
> `evaluateCall` routes declared value-domain ops through
> `opsModule.dispatch` and higher-order ops through
> `opsModule.dispatchHigherOrder`. `types.signatureOf` consults
> `ops.signatureOf` first (falls through to `SIGNATURE_FACTORIES`
> for ops that use `special:` markers, e.g. transpose / adjoint
> output-shape inference). `ARITH_OPS` entries for migrated ops
> thin-delegate.
>
> The op-declaration model now covers the full value-domain surface
> that isn't scalar arithmetic. Scalar primitives stay on
> `ARITH_OPS_N` / `_SCALAR_PRIM_ARITY` (a dispatch optimisation
> layer below `OpDecl`); measure-algebra ops live in the
> orchestrator's derivation classifier + materialiser dispatch;
> constructors / structural ops (`record` / `tuple` / `get` /
> etc.) handled inline in `evaluateCall`.

**`ops.ts`** holds:
- `register(decl)` / `lookup(name)` / `listDeclared()` /
  `signatureOf(name)` — the registry surface.
- `dispatch(name, args)` — the unified entry point.
  - Looks up the declaration.
  - For each arg, decides atom-indep (`rank == argRanks[k]`) vs
    atom-batched (`rank == argRanks[k] + 1` with leading dim N);
    enforces N consistency across batched args.
  - If all atom-indep: calls `logical(...args)` directly.
  - If any atom-batched: calls `batched(args, N)` if provided,
    else slices each batched arg into per-atom sub-Values
    (subarray views; complex `im` carried) and runs `logical` N
    times, stitching results back into a Value shape=[N, ...]
    via `_stackPerAtom`.
- `_classifyArg` / `_argAtAtom` / `_stackPerAtom` — exported for
  the conformance harness; not part of the public dispatcher API.

**`ops-declarations.ts`** holds the per-op `OpDecl` registrations.
Phase 2 covers the fixed-rank linalg family (10 ops). Phase 3
added `batched` fast-paths for the highest-value cases:
- `cross`: `_crossBatchedOrFallback` — `[N, 3]` tight loop
  (real-only); complex falls back to per-atom inline.
- `inv`: `_invBatchedOrFallback` — per-atom LU on `[N, n, n]` via
  `linalg._invValue` with subarray views.
- `lower_cholesky`: `_lowerCholeskyBatchedOrFallback` — same
  shape as `inv` via `linalg._choleskyValue`.

Each `batched` returns null when inputs don't match the fast-path
criteria (diag-stored, complex, non-Value); the wrapper runs the
per-atom loop inline. Conformance suite pins identical results
against the per-atom oracle.

**Conformance harness** lives in `test/ops-conformance.test.ts`.
Three properties pinned per declared op via fast-check:
1. Atom-indep `ops.dispatch(name, args)` ≡ `ARITH_OPS[name](args)`
   (legacy-reference equivalence).
2. Atom-batched dispatch ≡ stacked per-atom `logical` results.
3. Registry bookkeeping consistency (every declaration has a
   signature + matching argRanks + a logical impl).

Targeted unit tests cover edge cases: mixed atom-indep ×
atom-batched (broadcasting), batch-size mismatch detection,
unknown op, wrong arg count, rank incompatible with the
declaration's logical rank. 10 new tests as of 2026-05-26.

**Shape-pattern variants (P1 keystone, 2026-05-28).** Engine-
concepts §18.11 lifts shape / Klein-4 tag / structured-matrix /
dtype / wrappingOp dispatch into the registry as data: each
`OpDecl` carries an optional `variants: OpVariant[]` array, the
dispatcher picks the most-specific applicable variant on each
call. Adds three APIs:

- `registerVariant(opName, variant)` — accumulates variants on
  the op (auto-creates a minimal variant-only OpDecl if `opName`
  isn't registered yet).
- `dispatchVariant(name, args, opts)` — tries the variant
  registry; returns null if no variant matches. Used by callers
  that want to fall through to a cold path on miss.
- `hasVariantFor(name, wrappingOp)` — presence check without
  evaluating args; gates fast paths early.

`isDeclared(name)` still means "has a `logical` impl that
`dispatch(name, args)` can invoke directly" — variant-only ops
(no logical) return false here, so `evaluateCall`'s routing gate
is unchanged.

The 19 broadcasted-primitives (`add`/`sub`/`mul`/`div`/`pow`/
`mod`/`neg`/`exp`/`log`/`sqrt`/`sin`/`cos`/`tan`/`abs`/`abs2`/
`log10`/`log1p`/`expm1`/`floor`/`ceil`/`round`) register as
variant-only OpDecls with `wrappingOp: 'broadcast'` entries
pointing at the corresponding `valueOps.*Elem` impls. The legacy
`_BROADCASTED_PRIMS_CACHE` table is retired; `_maybeFastBroadcasted`
in `ops-declarations.ts` now does IR-shape recognition and routes
through `ops.dispatchVariant(opName, opInputs, { wrappingOp:
'broadcast' })`.

**`mul` direct-dispatch migration (2026-05-29).** The rank-and-
tag switch that lived in `valueOps.mul` as a sequence of
`if (sa.length === …)` clauses moves into the variant registry
as data — 11 variants with `wrappingOp: 'direct'`:

  - 2 scalar broadcast (rank-0 on either side)
  - 4 vec×vec Klein-4 variants — inner product
    (`tag: ['T','A']` × `tag: ['N','C']`), outer product
    (`['N','C']` × `['T','A']`), and 2 declarative "static error"
    variants for col×col and row×row
  - 2 matvec / vecmat variants requiring correct vector orientation
  - 2 matvec / vecmat error variants for wrong orientation
  - matmul variant

`valueOps.mul` becomes a thin entry point: diag fast-path and
complex branch stay as pre-dispatch hooks (diag's null-fallthrough
and "any-arg-is-complex" patterns don't fit cleanly into per-arg
ArgPattern matching; future refactors may migrate them). The
real path dispatches through `ops.dispatch('mul', [a, b],
{ wrappingOp: 'direct' })`.

`ArgPattern.tag` accepts an array form (`['T','A']` matches any
swapped Klein-4 tag; `['N','C']` matches any unswapped) — direct
expression of the swapped-bit semantics without per-tag-letter
combinatorial duplication.

`test/ops-variants.test.ts` pins variant-routed dispatch ≡
direct `valueOps.*Elem` calls for all 19 broadcasted primitives
plus the 11 mul direct-wrapping variants (scalar broadcast,
vec×vec inner/outer/error, matvec/vecmat/error, matmul), via
fast-check; plus pattern-matcher and specificity-ordering unit
tests (51 tests total).

**Unified value+measure dispatch via ArgInfo (P1↔P2↔P3a
integration, 2026-05-29).** The variant model extended to handle
BOTH value-domain AND measure-domain dispatch through one
registry. Three additions to ArgPattern:

  - `kind` discriminator (`'value' | 'measure' | 'function' |
    'kernel' | 'tuple'`, default `'value'`).
  - Measure-side shape fields (`sampleShape` / `batchShape` /
    `eventShape`) that consume P2 metadata on measure types.
  - `axisStack` constraint that consumes P3a IR-context entries.

`ArgInfo` is the typed wrapper that carries either a Value, a
measure type, a callable type, or an axisStack context. The
matcher splits cleanly into `_matchValueArg` /
`_matchMeasureArg`; the kind discriminator filters
categorically so value-only and measure-only variants don't
collide.

New API:
  - `argInfoFromValue(v, opts?)`, `argInfoFromMeasure(type, opts?)`,
    `argInfoFromFunction(type)`, `argInfoFromKernel(type)` —
    constructors that wrap raw payloads.
  - `dispatchTyped(name, argInfos, opts?)` — typed-dispatch entry.
    Unwraps ArgInfo to the impl's raw arg payload; passes the
    full `argInfos` via `ctx.argInfos` for impls that need the
    richer context.

Backward compat: `dispatch` / `dispatchVariant` keep their raw-
Value contract. The matcher auto-wraps raw args internally so
legacy variants stay reachable. Future fusion (b) wires this
surface through the materialiser; multivariate-density variants
and backend lowering both layer on the same registry.

**Atom-batched variant fast-paths (P1 follow-up, 2026-05-29).**
`OpVariant` gains a `batched?: (args, N, ctx?) => result` slot
alongside its atom-indep `impl`. `dispatch` / `dispatchVariant`
accept `opts.atomN: number`: when set, the dispatcher tries
atom-aware variant matching BEFORE exact-rank. The atom-aware
matcher (`_pickVariantAtomAware`) accepts an arg whose shape rank
== `pattern.rank + 1` AND whose leading dim == `atomN`, treats
that arg as atom-batched, and matches the pattern against the
rank-1-less sliced view. At least one arg must be atom-batched;
all atom-batched args must share `atomN`.

Caller signals atom-batching explicitly via `opts.atomN` — rank
alone can't distinguish "rank-2 matrix" from "atom-batched rank-1
vector (shape=[N, n])" so the dispatcher routes to the matmul
variant by default and to `_matBatchedVecMul` only when `atomN`
is set.

Variants registered (engine-concepts §18.11):
  - `add(rank-1, rank-1)`, `sub(rank-1, rank-1)` — atom-aware
    routes mixed atom-batched/atom-indep cases through
    `_atomBroadcastBinop` (the existing same-shape elementwise
    path serves the both-atom-batched case).
  - `neg(rank-1)` — atom-aware passthrough (`vo.neg` works at
    any rank).
  - `mul(rank-2 real, rank-1 real)` — atom-aware routes the
    MvNormal `L · z` hot path through `_matBatchedVecMul`.

Caller migration: `mat-multivariate.ts` and `mat-broadcast.ts`
now route the MvNormal `μ + L·z` composition through
`ops.dispatch('mul'/'add', [..], { atomN: N })`. Diag-stored `L`
stays on `value-ops.mulN`'s pre-check (its null-fallthrough fast-
path doesn't fit per-variant matching). `value-ops.mulN/addN/
subN/negN` are kept as the diag-aware façades.

Test coverage: `test/ops-atom-batched.test.ts` (12 tests) pins
atom-aware routing for add/sub/neg/mul, opt-in gating via
`opts.atomN`, the rank-2-as-matrix shadowing test (without
`atomN`, the matmul variant correctly fires), mismatched-N
refusal, and parity with `valueOps.mulN/addN` on the MvNormal
shape combinations.

`test/ops-variants-typed.test.ts` (21 tests): ArgInfo
constructors, kind-discriminated matching, measure-shape
matching (sampleShape / batchShape / eventShape with wildcards),
axisStack matching, specificity ordering, end-to-end registration
of a kind:`measure` variant that distinguishes iid sampleShape
from kernel-broadcast batchShape at the dispatch layer.

### `dissolver.ts` (~750 lines)

**Responsibility.** Term-rewriter for broadcast / aggregate
dissolution. Engine-concepts §20 (cross-engine architecture);
"Broadcast / aggregate dissolution" section below for JS-engine
specifics.

**Public surface.** `dissolveBindings(bindings)` runs in
`derivations.buildDerivations` immediately after
`liftInlineSubexpressions`. Rewrites each binding's cached `.ir` in
place when the binding matches a dissolution pattern; non-matching
bindings (and non-dissolvable broadcasts within matched bindings)
pass through unchanged. `dissolveExpr(ir, bindings)` is the
recursive walker; `_tryDissolveSingleOp(bcIR, bindings)` and
`_substituteBody(...)` are exported for unit tests.

**Phases landed (1–6).**
- Phase 1: scalar constants flow as rank-0 Values; ARITH_OPS
  routes Value inputs through valueOps unconditionally.
- Phase 2: single-op broadcast → direct call for
  `broadcast(functionof(<op>(_arg1_, _arg2_, …)), args)`.
- Phase 3: multi-op body via per-name placeholder substitution
  (`fn(_a + _b - _a)` → `sub(add(A, B), A)`).
- Phase 3.5: type-aware widening — split DISSOLVE_SAFE_OPS into
  AT_ANY_RANK and SCALAR_ONLY tiers + a lazy type resolver that
  derives types for inline / synthetic-anon args.
- Phase 4: user-fn inlining — `self`-ref heads look up the lifted
  `functionof` binding and inline.
- Phase 5: aggregate dissolution — matmul / matvec / outer product
  rewrite to direct `mul(…)` calls with Klein-4 transpose wrappers.
- Phase 6: declared fixed-rank linalg ops (cross / self_outer /
  trace / det / logabsdet / diagmat / inv / lower_cholesky /
  row_gram / col_gram) join AT_ANY_RANK — ops.dispatch's
  auto-batching covers the atom axis uniformly.

**Soundness gates.**
- DISSOLVE_SAFE_OPS split into two tiers:
    - `DISSOLVE_AT_ANY_RANK_OPS` — value-ops handles elementwise
      at any rank (add/sub/neg/pos) plus declared fixed-rank
      linalg ops where `ops.dispatch` auto-batches uniformly.
    - `DISSOLVE_SCALAR_ONLY_OPS` — multiplicative arith (mul / div
      / pow), unary scalar maths (exp / log / sqrt / sin / …),
      comparisons + logic, scalar restrictors. Permitted only
      when every outer broadcast arg has
      `inferredType.kind === 'scalar'`.
- Outer broadcast args must resolve to IDENTICAL inferred types
  with matching phase. Mixed-shape / mixed-phase broadcasts stay
  on the `_broadcastApply` cold path.
- Body refs: `%local` placeholders substituted by name; `self`
  refs allowed only when the target binding is fixed-phase
  (held constant per cell).
- Lazy type resolver derives types for inline calls and synthetic
  anon bindings (which lift creates without inferredType), so
  multi-step chains like `(a .* b) .+ c` dissolve end-to-end.

**Shape-driven constant folding (engine-concepts §20.10.7).**
`_foldShapeCall(ir, bindings)` runs inside `dissolveExpr` after each
subtree's children are walked; folds spec §07 shape→value functions
into literal IR when the input's shape is statically known:
- `lengthof(X)` → integer literal (first-axis size).
- `sizeof(X)` → literal vector of all dims; walks nested array
  types so `sizeof([[1,2,3],[4,5,6]])` folds to `vector(2, 3)`
  (matching the runtime's nested-JS-array walk in `sampler.sizeof`).
- `indicesof(X)` / `indicesof0(X)` → `vector(lit_1…lit_n)` /
  `vector(lit_0…lit_{n-1})`; rank-1 single-axis only (the multi-
  axis tuple-of-vectors form is deferred).
- Refusals: `%dynamic`, `%local` placeholders, oversize expansion
  (`_SHAPE_FOLD_MAX_LEN = 4096`). The runtime path stays.

The fold is the architectural prerequisite for fusion (a) —
broadcast-through-reductions — letting the
`broadcast(fn, args)` → `aggregate(R, [outer_axes], body)` rewrite
treat axis-derived values as ordinary literals rather than
needing a new "axis-as-value" IR form.

**Broadcast → aggregate rewrite (fusion (a) Step 2 MVP,
2026-05-29).** `_tryDissolveBroadcastReduction` recognises
`broadcast(<head>, args…)` whose functionof body is `R(<plain
elementwise>)` for R ∈ {sum, mean, prod} and emits
`aggregate(R, [.atom], <substituted_body>)`. The rewrite:
- Resolves head to a functionof (inline or via self-ref).
- Classifies each arg as Ref-wrap (per-cell value = inner of
  `vector(<single>)`) or rank-1 broadcast-over (per-cell value =
  `get(<arg>, .atom)`).
- Substitutes placeholders accordingly.
- Shape-folds the substituted body so
  `indicesof0(<concrete-ref>)` → literal vector before leaf-wrap.
- Wraps each rank-1 leaf (refs to vector bindings + literal
  vectors) in `get(<leaf>, .j)`.

**Conservative MVP gates** (engine-concepts §20.10.8): fixed-
phase only (stochastic broadcast args stay on runtime path);
plain-body only (refuses nested broadcast / aggregate / functionof
in body); single output axis (.atom); reducers ∈ {sum, mean,
prod}.

`test/fusion-a-step2.test.ts` (11 tests) pins canonical-fire,
reducer-set, phase gate, forbidden-op gate, structural refusals,
and the closure-form (single-param fn with self-ref in body).

**Kernel-broadcast inlining (fusion (b) MVP, 2026-05-29).**
`_tryDissolveKernelBroadcast` catches `broadcast(<ref to
user-kernel binding>, args…)` whose IR is `functionof(lawof
(<BuiltinDist>(kw…)), kw_params…)` and rewrites it to
`broadcast(<BuiltinDist>, mapped_kwargs)`. Lets `matKernelBroadcast`
(which only knows builtin dist names) take the existing
per-atom-params path for the simplest user-kernel case.

Conservative gates (engine-concepts §20.10.9): inner dist call is
a single builtin (uppercase op name, not in higher-order /
measure-algebra blocklist); broadcast args all positional OR all
kwarg; substitution preserves closed-over self-refs. Kernels
whose body is iid/joint/jointchain or kernel-broadcast itself
refuse cleanly; those cases need axis-stack-aware materialiser
expansion (follow-up).

`test/fusion-b-kernel-broadcast.test.ts` (12 tests) pins
canonical fires (kwarg / positional / multi-param / closure form)
and refusals (non-builtin inner op, measure-algebra inner,
higher-order inner, non-functionof head, non-lawof body, mixed
positional+kwarg, arity mismatch, unknown kwarg).

**Tests.** `test/dissolution.test.ts` (~270 lines) pins the
structural matcher in isolation plus end-to-end behaviour on
dotted-binary surfaces, refusal of unsafe ops, mismatched
types/phases, multi-op bodies, and user-fn inlining.
`test/shape-fold.test.ts` (17 tests) pins the shape-folding
pass.

**Axis-stack annotation (P3a, engine-concepts §18.11 / §20.10.5
item 4).** After the dissolution rewrites settle,
`propagateAxisStack(bindings)` walks each binding's IR and
attaches `axisStack: AxisStackEntry[]` metadata to recognised
measure-op IR nodes — pure metadata, no semantic change.

Currently annotates three IR shapes at the binding's top-level
IR:
- `iid(M, n)` → one `{source: 'iid', size}` entry; size is the
  literal value when n is a literal, the binding name when n is
  a ref, `'%dynamic'` otherwise.
- `broadcast(<head>, args…)` → one `{source: 'broadcast' |
  'kernel_broadcast', size, name?}` entry; source is
  `kernel_broadcast` when the head ref resolves to a
  kernel-typed binding, plain `broadcast` for function heads or
  unresolved.
- `aggregate(f, [axes…], body)` → one `{source: 'aggregate',
  size: '%dynamic', name}` entry per output axis name.

`axisStack` enumerates outer iteration axes the variate carries
from enclosing axis-introducing constructs, in OUTER-TO-INNER
order. The atom (sampling-time `N`) axis is deliberately NOT
included — that's an engine-internal concept the materialiser
prepends; backends without atom-axis semantics (RooFit, Stan)
consume IR axisStack as-is.

**No consumer reads axisStack yet.** It's infrastructure for
fusion thread (b) (kernel-broadcast fusion); the materialiser
and density walker pass through unchanged. Tests in
`test/axis-stack.test.ts` pin the shape-recognition logic and
the propagation pass's idempotency. Future refinements: nested
annotation (so `iid(broadcast(K, args), n)` produces the full
2-entry stack on the outer `iid`), axis-length resolution for
`aggregate` via the body's indexings, inlining into kernel
`functionof` bodies at materialise time.

### `traceeval.ts` (~400 lines)

**Responsibility.** Unified trace evaluator for measure expressions. One walk
handles both generative mode (sampling) and scoring mode (log-density at observed
values), driven by a `tally` argument.

**Dispatch** is via a `MEASURE_OP_WALKERS` table (line ~336). To add a new
measure-algebra op, add a handler function and register it here — no edits to the
core `walkInner`.

### `empirical.ts` (~700 lines)

**Responsibility.** Weighted empirical measure utilities: log-space arithmetic,
effective sample size, resampling (systematic + multinomial), normalization.

**Null-uniform-weight protocol.** A `null` weight array means "uniform weights"
without allocation. Most ops respect this; `materialiseUniform` forces explicit
allocation when an op needs it.

**Key invariant.** `logSumExp` and friends use max-subtraction stabilization;
`totalLogMass` returns 0 for null weights (treated as a probability measure).
These conventions are documented inline but are easy to violate when extending.

### `histogram.ts` (~350 lines)

**Responsibility.** Binning strategies (Freedman-Diaconis equal-width, integer
atoms for discrete distributions). Weighted histogram, weighted quantile,
quantile-trimmed plot range.

### `rng.ts` (~400 lines)

**Responsibility.** Pure Philox-4x32-10 counter-based PRNG.

**Why Philox.** Cipher-style: `(key, counter) → 4-tuple` with no internal
sequential state. Three properties this gives us: trivial reproducibility (cache
hit ≡ same value), embarrassingly parallel (workers compute disjoint counter
ranges), cross-implementation parity (cuRAND, NumPy, PyTorch all ship Philox).

**Public API.** `seedFromBytes(bytes)`, `stateFromKey(k0, k1)`,
`nextUint32(state)`, `nextUniform(state)`, `incrementCounter(c)`. The 32×32→64
multiply uses 16-bit decomposition to stay within JS safe-integer range; avoids
BigInt overhead on the hot path.

### `dataload.ts` (~400 lines)

**Responsibility.** Loaders for `load_data(source, valueset)` (spec
§07). Parses JSON (array-of-objects / object-of-arrays / vector
shapes), CSV, WSV (comma- / whitespace-separated with column names in
the first row). Arrow IPC support is a documented deferral (TODO §07).

Pure parsers — no I/O. The host (vscode-extension, web gallery) is
responsible for fetching the bytes and passing them through to the
engine via the `dataResolver(path)` callback.

### `standard-modules.ts` (~250 lines)

**Responsibility.** Engine-provided standard-module registry skeleton
(spec §09). Exposes `lookupStandardModule(name, compat)`,
`registerStandardModule`, `listStandardModules`. Each entry holds a
Map of binding-name → descriptor `{ kind: 'function' | 'value' |
'distribution', impl?, value?, sig? }`. Exact (name, compat) match for
now; semver matching is a follow-up.

The analyzer / orchestrator wiring to actually resolve
`standard_module(name, compat)` through the registry is pending — see
TODO §09.

### `engine-types.d.ts`

**Responsibility.** Cross-file type declarations consumed by the
engine's `.ts` sources. Declares `Value`, `ValueTag`, `ValueDtype`;
`IRNode` discriminated union over `IRLit | IRConst | IRRef | IRHole |
IRCall`; `DerivationKind` literal union and the per-kind interfaces
that compose the `Derivation` discriminated union; `DerivationsState`,
`BindingInfo`, `BindingType`, `EmpiricalMeasure`, `HistogramResult`.

Named `.d.ts` to avoid resolution clash with the existing runtime
`types.ts`. Also re-exported across packages via
`@flatppl/engine/engine-types` so the viewer (and any future external
consumer) can `import type` the public shapes without pulling in the
runtime engine.

---

## Cross-file invariants

These lists must agree across multiple files; drift produces silent runtime
failures. The cleanly-checkable ones ARE enforced by `test/invariants.test.js`
(distribution ↔ REGISTRY both ways, discrete flag, EVALUABLE_OPS ↔ ARITH_OPS,
BIN_OP_MAP / UN_OP_MAP type signatures, REGISTRY params ↔ types kwargs,
sampleable/discrete ⊆ builtins.DISTRIBUTIONS) — if you change a catalog the
failing test names the mirror to update. Not every cross-file relation forms a
clean subset, though: the measure-op catalogs (`MEASURE_OP_CLASSIFIERS`,
traceeval `MEASURE_OP_WALKERS`, `MEASURE_PRODUCING`) are intentionally
cross-cut by structural vs algebraic vs density-routed handling, so those are
NOT invariant-tested (an equivalence there would be false). Watch them by hand
when extending.

> **Forward pointer.** The unified `ops.ts` declaration model
> (engine-concepts §18) collapses much of this drift surface by
> having one declaration per non-scalar op drive the type signature,
> the runtime impl, and the atom-batched dispatch from one source.
> Phase 1 has shipped the registry + dispatcher + conformance harness
> additively (`cross` and `self_outer` are declared via `OpDecl`); the
> per-file catalogues below remain authoritative until Phase 2
> migrates each op family. Once an op is declared in `ops-
> declarations.ts`, its rows in the tables below collapse to "see
> the OpDecl".

### Adding a new built-in distribution (e.g. `Foo`)

| File | What to add | Reason |
|---|---|---|
| `builtins.ts` | `'Foo'` in `DISTRIBUTIONS` (and so in `ALL_KNOWN`) | parser/analyzer recognise the name |
| `builtins.ts` | (keep `MEASURE_PRODUCING` in sync) | typeinfer / orchestrator measure-classification |
| `types.ts` | `Foo: () => realDistKwargs({...})` in `SIGNATURE_FACTORIES` | typeinfer kwargs/result shape |
| `ir-shared.ts` | `'Foo'` in `SAMPLEABLE_DISTRIBUTIONS` (and `DISCRETE_DISTRIBUTIONS` if applicable) | chain builder admits it (re-bound into orchestrator/derivations via the `ir-shared` facade) |
| `sampler.ts` | `Foo: { params, aliases, discrete, Ctor, randFn, logpdfFn }` in `REGISTRY` | runtime sampling + density |
| `sampler.ts` | stdlib package `require()`s at top of file | Ctor / randFn / logpdfFn |
| `density-prims.ts` | per-kernel logpdf math + `MV_DENSITY_FNS` entry (multivariate only) | walkMultivariate dispatch |
| `density.ts` | `OP_HANDLERS[Foo] = walkMultivariate` (multivariate only) | density walker dispatch |
| `materialiser.ts` | `KIND_HANDLERS[Foo] = matFoo` + the `matFoo` handler (multivariate only) | per-atom sampling |
| `engine/package.json` | `@stdlib/...` deps for the new packages | npm install |
| `test/sampler.test.ts` and friends | regression tests | catch param-name drift |

The kwargs in `types.js` MUST equal the `params` array entries in
`sampler.REGISTRY[Foo]` (and both should match the spec). The cross-file
invariant test `test/invariants.test.js` enforces this — if you add a
distribution with mismatched param names it'll fail loudly.

### Adding a new evaluable built-in function (e.g. `bar`)

| File | What to add |
|---|---|
| `builtins.ts` | `'bar'` in `BUILTIN_FUNCTIONS` |
| `types.ts` | `bar: () => ({ args: [REAL], kwargs: {}, result: REAL })` in `SIGNATURE_FACTORIES` |
| `lower.ts` | only if `bar` has special syntax handling (most don't) |
| `ir-shared.ts` | `'bar'` in `EVALUABLE_OPS` |
| `sampler.ts` | `bar: a => Math.bar(a)` in `ARITH_OPS` (single-point) + `ARITH_OPS_N` (batched) if numeric |
| `test/sampler.test.ts` | regression test for `bar` evaluation |

`ir-shared.EVALUABLE_OPS` and `sampler.ARITH_OPS` must contain the same op
names (the orchestrator's static gate is the runtime's gate). This IS
enforced both ways by `test/invariants.test.js` block 3 (modulo the
documented `vector` / `cat` exemptions and the inline-evaluable handful).

### Adding a new measure-algebra op (e.g. `baz`)

| File | What to add |
|---|---|
| `builtins.ts` | `'baz'` in `MEASURE_OPS`; possibly in `MEASURE_PRODUCING` |
| `types.ts` | signature in `SIGNATURE_FACTORIES`, possibly `special: 'baz'` if structurally unusual |
| `typeinfer.ts` | a special-case handler in `inferCall` if `special` is set |
| `lower.ts` | `'baz'` in `FIELD_FORMS` if it has ordered named entries |
| `derivations.ts` | `classifyBaz` derivation classifier; entry in `MEASURE_OP_CLASSIFIERS`; a `case 'baz'` arm in `_expandByName`'s switch (explicit, never via the structural fallback) |
| `traceeval.ts` | walker function; entry in `MEASURE_OP_WALKERS` (sampling side) |
| `density.ts` | walker function; entry in `OP_HANDLERS` (scoring side) |
| `materialiser.ts` | per-kind handler; entry in `KIND_HANDLERS` |
| `disintegrate.ts` | dispatch entry if it can appear in joint-measure RHS |
| `test/measure-algebra.test.ts` + `closed-form-measure-algebra.test.ts` | regression tests |

### Catalog cross-references summary

| Catalog | Source of truth | Mirrored in |
|---|---|---|
| Built-in name set | `builtins.ALL_KNOWN` | (drives lower's user-vs-builtin dispatch) |
| Sampleable distributions | `sampler.REGISTRY` + multivariate `mat*` handlers | `ir-shared.SAMPLEABLE_DISTRIBUTIONS` |
| Discrete distributions | (subset of above) | `ir-shared.DISCRETE_DISTRIBUTIONS` |
| Evaluable functions | `sampler.ARITH_OPS` / `ARITH_OPS_N` | `ir-shared.EVALUABLE_OPS`, `typeinfer` op handlers |
| Type signatures | `types.SIGNATURE_FACTORIES` | (only place) |
| Measure-producing ops | `builtins.MEASURE_PRODUCING` | `analyzer.isMeasureExpr`, `typeinfer.isMeasure`, `orchestrator` |
| Derivation kinds | `derivations.ts` `_expandByName` switch | `materialiser.KIND_HANDLERS`; `engine-types.d.ts` `DerivationKind` union |
| Density-walker ops | `density.OP_HANDLERS` | (consumes `expandMeasure` output; no other mirror) |
| FlatPDL primitive surface | `density-prims.ts` (`builtin_logdensityof` / multivariate transports) | `sampler.ts` (`builtin_sample`); spec §07 sec:measure-eval-prims |

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

## Deterministic-evaluation authority

There is exactly **one** deterministic value evaluator in the engine:
`sampler.evaluateExpr(ir, env)` (single-point) and its batched sibling
`sampler.evaluateExprN(ir, refArrays, count, baseEnv, opts)`. Everything that
needs "compute the JS value of this deterministic IR" routes through it:

- **Fixed-phase pre-eval** (`orchestrator.buildDerivations`) — calls
  `evaluateExpr` per fixed binding (see next section).
- **`orchestrator.resolveIRToValue`** — a thin *adapter*, not a second
  evaluator: it keeps zero-dependency static fast paths (lit / neg / vector /
  record / ref) for its plain-JS output contract and delegates *every other*
  IR shape to `evaluateExpr`, feeding its own ref machinery (fixedValues →
  recursive binding walk → cycle guard) in as the env. `valueToPlain`
  normalizes the result back to the documented number|array|object contract.
- **Worker leaf-draw / param eval** — `evaluateExprN` over per-atom
  `refArrays` layered on the session env (fixed-phase bindings pushed via
  `setEnv`).

**The invariant: do not grow a parallel mini-interpreter.** The
`resolveIRToValue`-broadcast bug and the `get`-not-evaluable dead end were both
the same defect — a partial re-implementation of evaluation standing in for the
authority and throwing "the orchestrator should pre-resolve this" on anything
it didn't cover. When the evaluator is missing an op, the fix is to implement
that op *in `evaluateCall`* (the authority), not to special-case it in a
caller. `evaluateCall` covers ARITH_OPS (positional spread) plus dedicated
cases for the kwargs-/structure-shaped ops: `tuple`/`tuple_get`/`record`/
`get_field`/`get`/`get0`/`fixed`/`rnginit`/`rngstate`/`rand`/shape funcs/
binning/`filter`/`reduce`/`scan`/`broadcast`. The remaining throw is reached
only by genuinely non-deterministic-in-scope shapes (user-defined function
calls, module access) — and a fixed-phase binding that hits it is now surfaced
loudly by `buildDerivations`'s fixed-phase-dead-end diagnostic rather than
failing silently far downstream.

The `orchestrator.EVALUABLE_OPS` static gate must list every op
`evaluateCall` can handle (enforced by `test/invariants.test.js` block 3 +
the inline-evaluable set); they move together.

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

**Source.** `derivations.js` `expandMeasureIR` (resolveMeasureRef closure,
isMeasureBinding, the iterative loop). `viewer/src/viewer.js`: `fixedValueToMeasure`,
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

`packages/engine/test/` (~98 test files):
- Per-module unit tests: `tokenizer.test.ts`, `parser.test.ts`,
  `analyzer.test.ts`, `lower.test.ts`, `pir.test.ts`, `pir-sexpr.test.ts`,
  `types.test.ts`, `typeinfer.test.ts`, `dag.test.ts`,
  `disintegrate.test.ts`, `disintegrate-plan.test.ts`,
  `disintegrate-bi-equivalence.test.ts`,
  `disintegrate-spec-coverage.test.ts`, `histogram.test.ts`,
  `empirical.test.ts`, `rng.test.ts`, `sampler.test.ts`,
  `traceeval.test.ts`, `worker.test.ts`, `orchestrator.test.ts`,
  `materialiser.test.ts`, `density.test.ts`, `value.test.ts`,
  `value-ops-*.test.ts`.
- Distributions: `sampler.test.ts`, `multivariate-distributions.test.ts`,
  `new-univariate-distributions.test.ts`,
  `builtin-sample-transport.test.ts`,
  `multivariate-density.test.ts`, `builtin-logdensityof.test.ts`,
  `static-density-shape.test.ts`,
  `hierarchical-multivariate-density.test.ts`.
- Measure algebra: `measure-algebra.test.ts`,
  `closed-form-measure-algebra.test.ts`, `broadcast-semantics.test.ts`,
  `kernel-broadcast.test.ts`, `jointchain-first-class.test.ts`,
  `iid-multi-axis.test.ts`, `marginal-sampling.test.ts`.
- Type inference: `typeinfer.test.ts`, `typeinfer-conformance.test.ts`,
  `typeinfer-conformance-property.test.ts`,
  `typeinfer-tvector.test.ts`.
- Linear algebra: `linear-algebra.test.ts`, `diag-structure.test.ts`,
  `vector-atom-reductions.test.ts`, `vector-atom-softmax.test.ts`.
- Cross-cutting: `holes.test.ts`, `indexing.test.ts`,
  `selection.test.ts`, `phases.test.ts`, `rename.test.ts`,
  `invariants.test.ts`, `equivalence.test.ts`,
  `spec-patterns.test.ts`.
- Integration: `integration.test.ts` runs against `.flatppl` fixtures
  copied from the **flatppl-examples** repo into `test/fixtures/`.
  Update the copies when the originals change (no auto-sync).

Run with `npm --workspace=@flatppl/engine run test` (or `npm test` in
`packages/engine/`). 2428 tests in ~25-40 s as of writing.

---

## Where to look for X

| Concern | Start here |
|---|---|
| "Why is this name unknown?" | `builtins.ts` (catalogs), `analyzer.ts` (undefined-name warning) |
| "Why does this expression have type X?" | `typeinfer.ts` (inferCall), `types.ts` (signatures), `fixed-eval.ts` (const-eval at shape positions) |
| "What phase is this binding?" | `analyzer.ts` `computePhases` |
| "Why is the chain unsupported?" | `orchestrator.ts` (`classifyForChain`) |
| "Why doesn't this binding have a derivation?" | `derivations.ts` (`buildDerivations`, the per-op classifiers) |
| "What's the canonical measure IR for this binding?" | `derivations.ts` `expandMeasure(name, { derivations, bindings })` |
| "How is this distribution sampled?" | `sampler.ts` (REGISTRY entry); multivariate: `materialiser.ts` `mat*` handler |
| "How is this distribution's density computed?" | `density.ts` `OP_HANDLERS` for structural ops; `density-prims.ts` for the per-kernel math |
| "How is this measure-algebra op handled?" | `derivations.ts` classifier + `_expandByName` case; `materialiser.ts` `KIND_HANDLERS` entry; `density.ts` `OP_HANDLERS` (scoring); `traceeval.ts` `MEASURE_OP_WALKERS` (sampling-walker, deprecated for materialiser paths) |
| "Why does the DAG render this way?" | `dag.ts` |
| "Why did disintegrate produce this Plan?" | `disintegrate.ts` + `disintegrate-plan.ts` |
| "How does the worker thread work?" | `worker.ts` (handler), `worker-entry.ts` (transport) |
| "Why is the RNG result this number?" | `rng.ts` (Philox), `sampler.ts` (PRNG adapter) |
| "How are refs in a measure body resolved?" | `materialiser.ts` `prepareDensityRefs` (the canonical plumbing); `sampler.ts` `evaluateExprN` (worker side) |
| "What's the empirical measure shape?" | `engine-types.d.ts` `EmpiricalMeasure`; engine-concepts §2 |
| "How is a Value laid out?" | `value.ts` `isValue`; engine-concepts §2.1 (Klein-4 tag) + §2.2 (struct bitmask) |

---

## Known issues and gaps

Major outstanding work is tracked in
`flatppl-dev/TODO-flatppl-js.md` (the single source of truth for
what's left). Shortlist:

- **Multi-file models** — `processSource(src, { bundle })` accepts a
  pre-resolved source bundle (landed 2026-05-24) but the rest of
  `load_module` end-to-end is pending: cross-module reference
  resolution (`mod.x` → `ref(ns: mod)`), substitution semantics with
  phase-compatibility checks, host pre-fetching via
  `vscode.workspace.fs` / `fetch()`. See TODO "Multi-file models".
- **Standard modules** — registry skeleton landed
  (`standard-modules.ts`) but the analyzer / orchestrator don't yet
  resolve `standard_module(name, compat)` through it. Five
  std-modules outstanding: `particle-physics`, `generalized-linear-
  models`, `ext-linear-algebra`, `special-functions`, `polynomials`,
  `distances`. See TODO §09.
- **`PoissonProcess`** — point process with permutation-invariant
  arrays / tables. Sampling needs a new measure shape class for
  "ragged vector per atom" (per-atom event count varies). See
  TODO §08.
- **Multivariate transports beyond MvNormal** — Dirichlet
  (Connor–Mosimann stick-breaking) and other measure-specific
  transports per spec §08. See TODO §07.
- **Static type system gaps** — most ops have signatures, but a few
  array-construction shape rules (`fill` / `zeros` / `ones` /
  `eye` / `cartpow` / `cartprod`) still return `deferred()`; new
  stdlib functions (qr / diag / quadform / tile / splitblocks /
  blockdiagmat / bandedmat / conv / crosscorr) need signatures. See
  TODO §03 + §07.
- **`viewer/src/viewer.js` is still oversized** (~5 700 lines) and
  is the next decomposition target (the engine's
  orchestrator decomposition is done). It has natural seams but no
  module split yet. Tracked in flatppl-js' viewer-internal TODOs.
- **Cross-file invariants** are enforced by `test/invariants.test.ts`
  (SAMPLEABLE ↔ REGISTRY ↔ DISCRETE flag consistency, EVALUABLE_OPS
  ↔ ARITH_OPS, BIN_OP_MAP signatures, REGISTRY params ↔ types
  kwargs, sampleable ⊆ DISTRIBUTIONS). Adding a new catalog or a new
  "must-agree" link should come with a test there. The `MEASURE_OP_*`
  catalogs (`MEASURE_OP_CLASSIFIERS`, traceeval `MEASURE_OP_WALKERS`,
  density `OP_HANDLERS`, materialiser `KIND_HANDLERS`) are
  intentionally NOT invariant-checked — they cross-cut structural vs
  algebraic vs density-routed dispatch and a clean equivalence would
  be false. Watch them by hand when extending.

For the full list, current priorities, and per-section status, read
`flatppl-dev/TODO-flatppl-js.md`. That file is checkable surface (no
status duplication here).


## Broadcast / aggregate dissolution (JS-engine specifics)

Cross-engine architecture in `flatppl-dev/flatppl-engine-concepts.md`
§20. JS-engine implementation notes below.

> **Terminology.** Dissolution = our specific term for the
> spec-§04-driven term-rewriting that turns broadcast / aggregate
> IR into batched-primitive call chains. In the broader ML-compiler
> world the umbrella term is **fusion** (XLA elementwise / reduction
> fusion; JAX vmap composition; LLVM loop fusion). The two are
> compatible — dissolution is a particular flavor of broadcast +
> aggregate fusion tied to FlatPPL's spec-level constructs. The
> rest of this section uses "dissolution" for what the dissolver
> does; "fusion" when comparing to the ML-compiler tradition.

> **Status (2026-05-28).** The dissolver is in. Constants flow as
> rank-0 Values; `ARITH_OPS` no longer emits bare numbers when Values
> flow in; broadcast and aggregate IR rewrite to direct elementwise /
> linalg / matmul-family call chains via `dissolver.ts`. The runtime
> `_broadcastLogical` has a fast-path for `broadcast(<scalar_op>, args)`
> and `broadcasted(<op>)(args)` shapes, dispatching through value-ops'
> batched-elementwise primitives (`mulElem` / `divElem` / `powElem` /
> `expElem` / `logElem` / …). Spec-`mul` matrix semantics fully
> preserved — the fast-path only intercepts `broadcast(...)` IRs.
> Remaining follow-ups (widening the broadcasted-primitives table to
> the rest of the scalar surface, singleton-axis expansion in value-
> ops elementwise, `batched` fast-paths for the remaining declared
> linalg ops, length-N `.samples` retirement on rank-0 fixed measures)
> tracked in `flatppl-dev/TODO-flatppl-js.md`.

### Where dissolution slots into the pipeline

The dissolver runs in `derivations.buildDerivations`, **immediately
after `liftInlineSubexpressions`** — at this point each binding has
a cached lifted IR (`b.ir`) and inferred-type annotation
(`b.inferredType`, propagated from analyzer pass 7). The dissolver
rewrites the IR in place; classifiers / materialiser / sampler see
the dissolved form.

```
parse → analyze (incl. typeinfer) → lower → lift → DISSOLVE → derivations
                                                                  ↓
                                                              materialiser → sampler / density
```

The dissolver mutates `b.ir` only when a dissolution is applied; the
lifted AST (`b.node.value`) is untouched. Non-dissolvable broadcasts
(kernel-broadcast, table row-dispatch, recursive user-fns, mismatched-
shape broadcasts) keep their original IR — the existing
`_broadcastApply` runtime path handles them unchanged.

### The dissolver — pattern + soundness

`packages/engine/dissolver.ts` exports `dissolveBindings(bindings)`
and a single internal entry `_tryDissolveSingleOp(bcIR, bindings)`.
The dissolver matches

```
broadcast(<head>, args…)   # positional
broadcast(<head>, name=arg, …)   # kwarg
```

where `<head>` is either an inline `functionof` or a `self`-ref to a
user-defined `functionof` binding (Phase 4 inlining).

After resolving the head, the dissolver walks the body's expression
tree via `_substituteBody`, replacing every `%local _argN_` ref with
the corresponding broadcast arg by name. Each call-node along the
way must satisfy:

- `op` ∈ `DISSOLVE_AT_ANY_RANK_OPS` (no shape constraint) or
  `op` ∈ `DISSOLVE_SCALAR_ONLY_OPS` AND all outer broadcast args
  are scalar-typed. The two tiers are spelled out in the module-
  reference entry above.
- No kwargs / fields / nested functionof body inside the call.
- All leaf refs are either `%local` placeholders (substituted),
  fixed-phase `self` refs (held constant per cell), literals, or
  constants.

Soundness gate on the outer broadcast args: every positional
broadcast arg must resolve to a concrete `inferredType` with
matching phase across args. The `_resolveExprType` lazy resolver
handles both binding refs (consults `inferredType`; falls back to
walking the binding's IR) and inline call expressions (computes
the result type from the args' types). Mixed-shape (`scalar .+
vec`) and mixed-phase (`fixed_vec .+ parameterised_vec`)
broadcasts leave the broadcast IR in place — `_broadcastApply`
runs them on the cold path.

The dissolver iterates `dissolveBindings` to a fixed point so a
newly-dissolved leaf unblocks a parent in the same pass
(`(a .* b) .+ c` etc.).

`DISSOLVE_SAFE_OPS` is deliberately narrow until type-aware shape
safety lands (`mul` has matrix semantics on rank-1 vectors via
Klein-4; scalar unaries like `exp` / `log` route through
`broadcast1` which assumes rank-0/rank-1 inputs — silently breaks
on higher-rank cells). Widening lands in a Phase 3 follow-up.

### The runtime contract after dissolution

Post-dissolution, the dissolved IR runs through `evaluateExprN`
which dispatches scalar primitives via `ARITH_OPS_N` and shape-rich
ops via the existing value-ops paths.

For broadcasts the dissolver leaves untouched (or that the user
writes explicitly via `broadcasted(op)(args)`), the
`_broadcastLogical` op handler tries a runtime fast-path first
(`_maybeFastBroadcasted` in `ops-declarations.ts`):

- Recognises `broadcast(<ref to op>, args)` and `broadcast(functionof
  (<op>(_, …)), args)` when `<op>` is in a built-in scalar-primitive
  table (`_BROADCASTED_PRIMS_CACHE`).
- Coerces all broadcast args to Values via `asValue`. Rejects (returns
  null → cold path) when an arg is a nested JS array (Ref-wrap idiom)
  or a non-coercible host value.
- Shape check: all non-rank-0 inputs must have the SAME shape (value-
  ops elementwise contract). Singleton-axis expansion (`[3] .+ [1]`)
  is NOT handled here; falls through to the cold path which expands
  size-1 axes per spec §04.
- Dispatches through value-ops' `mulElem` / `divElem` / `powElem` /
  `addElem` / `expElem` / `logElem` / … impls. Flat row-major loops at
  any rank; no per-cell `evaluateExpr` reentry.

`_broadcastApply` remains as the cold path for everything the fast
path doesn't catch — see "What's still on the cold path" in
`flatppl-dev/TODO-flatppl-js.md` for the full audit. Most cold-path
cases are fusable in principle (including kernel-broadcast — the
outer broadcast axis can be pushed through the kernel body's ops
the same way value-broadcast pushes through a `functionof` body);
they're cold TODAY pending dissolution-side work.

`valueOps` is the canonical batched-primitive surface:
- Elementwise ops (add, sub, neg) operate on flat row-major data
  regardless of rank; the leading dims (atom axis + user broadcast
  axes) collapse uniformly into the elementwise loop.
- Reductions (sum, mean, var, …) take an explicit axis argument
  (the dissolver will pass axis indices once Phase 5 lands).
- Matrix ops (matmul, dot, …) handle batched shapes via leading
  dim broadcast (NumPy convention: trailing dims contracted; all
  leading dims broadcast).

The Klein-4 transpose tag and structured-matrix tags remain
**runtime-internal optimisation markers**, hidden behind the IR
contract. valueOps reads them where they help (free transpose
via tag flip; sparse-storage exploitation for diag / lower /
upper); op-boundary handoffs materialise to contiguous form.

### Constants and the rank-0 path

`fixedValueToMeasure` for finite scalars now carries the canonical
rank-0 Value at `m.value` (Phase 1 — engine commit 919dab7). The
length-N `.samples` buffer is kept transitionally for legacy
histogram / viewer consumers; future work retires it via a
`samplesAsLengthN(m, N)` helper when needed at consumers (lands
with Phase 6 atom-axis unification).

The downstream effect: any binding that reads a fixed-phase
constant via `valueOf(m)` sees a rank-0 Value, and arithmetic
through ARITH_OPS / value-ops handles rank-0 × rank-N broadcasting
via NumPy-style stride-0 reads on the smaller operand.

### Status + follow-ups — see TODO

Phases 1–4 of the migration are in. Remaining work (DISSOLVE_SAFE_OPS
widening, Phase 5 aggregate dissolution, Phase 6 atom-axis
unification, and the `.samples` length-N retirement) is tracked in
`flatppl-dev/TODO-flatppl-js.md` under "Broadcast / aggregate
dissolution and value-contract unification". Each remaining phase
has exit criteria + cross-cutting impacts so a future session can
execute one phase without re-reading the rest.
