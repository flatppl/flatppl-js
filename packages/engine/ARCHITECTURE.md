# `@flatppl/engine` — Architecture

Deep-dive companion to the root [`AGENTS.md`](../../AGENTS.md), covering the
engine package only. For the language semantics it implements, read the spec
in **flatppl-design** (resolution order in `AGENTS.md`); for cross-engine
architectural concepts (the shape contract, FlatPDL, dissolution, the
multivariate-as-derived-measure doctrine), read
`flatppl-dev/flatppl-engine-concepts.md` — this doc is the JS-specific
realisation and does not repeat that rationale.

## Status

Reference implementation of FlatPPL v0.1. All sources TypeScript
(`strict: true`). The engine drives the VS Code visualizer and the web
gallery end-to-end; the measure-algebra core is feature-complete for the
spec's Bayesian / measure-theoretic vocabulary. Univariate + multivariate
distributions (MvNormal, Dirichlet, Multinomial, Wishart, InverseWishart,
LKJ / LKJCholesky, BinnedPoissonProcess), the FlatPDL measure-eval primitives
(`builtin_logdensityof` / `builtin_sample` / the four transports), `metricsum`
(metric-aware Einstein summation), and spec §03 vec-of-vec ≠ matrix
enforcement all landed. The §22 multivariate-as-derived-measure arc
(engine-concepts §22) is live: static-D `MvNormal` lowers at lift time to
`pushfwd(affine, iid(Normal, D))` and dispatches through the bijection
registry on both sample and density sides; `matMvNormal` / `density-prims`
persist as the terminal materialisers for the cases the lift gate can't
statically lower (dynamic-shape cov, matrix-form mean, positional form).

Outstanding work (the single source of truth is
`flatppl-dev/TODO-flatppl-js.md`): standard modules (particle-physics, GLM,
ext-linear-algebra, special-functions), `PoissonProcess`, multi-file
`load_module` end-to-end, multivariate transports beyond MvNormal, and finer
typeinfer coverage. Test count + per-area status live in the TODO + `git log`,
not here.

## Pipeline overview

```
source → tokenizer.ts → parser.ts → analyzer.ts → pir.ts (LoweredModule)
         → typeinfer.ts (+ fixed-eval.ts) → orchestrator.ts
         → materialiser.ts → worker.ts → sampler / density / traceeval
```

```
┌──────────────┐  source text
│ tokenizer.ts │
└──────┬───────┘ Token[]
┌──────▼───┐
│ parser.ts │  recursive-descent precedence climbing → AST
└──────┬───┘
┌──────▼─────┐
│ analyzer.ts│  bindings, deps, classification, phase analysis, multi-LHS
└──────┬─────┘  rewrite, disintegrate detection, restrict expansion
       ├── dag.ts                          ── ancestor / full DAG (viewer)
       ├── disintegrate.ts                  ── structural disintegration plans
       │ analyzer.bindings (Map<name, BindingInfo>)
┌──────▼──────┐
│   pir.ts    │  LoweredModule = ordered LoweredBinding(rhs = PIR-JSON)
└──────┬──────┘
       ├── lower.ts        ── AST → FlatPIR-JSON
       ├── types.ts        ── type constructors, unify, signatureOf
       ├── typeinfer.ts + fixed-eval.ts  ── inferTypes; demand-driven
       │                                    const-eval at shape positions
       ├── pir-sexpr.ts    ── FlatPIR S-expr printer + reader
┌──────▼─────────┐
│ orchestrator.ts│  buildSampleChain, buildDerivations, scope materialisation
└──────┬─────────┘  (facade over ir-shared / lift / derivations / signatures /
       │             profile-plan; dissolver runs here after lift)
       │ derivations: Map<name, Derivation> + fixedValues
┌──────▼──────────┐
│ materialiser.ts │  per-kind handlers → EmpiricalMeasure
└──────┬──────────┘  (samples + value + logWeights + n_eff + logTotalmass)
       │ worker messages (postMessage)
┌──────▼──────┐
│ worker.ts   │  thin shell over sampler / density / traceeval
└─────────────┘
```

Each layer is independently testable; the suite covers per-module unit tests
plus broad cross-cutting suites (`closed-form-measure-algebra`,
`measure-algebra`, `broadcast-semantics`, `kernel-broadcast`,
`jointchain-first-class`, …).

## Public entry, bundles, multi-file

`processSource(source, opts?)` runs tokenize → parse → analyze and returns
`{ ast, bindings, loweredModule, symbols, diagnostics, variant, bundle }` — the
extension-host entry point. `opts`: `variant` (canonical FlatPPL is the only
surface form, spec §05; seam retained for forward-compat), `path` (reserved),
`bundle` (`{ sources: { [path]: text } }` of pre-resolved `.flatppl` deps —
the engine pipeline stays **synchronous**; async I/O is the host's job:
`vscode.workspace.fs` in the extension, `fetch()` in the web host).

**`sampler.ts` / `worker.ts` are deliberately NOT re-exported** from the main
entry — they pull in ~1 MB of stdlib distribution code only needed in the
worker bundle. Host/main-thread code that needs to sample drives the worker
via postMessage.

**Four bundle entry points.** (1) `engine.min.js` (`engine/index.ts`) — main
entry; excludes sampler/worker. (2) `sampler-worker.min.js`
(`engine/worker-entry.ts`) — transport shim → `worker.createWorkerHandler`;
detects Web Worker vs Node `worker_threads`; bundled with full stdlib. (3)
`viewer.js` (`viewer/src/index.ts`) — visualization (cytoscape + echarts);
includes engine, loads sampler-worker externally. (4) `vscode-extension.js` —
editor integration.

**Multi-file models (decisions locked 2026-05-10).** Engine stays sync (bundle
is pre-resolved source text). Eager bundling for `.flatppl` sources, lazy
fetching for data (separate `dataResolver(path)` callback). Modules don't
flatten — each compiles to its own `LoweredModule`; cross-module access
(`mod.x`) lowers to `(ref mod x)`; the viewer navigates between module DAGs.
Standard modules are engine-provided (a JS registry, not `.flatppl` text).
Currently implemented: bundle-shape API plumbing only; cross-module resolution,
`load_module` end-to-end, and the std-module registry wiring are TODO items.

## Bundle build gotcha — ESM/CJS mixing (load-bearing)

esbuild infers each engine source file's module format from its syntax. A file
must be **pure CJS** (no top-level `export`/`import`; runtime exports via one
`module.exports = {…}`) or **pure ESM** (every binding `export`ed; no
`module.exports`). The trap: a file with a top-level `export interface Foo {…}`
(types-only) that *also* ends with `module.exports = {…}` makes esbuild see
`export`, classify the file ESM, and emit `[WARNING] The CommonJS "module"
variable is treated as a global variable in an ECMAScript module`. The build
succeeds and loads fine in Node + permissive browsers, then **crashes in the
VS Code webview's strict-CSP Chromium sandbox** (`module is not defined`) — the
engine global never initialises and the viewer stays blank with no engine-side
error. Fix: drop the `export` from types-only `interface`/`type` in a CJS file
(they're already visible within the file); if a type must be importable, make
the whole file pure ESM. **Treat the `commonjs-variable-in-esm` warning as
fatal** (`--log-override:commonjs-variable-in-esm=error`) — it catches the
regression at build time rather than at webview load, where the symptom is far
from the cause.

## Module reference

Line counts are approximate size hints. Bug-preventing caveats are called out
after the table.

| Module (~lines) | Responsibility · key exports |
|---|---|
| `tokenizer.ts` (~300) | source → Token[]. Tracks paren/bracket depth (implicit line continuation); distinguishes `_` HOLE / `_name_` PLACEHOLDER / identifier at lex time. |
| `parser.ts` (~750) | Token[] → AST, recursive-descent precedence climbing. `(x)` parens, `(x,y)` tuple, `(x,)` rejected. MixedArgs admitted (analyzer flags ops disallowing it). |
| `ast.ts` (~150) | AST node factories (`type` discriminator + `loc`). `synthLoc` for engine-synthesized nodes. |
| `builtins.ts` (~300) | **Single source of truth** for built-in names (constants, sets, special ops, functions, distributions, measure ops). Drives lower's builtin-vs-user dispatch + analyzer's unknown-name diagnostic. `tuple_get` listed here though engine-internal. |
| `analyzer.ts` (~2500) | AST → `bindings: Map<name, BindingInfo>`. 8 passes: collect names → classify + validate + deps → disintegrate plans → multi-LHS rewrite (`a,b = call` → `tuple_get`) → `computePhases` → `pir.lowerToModule` → `typeinfer.inferTypes` → bin diagnostics onto bindings. `isMeasureExpr` predicate (historical dup of types' `isMeasure`). |
| `dag.ts` (~800) | `computeSubDAG` / `computeFullDAG` for the cytoscape renderer; materialises reified-callable scopes as "bubbles" with scope-local phase colouring. **Gotcha:** synthetic node IDs use `:` separator (`binding:boundary`); the renderer detects scope-local nodes via `id.indexOf(':')` — fragile to refactor. |
| `disintegrate.ts` (~880) | Structural disintegration of joints → `(kernel, prior)`: `Synthesized` / `Delegate` / `Unsupported`, factored `decompose → partition → admissibility → synthesize` (engine-concepts §8). Hosts `disintegratePlan`. `restrict(M, x)` (spec §06) expands here + in analyzer pass 3 as a pure structural rewrite over disintegrate + bayesupdate + likelihoodof — no new derivation kind. |
| `lower.ts` (~800) | AST expr → FlatPIR-JSON (pure). `kernelof(x,kw)` → `functionof(lawof(x),kw)`; `fn(holes)` → functionof with `_argN_` placeholders; operators desugar to calls (`BIN_OP_MAP`/`UN_OP_MAP`); ordered-named forms (record/joint/jointchain/cartprod/table) use `fields:[{name,value}]`. IR-JSON shapes: `lit`/`const`/`ref`/`hole`/`call`(builtin or `target`)/`functionof`/`load_module`. |
| `pir.ts` (~250) + `pir-sexpr.ts` (~600) | `LoweredModule` (insertion-ordered `bindings`, `publicSet`, `source`) + `LoweredBinding` (`rhs` PIR-JSON, `originLoc`/`originName`, `inferredType`, `phase`). `walkCalls` post-order visitor (args/kwargs/fields/body). `pir-sexpr` round-trips `.flatpir` ↔ LoweredModule (spec §11); `%meta` + cross-module `(%ref mod name)` are TODO. |
| `types.ts` (~800) | Type constructors (`deferred`/`failed`/`any`/`scalar`/`array(rank,shape,elem)`/`record`/`tuple`/`measure(domain,opts?)`/`funcType`/`kernelType`/`rngstate`/`tvar`) + `unify` (deferred/any unify with anything; failed never unifies; scalar promotion `bool⊂int⊂real→complex`) + `SIGNATURE_FACTORIES` (per-call fresh tvars). `measure.opts` carries optional `{sampleShape,batchShape,eventShape}` — purely additive. ~90 ops covered; gaps in TODO §07/§17. |
| `typeinfer.ts` (~1600) + `fixed-eval.ts` (~250) | Inference over a LoweredModule; sets `inferredType` + per-call `meta.type`. `inferTypes(module)`; `inferExprInScope` (viewer profile-plot). Special-cased ops + `inferReification` (body in extended scope → kernel/func) + monomorphic-at-definition `inferUserCall` + auto-splat detection. **`fixed-eval`**: demand-driven const-eval at shape positions (engine-concepts §17.1) — lazy resolver, shape-observer short-circuit (`length`/`sizeof` read off inferredType), side-table only. **Invariant: typeinfer never changes bytes in the IR.** |
| `orchestrator.ts` (~460) + splits | `buildSampleChain`, `buildDerivations` (→ `{derivations, discrete, bindings, fixedValues}`), `signatureOf`, profile-plot range derivation, scope materialisation. Facade over 5 one-way-dependency modules (below). |
| `ir-shared.ts` | Dependency ROOT: constant folding (`resolveConstant`), IR→value (`resolveIRToValue`, `valueToPlain`), `collectSelfRefs`, set parsing, measure-IR canonicalisation, and the **static gates** `SAMPLEABLE_DISTRIBUTIONS` / `DISCRETE_DISTRIBUTIONS` / `EVALUABLE_OPS` / `VECTOR_OUTPUT_DISTRIBUTIONS`. |
| `lift.ts` | Inline-subexpression lifting (`liftInlineSubexpressions`, `isEvaluable`). Hosts `inlineMvNormalLift` (lift-time MvNormal → `pushfwd(affine, iid)`) and `inlineMetricsumLift` (metricsum → aggregate + metric factors, Form-B). |
| `derivations.ts` | Measure-algebra heart: `buildDerivations`, `classifyDerivation`, `MEASURE_OP_CLASSIFIERS`, the unified `expandMeasure(input, ctx, visited)` walker (replaces the former 4-walker maze; two back-compat shims survive). 15 derivation kinds: alias / sample / mvnormal / iid / record / tuple / superpose / select / jointchain / weighted / normalize / pushfwd / kernelbroadcast (+ structural fallback). |
| `signatures.ts` / `profile-plan.ts` | Callable introspection + profile-plot range/preset derivation for the viewer. |
| `materialiser.ts` (~2800) | Per-binding-name → `EmpiricalMeasure`. `KIND_HANDLERS` (~27: measure-algebra kinds + multivariate dists + FlatPDL surface). Bijection-binding contract (`binding.bijection.{registryName, paramIRs}`, purely additive over `fName`/`fInvName`/`logVolume`); consumer fast paths `matPushfwd` / `density.walkPushfwd` (vector-atom base, outerRank=1). `mat-broadcast` composite executors (bare-vector-output / joint / nested / jointchain). **Batch-flatten** (`_executeIidCompositeBatchFlatten`, engine-concepts §20.10): an iid-bodied kernel-broadcast folds its three parallel axes (atom N × cell K × inner-iid D) into ONE `count = N·K·D` batch — every input laid out as a single flat `[count]` buffer in (i,j,r) order (`_layoutFlat`, size-1 broadcast per axis), the inner dist's params evaluated once, ONE `sampleN`, result already in `[N,K,D]` atom-major order. One worker call replaces the former K (scalar-per-cell) / K·D (vector-per-cell) per-cell calls; the within-atom conditional independence (spec §06) falls out of the layout (a per-atom draw, kind 'N', lays out constant along (j,r)). **matIid repeat axis**: `iid(composite, k)` re-enters the pipeline at inflated count `N·k` with `ctx.repeatBlock=k`; shared atom-level value draws tile ×k (`tileMeasureAtomMajor`, atom-major) while the measure subtree redraws freshly — within-atom conditional independence (spec §06, engine-concepts §22.4). Resampling handlers honour `repeatBlock` (matSuperpose resamples within each k-block); order-preserving handlers ride tiling alone. Shared plumbing `prepareDensityRefs` / `measureToRefValue` / `tileMeasureAtomMajor` / `pushFixedEnv` / `collectRefArrays` (one owner; auto-pushes fixed refs to worker session env). |
| `bijection-registry.ts` (~330) | Single point of bijection support (engine-concepts §22): per-entry `atomBatchedForward` (sample) / `atomBatchedInverse` + `logDetJ` (density) / shape contract. `affine` entry done (covers MvNormal/MvStudentT/MvLogNormal); accepts atom-batched `b=[N,D]` / `L=[N,D,D]` via stride-0 reuse. Adding a bijection = one entry. |
| `density.ts` (~1400) + `density-prims.ts` (~1100) | Single density implementation. `logDensityConsumeN(ir, value, refArrays, count, opts)` foundation + 3 wrappers; `OP_HANDLERS` consume/rest table (engine-concepts §4) over 11 measure ops + 8 multivariate kernels (via `walkMultivariate` → `density-prims.builtinLogdensityof`). `density-prims` is the FlatPDL `builtin_logdensityof` ABI + the four transports; MvNormal density routes through the `affine` registry entry. Also hosts the type-mode `staticConsume` / `staticDensityShapeCheck` (§17). |
| `value.ts` (~700) + `value-ops.ts` (~1100) | The shape-tagged `Value` (engine-concepts §2.1): `shape`/`data`/`dtype`/`im`/`t` (Klein-4)/`struct`/`outerRank`. `transpose`/`adjoint`/`conjugate` toggle tag bits (no storage). `struct` bitmask (§2.2). `requireMatrix(v, op)` gates matrix-input ops (refuses nested-vector, spec §03); `promoteNestedToMatrix` is the sanctioned tag-stripper; `densify(v)` the one structured→dense fallback (**correctness never depends on a fast-path existing**). `value-ops`: shape-aware add/sub/neg/mul (+ batched `mulN`/`addN`), structured Cholesky/det/inv, complex (planar). |
| `worker.ts` (~430) + `worker-entry.ts` (~90) | Stateless message handler over sampler/traceeval; transport shim sniffs Web Worker vs Node. Messages: init/sample/density/evaluate/sampleN/evaluateN/logDensityN/profileN/dispose (transferable-aware). `sampleN` static-params vs parametric path (parametric is critical — naive per-draw factory rebuild makes setup ~10× the sampling cost). |
| `sampler.ts` (~2800) + splits | Single-point deterministic evaluator (`evaluateExpr`/`evaluateCall`), `ARITH_OPS`, worker entry points (`rand`/`makeSampler`/`makeParametricSampler`/`density`). Split: `sampler-registry` (REGISTRY + custom Ctors + PRNG bridge), `sampler-complex` (ESM leaf), `sampler-linalg` (ESM leaf — nested-array + Value-native LU/Cholesky/etc.), `sampler-aggregate` (CJS — `AGGREGATE_PATTERNS` + broadcast-reduce default), `sampler-eval-batched` (CJS — `evaluateExprN` + `ARITH_OPS_N` + complex-batched). ESM=pure-leaf, CJS=needs require()-back-to-sampler cycle (lazy, non-memoised). |
| `ops.ts` (~250) + `ops-declarations.ts` (~650) | Unified op-declaration registry + atom-batched dispatcher (engine-concepts §18). 20 ops across fixed-rank / rank-polymorphic / variadic / higher-order kinds; 3 batched fast-paths (cross/inv/lower_cholesky). **Shape-pattern variants** (`OpVariant` with `argPatterns` + `wrappingOp` + specificity scoring): 19 broadcasted primitives + 11 `mul` direct variants register as variants; `dispatch`/`dispatchVariant`/`dispatchTyped`(ArgInfo value+measure). |
| `dissolver.ts` (~750) | Broadcast/aggregate dissolution term-rewriter (engine-concepts §20; JS specifics below). `dissolveBindings` runs in `buildDerivations` after lift; rewrites `.ir` in place. Phases 1–6 + shape-fold + fusion (a)/(b) MVPs + `propagateAxisStack` (authoritative for the iid / kernel_broadcast / aggregate axes it records; consumed by `axis-stack.ts` → matIid repeat axis + mat-broadcast K). |
| `traceeval.ts` (~400) | Unified sample/score walker for measure IR (`tally` arg); `MEASURE_OP_WALKERS` table. |
| `empirical.ts` (~700) | Weighted empirical measure utilities (log-space, ESS, resample). Null-uniform-weight protocol (null = uniform, no alloc); `logSumExp` max-subtraction; `totalLogMass` = 0 for null. |
| `histogram.ts` (~350) | Freedman-Diaconis / integer-atom binning, weighted histogram + quantile + plot range. |
| `rng.ts` (~400) | Pure Philox-4x32-10 counter-PRNG (`(key,counter)→4-tuple`, no sequential state — reproducible, parallel, cross-impl parity). 32×32→64 via 16-bit decomposition (no BigInt on the hot path). |
| `dataload.ts` (~400) | `load_data` parsers: JSON / CSV / WSV (pure, no I/O; host fetches bytes). Arrow IPC deferred. |
| `standard-modules.ts` (~250) | Std-module registry skeleton (spec §09). Exact (name, compat) match; analyzer/orchestrator wiring pending. |
| `engine-types.d.ts` | Cross-file type declarations (`Value`, `IRNode` union, `DerivationKind`, `EmpiricalMeasure`, `BindingInfo`, …). Re-exported as `@flatppl/engine/engine-types`. |

**`EmpiricalMeasure` shape** (engine-concepts §2 universal value): `samples`
(per-atom scalar view) · optional `value` (shape-`[N,…dims]` batched Value) ·
`logWeights` (null ⇒ uniform) · `logTotalmass` (0 ⇒ probability) · `n_eff` ·
optional `fields` (record) / `elems` (tuple) / `dims`. Composite measures
recurse via `ctx.getMeasure(name)`; the orchestrator caches per name-per-context.

**Deterministic-evaluation authority.** There is exactly ONE deterministic
evaluator: `sampler.evaluateExpr` / `evaluateExprN`. Everything routes through
it; `orchestrator.resolveIRToValue` is a thin adapter (static fast paths for
lit/neg/vector/record/ref, delegates everything else). **Invariant: do not grow
a parallel mini-interpreter** — when the evaluator misses an op, implement it in
`evaluateCall`, not in a caller. A fixed-phase binding that hits the residual
throw is surfaced loudly by `buildDerivations`' fixed-phase-dead-end diagnostic.

## Cross-file invariants

These lists must agree across files; drift produces silent runtime failures.
The cleanly-checkable ones ARE enforced by `test/invariants.test.ts`
(distribution ↔ REGISTRY both ways, discrete flag, EVALUABLE_OPS ↔ ARITH_OPS,
BIN_OP_MAP/UN_OP_MAP signatures, REGISTRY params ↔ types kwargs, sampleable/
discrete ⊆ `builtins.DISTRIBUTIONS`) — a failing test names the mirror to
update. The `MEASURE_OP_*` catalogs (`MEASURE_OP_CLASSIFIERS`, traceeval
`MEASURE_OP_WALKERS`, density `OP_HANDLERS`, materialiser `KIND_HANDLERS`) are
intentionally NOT invariant-tested (cross-cut by structural vs algebraic vs
density-routed handling — a clean equivalence would be false); watch them by
hand. The unified `ops.ts` declaration model (engine-concepts §18) collapses
much of this drift surface for declared ops.

### Adding a new built-in distribution (`Foo`)

| File | What to add |
|---|---|
| `builtins.ts` | `'Foo'` in `DISTRIBUTIONS` (+ keep `MEASURE_PRODUCING` in sync) |
| `types.ts` | `Foo: () => realDistKwargs({...})` in `SIGNATURE_FACTORIES` |
| `ir-shared.ts` | `'Foo'` in `SAMPLEABLE_DISTRIBUTIONS` (+ `DISCRETE_DISTRIBUTIONS` if discrete) |
| `sampler.ts` | `Foo: {params, aliases, discrete, Ctor, randFn, logpdfFn}` in `REGISTRY` + stdlib `require`s |
| `density-prims.ts` | per-kernel logpdf + `MV_DENSITY_FNS` entry (multivariate only) |
| `density.ts` | `OP_HANDLERS[Foo] = walkMultivariate` (multivariate only) |
| `materialiser.ts` | `KIND_HANDLERS[Foo] = matFoo` + handler (multivariate only) |
| `test/sampler.test.ts` & friends | regression tests |

The `types.ts` kwargs MUST equal the `sampler.REGISTRY[Foo].params` (and match
the spec) — `test/invariants.test.ts` enforces this. (Under the §22 doctrine, a
new *multivariate* dist preferably lands as a lift-time lowering to its
`pushfwd`-of-iid decomposition + one bijection-registry entry, not a new
`matFoo`.)

### Adding a new evaluable built-in function (`bar`)

| File | What to add |
|---|---|
| `builtins.ts` | `'bar'` in `BUILTIN_FUNCTIONS` |
| `types.ts` | signature in `SIGNATURE_FACTORIES` |
| `ir-shared.ts` | `'bar'` in `EVALUABLE_OPS` |
| `sampler.ts` | `bar` in `ARITH_OPS` (single-point) + `ARITH_OPS_N` (batched) if numeric |
| `test/sampler.test.ts` | regression |

`EVALUABLE_OPS` and `ARITH_OPS` must contain the same op names (enforced both
ways by `invariants.test.ts`, modulo documented `vector`/`cat` exemptions).

### Adding a new measure-algebra op (`baz`)

| File | What to add |
|---|---|
| `builtins.ts` | `'baz'` in `MEASURE_OPS`; possibly `MEASURE_PRODUCING` |
| `types.ts` | signature (+ `special: 'baz'` if structurally unusual) |
| `typeinfer.ts` | special-case handler in `inferCall` if `special` set |
| `lower.ts` | `'baz'` in `FIELD_FORMS` if it has ordered named entries |
| `derivations.ts` | `classifyBaz` + `MEASURE_OP_CLASSIFIERS` entry + `_expandByName` arm |
| `traceeval.ts` | walker + `MEASURE_OP_WALKERS` (sampling) |
| `density.ts` | walker + `OP_HANDLERS` (scoring) |
| `materialiser.ts` | handler + `KIND_HANDLERS` |
| `disintegrate.ts` | dispatch entry if it can appear in a joint-measure RHS |
| `test/measure-algebra.test.ts` + `closed-form-measure-algebra.test.ts` | regression |

### Catalog source-of-truth → mirror

| Catalog | Source of truth | Mirrored in |
|---|---|---|
| Built-in name set | `builtins.ALL_KNOWN` | lower's user-vs-builtin dispatch |
| Sampleable dists | `sampler.REGISTRY` + `mat*` handlers | `ir-shared.SAMPLEABLE_DISTRIBUTIONS` |
| Discrete dists | (subset) | `ir-shared.DISCRETE_DISTRIBUTIONS` |
| Evaluable fns | `sampler.ARITH_OPS`/`ARITH_OPS_N` | `ir-shared.EVALUABLE_OPS`, typeinfer handlers |
| Type signatures | `types.SIGNATURE_FACTORIES` | (only place) |
| Measure-producing ops | `builtins.MEASURE_PRODUCING` | `analyzer.isMeasureExpr`, `typeinfer.isMeasure` |
| Derivation kinds | `derivations._expandByName` | `materialiser.KIND_HANDLERS`; `engine-types.DerivationKind` |
| FlatPDL primitive surface | `density-prims.ts` | `sampler.ts` (`builtin_sample`); spec §07 |

## Engine-internal vs. spec FlatPIR

Engine extensions (allowed; not in the spec): `tuple_get` (multi-LHS
projection, in `SPECIAL_OPERATIONS`); `get_field` (from `obj.field` — the spec's
only access op is `get`; emitted so typeinfer dispatches on the literal name;
convertible back); `%mlhs:line:col` synthetic bindings (`%` can't collide with
user names; `synthetic: true`); `paramSources` on reified callables (Identifier
vs Placeholder boundary kwarg — viewer plot ranges); `numType` on lit nodes
(`1` integer vs `1.0` real); `originLoc`/`originName` on LoweredBindings
(diagnostics/DAG back-refs).

## Phase analysis

Per spec §04: `'fixed'` / `'parameterized'` / `'stochastic'` by ancestor
analysis (`analyzer.computePhases`, memoised + cycle-guarded). Standard rule:
max of deps' phases (`stochastic > parameterized > fixed`). Special: `draw`
self → stochastic; `elementof` → parameterized; `external` → fixed; `lawof`
absorbs stochasticity (via `absorbedPhaseOf`, collapsing stochastic to the
underlying phase — spec §sec:lawof); `functionof`/`kernelof`/`fn` → fixed
(per-call body phase via `computePhasesForScope` for DAG colouring).
Degenerate-Dirac sharpening: `draw(Dirac(value=e))` / `draw(lawof(e))` →
phase(e). Inline draws (`s = 2 * draw(m)`) caught by `rhsContainsInlineDraw`
(doesn't recurse into reification bodies — different scope).

## Fixed-phase pre-eval and `fixedValues`

Per spec §04, fixed-phase bindings are compile-time-determinate.
`buildDerivations` runs an iterative (fixed-point) pre-eval pass over
fixed-phase bindings in topological order via `sampler.evaluateExpr`, exposing
`fixedValues: Map<name, JSValue>`. This matters whenever a fixed value isn't a
per-atom `Float64Array` slice: compile-time-known-length numeric arrays
(`rand(rstate, iid(Normal,10))` → length 10, not `N`), records/tuples, opaque
values (rngstate). Per binding: walk self-refs (recursing measure subtrees via
`expandMeasure`), skip measure-typed refs (resolved by traceeval at runtime),
require value refs in `fixedValues` (else defer), `evaluateExpr` (catch+skip,
retry next iteration). **Cascade-prune** (`derivationRefsValid`, runs after
pre-eval) accepts refs resolving through `fixedValues`, not just refs with their
own derivations. **Viewer**: pushes `fixedValues` to the worker via `setEnv` on
every `rebuildDerivations`; `collectRefArrays` drops fixed-phase refs (so the
per-atom path doesn't shadow the session value); `getMeasure` short-circuits any
binding in `fixedValues` (synthesising the SoA shape). Opaque values aren't
plottable (`buildPlotPlan` returns null for `rngstate`).

## Type inference details

Per spec §11. `typeinfer.ts` operates on a LoweredModule; mutates
`inferredType` + writes `meta.type`. Inference must succeed on well-formed
modules — when it can't resolve, the type is `failed(reason)` and cascades via
"failed never unifies." Built-in signatures use type variables, freshly keyed
per `signatureOf` call. User-defined callables are monomorphic-at-definition
(stored `result` used directly; arg types still checked). Auto-splatting (a
single positional record arg whose fields ⊆ callee inputs) routes through the
kwarg path.

## Diagnostics

`{ severity: 'error'|'warning'|'information', message, loc: {start,end} }`
(0-based). Flow: tokenizer → parser → analyzer → typeinfer; `processSource`
concatenates. Analyzer pass 8 bins each diagnostic onto its binding
(`binding.diagnostics`) so the DAG view answers "does this binding have an
error?" without re-walking.

## Test layout

`packages/engine/test/` (~100 files): per-module unit tests; distributions
(`multivariate-distributions`, `multivariate-density`, `builtin-logdensityof`,
`builtin-sample-transport`, `hierarchical-multivariate-density`); measure
algebra (`measure-algebra`, `closed-form-measure-algebra`,
`broadcast-semantics`, `kernel-broadcast`, `jointchain-first-class`); type
inference (incl. `-conformance-property`); linear algebra; cross-cutting
(`invariants`, `equivalence`, `phases`, `spec-patterns`,
`spec-vec-of-vec-not-matrix`); `integration.test.ts` against `.flatppl`
fixtures copied from **flatppl-examples** into `test/fixtures/` (no auto-sync —
refresh copies when originals change). Run: `npm --workspace=@flatppl/engine run
test`.

## Where to look for X

| Concern | Start here |
|---|---|
| "Why is this name unknown?" | `builtins.ts`; `analyzer.ts` (undefined-name warning) |
| "Why does this expr have type X?" | `typeinfer.ts` (`inferCall`); `types.ts`; `fixed-eval.ts` (shape positions) |
| "What phase is this binding?" | `analyzer.ts` `computePhases` |
| "Why is the chain unsupported?" | `orchestrator.ts` `classifyForChain` |
| "Why no derivation for this binding?" | `derivations.ts` `buildDerivations` + classifiers |
| "Canonical measure IR for this binding?" | `derivations.ts` `expandMeasure(name, {derivations, bindings})` |
| "How is this distribution sampled?" | `sampler.ts` REGISTRY; multivariate: `materialiser.ts` `mat*` |
| "How is its density computed?" | `density.ts` `OP_HANDLERS`; `density-prims.ts` per-kernel math |
| "How is this measure-algebra op handled?" | `derivations.ts` classifier + `_expandByName`; `materialiser.KIND_HANDLERS`; `density.OP_HANDLERS`; `traceeval.MEASURE_OP_WALKERS` |
| "Why does the DAG render this way?" | `dag.ts` |
| "Why did disintegrate produce this Plan?" | `disintegrate.ts` (`disintegratePlan`) |
| "Why is the RNG result this number?" | `rng.ts` (Philox); `sampler.ts` PRNG adapter |
| "How are refs in a measure body resolved?" | `materialiser.ts` `prepareDensityRefs`; `sampler.ts` `evaluateExprN` |
| "Empirical measure / Value shape?" | `engine-types.d.ts`; engine-concepts §2 / §2.1–2.2 |

## Known issues and gaps

Full list + priorities in `flatppl-dev/TODO-flatppl-js.md` (the checkable source
of truth — no status duplication here). Shortlist: multi-file `load_module`
end-to-end (bundle API landed; cross-module resolution + substitution +
host pre-fetch pending); standard modules (registry skeleton landed; analyzer/
orchestrator wiring + the named modules pending); `PoissonProcess` (needs a
ragged-vector-per-atom measure shape class); multivariate transports beyond
MvNormal (Dirichlet stick-breaking etc.); a few array-construction shape rules
(`fill`/`zeros`/`ones`/`eye`/`cartpow`/`cartprod`) + new stdlib fns still
`deferred()`; `viewer/src/viewer.js` (~5700 lines) is the next decomposition
target.

## Broadcast / aggregate dissolution (JS-engine specifics)

Cross-engine architecture in engine-concepts §20; JS notes here. The dissolver
runs in `buildDerivations` **immediately after `liftInlineSubexpressions`** (each
binding has cached lifted `.ir` + `inferredType`), rewriting `.ir` in place when
a dissolution applies; non-dissolvable broadcasts keep their IR and ride the
`_broadcastApply` cold path.

```
parse → analyze (incl. typeinfer) → lower → lift → DISSOLVE → derivations → materialiser
```

**Pattern + soundness.** Matches `broadcast(<head>, args…)` where `<head>` is an
inline `functionof` or a `self`-ref to one (Phase 4 inlining); `_substituteBody`
walks the body replacing `%local _argN_` refs with broadcast args. Each call
node must be in `DISSOLVE_AT_ANY_RANK_OPS` (value-ops handles elementwise at any
rank: add/sub/neg + declared fixed-rank linalg where `ops.dispatch`
auto-batches) or `DISSOLVE_SCALAR_ONLY_OPS` (mul/div/pow, scalar unaries,
comparisons, logic) AND all outer broadcast args scalar-typed. Outer args must
resolve to IDENTICAL inferred types + matching phase (the lazy `_resolveExprType`
handles binding refs + inline calls); mixed-shape / mixed-phase stay cold.
Iterates to a fixed point so `(a .* b) .+ c` dissolves end-to-end.

**Shape-fold** (`_foldShapeCall`, engine-concepts §20.10): folds
`lengthof`/`sizeof`/`indicesof`/`indicesof0` into literal IR when the input shape
is statically known (cap 4096; `%dynamic` / `%local` refuse) — the prerequisite
for fusion (a). **Fusion (a)** (`_tryDissolveBroadcastReduction`):
`broadcast(<head>, args)` whose body is `R(<plain elementwise>)` for R ∈
{sum,mean,prod} → `aggregate(R, [.atom], <body>)` (conservative MVP gates).
**Fusion (b)** (`_tryDissolveKernelBroadcast`): `broadcast(<user-kernel ref>,
args)` with body `functionof(lawof(<BuiltinDist>(kw)), …)` →
`broadcast(<BuiltinDist>, mapped_kwargs)`, letting `matKernelBroadcast` take its
per-atom path. **`propagateAxisStack`** annotates measure-op IR with outer-axis
`{source, size}` entries (iid / broadcast / kernel_broadcast / aggregate),
outer-to-inner; the atom axis is deliberately excluded (engine-internal). It is
**authoritative** (via `axis-stack.ts`) for the literal-size axes it records:
the matIid repeat axis and mat-broadcast's K trust it, the runtime shape-sniff
remaining only for symbolic ('%dynamic' / ref-name) sizes. Composites whose
variate shape isn't a single inner's (superpose / select / joint) carry no
stack by design.

**Runtime fast-path.** For broadcasts the dissolver leaves (or that the user
writes as `broadcasted(op)(args)`), `_maybeFastBroadcasted` recognises the
`broadcast(<scalar op>, args)` shape, coerces args to Values, checks all
non-rank-0 inputs share a shape (singleton-axis expansion falls to the cold
path), and dispatches through value-ops' `*Elem` primitives (flat row-major,
no per-cell `evaluateExpr` reentry). Spec-`mul` matrix semantics are preserved —
the fast path intercepts only `broadcast(...)` IRs, not direct `mul(A,B)`.
**Constants stay rank-0** (`fixedValueToMeasure` carries the rank-0 Value at
`m.value`; the length-N `.samples` buffer is transitional for legacy consumers);
arithmetic broadcasts rank-0 × rank-N via stride-0 reads. The Klein-4 + struct
tags stay runtime-internal markers behind the IR contract (free transpose via
tag flip; structured-storage exploitation; op-boundary handoffs materialise to
contiguous form). Remaining phases + the cold-path audit are in the TODO.
