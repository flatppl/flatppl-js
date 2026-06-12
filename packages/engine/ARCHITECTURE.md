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
(engine-concepts §22) is live and, for MvNormal, COMPLETE at the binding
level (5h-A): every lowerable `MvNormal` — static OR dynamic D
(`lengthof(<mu-ref>)` count, resolved by classifyIid's deferred pass),
kwarg or positional form, named or hoisted-inline args — rewrites at
lift time to `pushfwd(affine, iid(Normal, D))` and dispatches through
the bijection registry on both sample and density sides. `matMvNormal`
is a THIN refusal channel for the shapes the gate positively rejects
(matrix-form mean, visible dim mismatch — spec §08 violations). The
density-side MvNormal entries stay: `MV_DENSITY_FNS.MvNormal` is the
spec-§07 `builtin_logdensityof` FlatPDL ABI, and the `OP_HANDLERS`
walker scores raw MvNormal nodes inside composite kernel bodies
(formal-carrying mu/cov — the 5h-B scope).

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
         → materialiser.ts → worker.ts → sampler / density
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
│ worker.ts   │  thin shell over sampler / density
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
Currently implemented: bundle-shape API plumbing + cross-module resolution and
dispatch for **standard** modules (registry + typeinfer `resolveStandardModuleRef`
+ sampler `_evaluateStandardModuleCall`). Still TODO: `load_module` end-to-end
(`.flatppl` cross-module resolution + substitution + host pre-fetch).

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
| `tokenizer.ts` (~520) | source → Token[]. Tracks paren/bracket depth (implicit line continuation); distinguishes `_` HOLE / `_name_` PLACEHOLDER / identifier at lex time. |
| `parser.ts` (~1200) | Token[] → AST, recursive-descent precedence climbing. `(x)` parens, `(x,y)` tuple, `(x,)` rejected. MixedArgs admitted (analyzer flags ops disallowing it). |
| `ast.ts` (~150) | AST node factories (`type` discriminator + `loc`). `synthLoc` for engine-synthesized nodes. |
| `builtins.ts` (~300) | **Single source of truth** for built-in names (constants, sets, special ops, functions, distributions, measure ops). Drives lower's builtin-vs-user dispatch + analyzer's unknown-name diagnostic. `tuple_get` listed here though engine-internal. |
| `analyzer.ts` (~3500) | AST → `bindings: Map<name, BindingInfo>`. 8 passes: collect names → classify + validate + deps → disintegrate plans → multi-LHS rewrite (`a,b = call` → `tuple_get`) → `computePhases` → `pir.lowerToModule` → `typeinfer.inferTypes` → bin diagnostics onto bindings. `isMeasureExpr` predicate (historical dup of types' `isMeasure`). |
| `dag.ts` (~800) | `computeSubDAG` / `computeFullDAG` for the cytoscape renderer; materialises reified-callable scopes as "bubbles" with scope-local phase colouring. **Gotcha:** synthetic node IDs use `:` separator (`binding:boundary`); the renderer detects scope-local nodes via `id.indexOf(':')` — fragile to refactor. |
| `disintegrate.ts` (~880) | Structural disintegration of joints → `(kernel, prior)`: `Synthesized` / `Delegate` / `Unsupported`, factored `decompose → partition → admissibility → synthesize` (engine-concepts §8). Hosts `disintegratePlan`. `restrict(M, x)` (spec §06) expands here + in analyzer pass 3 as a pure structural rewrite over disintegrate + bayesupdate + likelihoodof — no new derivation kind. |
| `lower.ts` (~800) | AST expr → FlatPIR-JSON (pure). `kernelof(x,kw)` → `functionof(lawof(x),kw)`; `fn(holes)` → functionof with `_argN_` placeholders; operators desugar to calls (`BIN_OP_MAP`/`UN_OP_MAP`); ordered-named forms (record/joint/jointchain/cartprod/table) use `fields:[{name,value}]`. IR-JSON shapes: `lit`/`const`/`ref`/`hole`/`call`(builtin or `target`)/`functionof`/`load_module`. |
| `pir.ts` (~250) + `pir-sexpr.ts` (~750) | `LoweredModule` (insertion-ordered `bindings`, `publicSet`, `source`) + `LoweredBinding` (`rhs` PIR-JSON, `originLoc`/`originName`, `inferredType`, `phase`). `walkCalls` post-order visitor (args/kwargs/fields/body). `pir-sexpr` round-trips `.flatpir` ↔ LoweredModule (spec §11). Reified callables use the spec's fixed-arity form (`<output> %specinputs ((name ref)…)` \| `%autoinputs %deferred`); bodies are carried VERBATIM — engine-internal IR is spec-shaped (`%local` is placeholders-only; see "Engine-internal vs. spec FlatPIR"). `kernelof` is desugared to `functionof(lawof(…))` at BOTH ingest points (surface lower + PIR read); JS never emits it. `%meta` EMISSION landed (opt-in `toSexpr(mod, {meta:true})`; the reader parses-and-skips); %meta READER reconstruction, filled `%autoinputs`, and the cross-module `(%ref mod name)` reader resolution pass are TODO. |
| `types.ts` (~1400) | Type constructors (`deferred`/`failed`/`any`/`scalar`/`array(rank,shape,elem)`/`record`/`tuple`/`measure(domain,opts?)`/`funcType`/`kernelType`/`likelihood(inputs?)`/`rngstate`/`tvar`) + `unify` (deferred/any unify with anything; failed never unifies; scalar promotion `bool⊂int⊂real→complex`) + `SIGNATURE_FACTORIES` (per-call fresh tvars). `measure.opts` carries optional `{sampleShape,batchShape,eventShape}` — purely additive. ~185 ops covered; gaps in TODO §07/§17. |
| `typeinfer.ts` (~2700) + `fixed-eval.ts` (~250) | Inference over a LoweredModule; sets `inferredType` + per-call `meta.type`. `inferTypes(module)`; `inferExprInScope` (viewer profile-plot). Special-cased ops + `inferReification` (body in extended scope → kernel/func) + monomorphic-at-definition `inferUserCall` + auto-splat detection. **`fixed-eval`**: demand-driven const-eval at shape positions (engine-concepts §17.1) — lazy resolver, shape-observer short-circuit (`length`/`sizeof` read off inferredType), side-table only. **Invariant: typeinfer never changes bytes in the IR.** |
| `orchestrator.ts` (~460) + splits | `buildSampleChain`, `buildDerivations` (→ `{derivations, discrete, bindings, fixedValues}`), `signatureOf`, profile-plot range derivation, scope materialisation. Facade over 5 one-way-dependency modules (below). |
| `ir-shared.ts` | Dependency ROOT: constant folding (`resolveConstant`), IR→value (`resolveIRToValue`, `valueToPlain`), `collectSelfRefs`, set parsing, measure-IR canonicalisation, `isCallableLikeBindingType` (the ONE callable-binding catalogue — fn/functionof/kernelof/bijection/fchain; derivations re-exports it), and the **static gates** `SAMPLEABLE_DISTRIBUTIONS` / `DISCRETE_DISTRIBUTIONS` / `EVALUABLE_OPS` / `VECTOR_OUTPUT_DISTRIBUTIONS`. |
| `materialiser-shared.ts` (~1200) | Shared materialiser plumbing (one owner): `prepareDensityRefs` / `collectRefArrays` (boundary-feed nets, `_extraRefArrays` overlay), `measureToRefValue`, `tileMeasureAtomMajor` (repeat-axis tiling), `pushFixedEnv`, `resolveFnBody`, `isFunctionLikeBinding` (delegates to `ir-shared.isCallableLikeBindingType`). |
| `lift.ts` | Inline-subexpression lifting (`liftInlineSubexpressions`, `isEvaluable`). Hosts `inlineMvNormalLift` (lift-time MvNormal → `pushfwd(affine, iid)`) and `inlineMetricsumLift` (metricsum → aggregate + metric factors, Form-B). Also the composite-`rand` succession rewrite (engine-concepts §11): `classifyRandTuple` (the SHARED decompose + leaf-vs-composite gate used by both the index-0 draw classifier and the index-1 successor rewrite, so they can't drift) + `rewriteCompositeRandSucc` (post-pass after IR-cache + alias-resolution: a binding `tuple_get(<rand>,1)` over a COMPOSITE inner → the value-domain `rand_succ` op; leaf state halves stay `tuple_get`/threaded). Rewrites the BINDING IR so every consumer (FixedValues, resolveIRToValue) agrees. `inlineOnce` wraps a **kernelof** application in `lawof(...)` (a kernel application yields a measure = the law of the reified value, not the bare value — mirrors lower's `kernelof(x,kw)→functionof(lawof(x),kw)`; functionof/fn stay unwrapped). |
| `derivations.ts` | Measure-algebra heart: `buildDerivations`, `classifyDerivation`, `MEASURE_OP_CLASSIFIERS`, the unified `expandMeasure(input, ctx, visited)` walker (replaces the former 4-walker maze; two back-compat shims survive). 16 derivation kinds: alias / sample / mvnormal / iid / randsample / record / tuple / superpose / select / jointchain / weighted / normalize / pushfwd / kernelbroadcast (+ structural fallback). `randsample` (`classifyRandSample`) is the demand-driven composite-`rand` draw (engine-concepts §11, demand-driven composite rand): `tuple_get(rand(state, iid(<composite M>, n)), 0)` → materialise M at count n; leaf-dist inner stays on the batched-leaf path via pre-eval. Uses the shared `lift.classifyRandTuple` gate (the index-1 STATE half is handled upstream by lift's `rewriteCompositeRandSucc` → the `rand_succ` op, so the two halves can't drift). `buildDerivations` returns `fixedValues` as a `FixedValues` resolver (below). |
| `fixed-values.ts` (~280) | `FixedValues` — demand-driven, memoised, cycle-guarded resolver for fixed-phase VALUES (engine-concepts §17.1); Map-compatible (`.has`/`.get`/`.set`/iterate). Replaces the former eager `while (progress)` pre-eval sweep in `buildDerivations` (per-binding logic moved verbatim into `_compute`). Dependency-injected (zero engine imports). Load-bearing: **iterate-forces-all**, **no negative cache**. See "Fixed-phase values" below. |
| `signatures.ts` / `profile-plan.ts` | Callable introspection + profile-plot range/preset derivation for the viewer. |
| `clm.ts` (~650) | **Canonical Lowered Measure** — THE measure-lowering pass (measure-lowering unification, flatppl-dev plan): `lowerMeasure(name\|ir, ctx, opts?) → clm{body, inputs, reduce, mc?}`, the ONE lowering both the sample walker (`matClm`) and the density walker (`mat-density.matScore`) consume, making sampling ≡ density structural. Performs in fixed order: peel + expand (via `expandMeasure`), derived-value inlining to the boundary set (`inlineBoundaryDerivations`) OR the generative `buildMcMarginalForm`, input enumeration with shape/axis descriptors (boundary / shared / fixed / explicit / free). `feedInputs(node, ctx)` is the ONE feeding contract (reference-identity preserving). Phase-6 guarantees (both main-thread, plan critique F): `lowerMeasure` THROWS when a body self-ref is covered by no declared input (⊆ invariant — builtins + callables excluded, they resolve by name); `assertFedCoverage` (called by matScore after the overlay merge) THROWS when a declared boundary the body references (incl. `%local` refs) has no fed column. `opts.boundaries`/`opts.freeInputs` = the viewer kernel/profile explicit-boundary route (spec §04 substitution). |
| `materialiser.ts` (~2800) | Per-binding-name → `EmpiricalMeasure`. `KIND_HANDLERS` (~28: measure-algebra kinds + multivariate dists + FlatPDL surface). `matRandSample` materialises a composite `rand` draw in a child ctx (fresh cache + `sampleCount=count` + `rootKey` split off the rand state) — the demand-driven analogue of matIid's composite fallback (engine-concepts §11, demand-driven composite rand). Bijection-binding contract (`binding.bijection.{registryName, paramIRs}`, purely additive over `fName`/`fInvName`/`logVolume`); consumer fast paths `matPushfwd` / `density.walkPushfwd` (vector-atom base, outerRank=1). `mat-broadcast` composite executors. **Batch-flatten** (Phase 8, engine-concepts §20.10): all four composite kernel-broadcast bodies fold through ONE shared N-axis layout primitive (`_layoutFlat(v, lead, axes[], atomVaries, axisSizes[])` row-major odometer + `_spanOf` + `_resolveBatchFlattenParam`) driven by the axisStack ladder — lay every input out to `count = N·∏(parallel axes)`, evaluate the inner dist's params once, one `sampleN`, reshape: **iid** `_executeIidCompositeBatchFlatten` (axes `[K,D]`; within-atom conditional independence falls out of the layout — a per-atom draw, kind 'N', lays out constant along the cell/inner axes); **nested** `_executeNestedBroadcastBatchFlatten` (scalar inner; axes `[K_outer,K_inner]`, two-axis, mixing-tolerant) + `_executeNestedBroadcastVectorFold` (MvNormal inner — per-(outer,inner)-cell mu laid over `count = N·K_outer·K_inner`, shared cov → `[N,K_outer,K_inner·n]`); **joint** `_executeJointCompositeBatchFlatten` (product — one `sampleN` per component over `[K]`, concat → `[N,K,C]`); **jointchain** `_executeJointChainScan` (scan — one `sampleN` per step, carry = prev step's `[count]` column). One worker call (or one-per-component/step) replaces the former per-cell K·… loops. **All `_execute*PerCell` siblings are retired** — the folds are the only path (an unfoldable shape is a clear error, not a silent reroute); vector-output inner/component draws fold via the shared `_mvNormalFoldOverCells` affine (MvNormal → bijection registry §22, not a scalar `sampleN`). A 5th kind **generative** `_executeGenerativeComposite` (engine-concepts §21) handles `broadcast(K, …)` where K's body is `lawof(<value-expr with an internal draw>)` (a pushforward of a marginalized draw): inline the deterministic ancestor chain into one expr, sample each internal draw ONCE over `count=N·K` (fresh per (atom,cell), §22.4), one `evaluateExprN` over the inlined transform → `[N,K]` — reuses `_layoutFlat`/`_spanOf` for boundary columns but is NOT a single-dist `sampleN` like the four folds. Recognizer + near-miss arm in `kernel-broadcast-shape.detectGenerativeKernelBinding`; density: the sub-case bijective in one retained innovation + a single marginalised latent is handled by the `mcmarginal` rule (via `mc-recipe`, engine-concepts §6) — the genuinely-intractable remainder is still a loud spec-§06 refusal. **matIid repeat axis**: `iid(composite, k)` re-enters the pipeline at inflated count `N·k` with `ctx.repeatBlock=k`; shared atom-level value draws tile ×k (`tileMeasureAtomMajor`, atom-major) while the measure subtree redraws freshly — within-atom conditional independence (spec §06, engine-concepts §22.4). Resampling handlers honour `repeatBlock` (matSuperpose resamples within each k-block); order-preserving handlers ride tiling alone. Shared plumbing `prepareDensityRefs` / `measureToRefValue` / `tileMeasureAtomMajor` / `pushFixedEnv` / `collectRefArrays` (one owner; auto-pushes fixed refs to worker session env). |
| `bijection-registry.ts` (~620) | Single point of bijection support. **(a) Named multivariate entries** (engine-concepts §22): per-entry `atomBatchedForward` (sample) / `atomBatchedInverse` + `logDetJ` (density) / shape contract; `affine` done (covers MvNormal/MvStudentT/MvLogNormal), atom-batched `b=[N,D]` / `L=[N,D,D]` via stride-0 reuse; adding a bijection = one entry. **(b) Symbolic scalar inverse + LADJ** (spec §06 case-1; the InverseFunctions + ChangesOfVariables analogue over FlatPIR): `ELEMENTARY_BIJECTIONS` per-op inverse/forward-LADJ rules (exp/log/neg/add/sub/mul/divide/pow-literal, with Fix1/Fix2 free-arg read structurally) + `invertExpr({outputExpr,freeRef,outputValue})→{inverseIR,ladjIR}` (straight-line path inversion; output is value IR → rides the batch-flatten evaluator). `ladjIR` is FORWARD LADJ; `logDetJ = −ladjIR(f⁻¹(y))`. v1 symbolic-only (no numerical fallback). |
| `density.ts` (~1500) + `density-prims.ts` (~1100) | Single density implementation. `logDensityConsumeN(ir, value, refArrays, count, opts)` foundation + 3 wrappers; `OP_HANDLERS` consume/rest table (engine-concepts §4) over 12 measure ops + 8 multivariate kernels (via `walkMultivariate` → `density-prims.builtinLogdensityof`). `density-prims` is the FlatPDL `builtin_logdensityof` ABI + the four transports; MvNormal density routes through the `affine` registry entry. **`mcmarginal` / `walkMcMarginal`** (engine-concepts §6) is the marginalising-pushforward rule — sibling of `walkPushfwd` for a per-event pushforward bijective in one RETAINED innovation that MARGINALISES a latent; runs SYNC in-worker (`sampler.sampleLeafN` draws M latents, batched `evaluateExprN` for the inverse/LADJ grid, logsumexp); the node is self-contained (the worker supplies retained/marginal/out internally + threads `mcRng`/`mcMarginalizationCount` from `logDensityN`). `iid(mcmarginal, n)` factorises per-event via `walkIid`. Fed by the `mc-recipe` transform (below). Also hosts the type-mode `staticConsume` / `staticDensityShapeCheck` (§17). |
| `mc-recipe.ts` (~250) | Pure IR→IR functor-law transform `buildMcMarginalForm(measureIR, bindings, expand)` → `iid(mcmarginal{inverse, ladj, retained, marginal}, n)` \| null. Re-expresses a generative-composite measure tree `broadcast(post, broadcast(transport, iid(Normal,n), [pars]))` into the canonical density form: peels deterministic broadcast layers (folds their maps INTO the per-event recipe), reaches the generative kernel layer (`kernel-broadcast-shape.detectGenerativeKernelBinding`), maps broadcast args→kernel params to find the iid-collection latent (marginalised) vs the internal innovation (retained), derives the exact inverse+LADJ via `bijection-registry.invertExpr`. Also exports `inlineGenerativeClosure` — the ONE generative-closure inliner, consumed by BOTH eval directions (mat-broadcast's sampling path delegates to it). No materialiser/sampler dependency. SUPERSEDES the name-based `mat-density.deriveMcLikelihoodRecipe`. |
| `value.ts` (~700) + `value-ops.ts` (~1900) | The shape-tagged `Value` (engine-concepts §2.1): `shape`/`data`/`dtype`/`im`/`t` (Klein-4)/`struct`/`outerRank`. `transpose`/`adjoint`/`conjugate` toggle tag bits (no storage). `struct` bitmask (§2.2). `requireMatrix(v, op)` gates matrix-input ops (refuses nested-vector, spec §03); `promoteNestedToMatrix` is the sanctioned tag-stripper; `densify(v)` the one structured→dense fallback (**correctness never depends on a fast-path existing**). `value-ops`: shape-aware add/sub/neg/mul (+ batched `mulN`/`addN`), structured Cholesky/det/inv, complex (planar). |
| `worker.ts` (~770) + `worker-entry.ts` (~90) | Stateless message handler over sampler/density; transport shim sniffs Web Worker vs Node. Messages: init/sample/density/evaluate/sampleN/evaluateN/logDensityN/profileN/dispose (transferable-aware). `sampleN` static path and per-i path both fold onto the single batched leaf math (`makeBulkSampler`): the per-i branch dispatches on `hasRandNFn` — the randNFn family (Normal/LogNormal/Uniform/Exp) takes `makeBulkSampler`'s `perI` mode (per-atom param columns, one batched draw, atom-major for repeat>1); non-randNFn dists + non-interval Uniform return `{unhandled:true}` and fall back to `makeParametricSampler` (per-draw factory built ONCE — naive rebuild is ~10× the sampling cost). *One endpoint, not one algorithm*; the variable-length truncate-rejection loops stay on the scalar samplers. |
| `sampler.ts` (~2800) + splits | Single-point deterministic evaluator (`evaluateExpr`/`evaluateCall`), `ARITH_OPS`, worker entry points (`rand`/`makeSampler`/`makeParametricSampler`/`density`). **Hosts the measure walker** `walk` (was `traceeval.ts`; folded into sampler.ts — engine-concepts §11 — so the sampler↔traceeval require cycle is gone) — the per-draw value-position sampling primitive for `rand(state, M)` / `builtin_sample`. Leaf base case → the **single batched leaf endpoint** `sampleLeafN` (scalar = batch-of-1, so `rand(state, <leaf>)` ≡ `rand(state, iid(<leaf>, 1))[0]`); composite (joint / record / iid-of-composite / weighted / lawof) recurses via `MEASURE_OP_WALKERS`. Sample-side only — scoring is `density.ts`. Also dispatches the value-domain `rand_succ(state)` op (the COMPOSITE-`rand` successor — split lane 1 via the shared `rng.randSplitLanes`; counter-reset fresh state; engine-concepts §11), inline like `rand`/`rngstate` (not ARITH_OPS). Split: `sampler-registry` (REGISTRY + custom Ctors + PRNG bridge), `sampler-complex` (ESM leaf), `sampler-linalg` (ESM leaf — nested-array + Value-native LU/Cholesky/etc.), `sampler-aggregate` (CJS — `AGGREGATE_PATTERNS` + broadcast-reduce default), `sampler-eval-batched` (CJS — `evaluateExprN` + `ARITH_OPS_N` + complex-batched). ESM=pure-leaf, CJS=needs require()-back-to-sampler cycle (lazy, non-memoised). |
| `ops.ts` (~1100) + `ops-declarations.ts` (~2200) | Unified op-declaration registry + atom-batched dispatcher (engine-concepts §18). 20 ops across fixed-rank / rank-polymorphic / variadic / higher-order kinds; 12 batched fast-paths (incl. cross/inv/lower_cholesky/det/logabsdet/diagmat/row_gram/col_gram/linsolve). **Shape-pattern variants** (`OpVariant` with `argPatterns` + `wrappingOp` + specificity scoring): ~63 broadcasted primitives (the full scalar-primitive closure incl. comparisons / logic / link fns / complex accessors), 11 `mul` direct variants, and 3 atom-aware mul variants (rank-2×rank-1 gemv; rank-2×rank-2 batched matmul, real + complex) register as variants; `dispatch`/`dispatchVariant`/`dispatchTyped`(ArgInfo value+measure). |
| `dissolver.ts` (~2250) | Broadcast/aggregate dissolution term-rewriter (engine-concepts §20; JS specifics below). `dissolveBindings` runs in `buildDerivations` after lift; rewrites `.ir` in place. Phases 1–6 + shape-fold + fusion (a)/(b) MVPs + `propagateAxisStack` (authoritative for the iid / kernel_broadcast / aggregate axes it records; consumed by `axis-stack.ts` → matIid repeat axis + mat-broadcast K). |
| `aggregate-patterns.ts` (~220) + `aggregate-shape.ts` (~250) | P5 shared structural classifier: `classifyMatmulBody` (kinds `matmul` / `matvec` / `outer` / `dot`) + the leaf axis-get recognisers — ONE owner of the aggregate body pairing, consumed by BOTH `dissolver._tryDissolveAggregate` (IR rewrite) and `sampler-aggregate.AGGREGATE_PATTERNS` (runtime specialisers: matmul-family, matvec, outer-product, dot-product, batched-matmul, pure-axis-reduction; first-match-wins, gated by the `aggregate` toggle, equivalence-tested against the broadcast-reduce default). `aggregate-shape` infers aggregate output shapes for typeinfer. |
| `composite-body-recognizers.ts` (~400) + `kernel-broadcast-handlers.ts` (~240) | Kernel-broadcast body-shape recognition (engine-concepts §21): ordered recognizers → tagged-union `CompositeBody` (`iid` / `nested_broadcast` / `jointchain` / `joint` / `generative`); handler table maps kinds to the mat-broadcast executors; near-miss diagnostic names the closest shape. |
| `mat-broadcast.ts` (~2200), `mat-density.ts` (~490), `mat-multivariate.ts` (~570), `mat-transformations.ts` (~460) | Materialiser splits: kernel-broadcast composite executors + batch-flatten folds (mat-broadcast); density-routed kinds matScore / matBayesupdate / matLogdensityof (mat-density); multivariate handlers (mat-multivariate); pushfwd / locscale / relabel (mat-transformations). |
| `perf-config.ts` (~150) | **Optimization toggles & dual-mode testing** (engine-concepts §15): registry of default-true toggles (`aggregate`, `disintegrate.delegate`); `getOptimization(key)` gates each specialised dispatch against its general/reference path; `test/_perf-helpers.ts` `inBothModes(name, key, fn)` runs equivalence tests under both settings. Adding a specialised path = one toggle + a dual-mode test. |
| `sampler-eval-compile.ts` (~180) | Fused-loop codegen fast path for `evaluateExprN` (pure accelerator; bit-identical to the `_evalN` interpreter, which stays the source of truth; `FLATPPL_NO_EVALN_COMPILE=1` kill switch). |
| `ext-linalg.ts` (~690) | `ext-linear-algebra` std-module content (lu / svd / eigen / qr / kron / matexp / lstsq / rank …), registered via `standard-modules.ts`. |
| `ir-walk.ts` (~430) + `axis-stack.ts` (~170) | One declarative IR child-position enumeration (`forEachIRChild`, incl. `.bijection` descent) + scoped walkers (`walkIRScoped` / `mapIRScoped`, identifier-bound-param shadowing); axis-stack readers for the dissolver's authoritative outer-axis annotations. |
| `empirical.ts` (~700) | Weighted empirical measure utilities (log-space, ESS, resample). Null-uniform-weight protocol (null = uniform, no alloc); `logSumExp` max-subtraction; `totalLogMass` = 0 for null. |
| `histogram.ts` (~350) | Freedman-Diaconis / integer-atom binning, weighted histogram + quantile + plot range. |
| `rng.ts` (~740) | Pure Philox-4x32-10 counter-PRNG (`(key,counter)→4-tuple`, no sequential state — reproducible, parallel, cross-impl parity). 32×32→64 via 16-bit decomposition (no BigInt on the hot path). |
| `dataload.ts` (~220) | `load_data` parsers: JSON / CSV / WSV (pure, no I/O; host fetches bytes). Arrow IPC deferred. |
| `standard-modules.ts` (~430) | Std-module registry (spec §09): `_registerBuiltinStandardModules` auto-registers polynomials / particle-physics / ext-linear-algebra at load; typeinfer (`resolveStandardModuleRef`) + sampler (`_evaluateStandardModuleCall`, incl. higher-order std-module callables) dispatch are wired. Exact (name, compat) match (semver / loose-compat pending). |
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

**Materialiser authority — one set of measure-op handlers (Smell A).** The
measure-algebra ops have ONE realisation: the by-name `KIND_HANDLERS`
(matTruncate / matWeighted / matNormalize / matSelect / matPushfwd / matIid /
…). `materialiseMeasureIR` — the IR-direct walker the viewer's
`materialiseConcreteMeasure` and `matClm` (CLM bodies) consume for an inline /
reified measure IR — does **not** re-implement them. A thin per-op
IR→derivation bridge (`_bridgeDerivation` + `_bridgeToHandler`) installs an
inline sub-measure as a first-class synthetic binding (resolvable by
`getMeasure` / `expandMeasure`) and DELEGATES to the canonical handler ("by
reuse not reimplementation", engine-concepts §7 — the same
anti-parallel-interpreter discipline as the evaluator above): truncate,
weighted/logweighted, normalize, select (constant-weight selector synthesis
folded into matSelect — the former IR-direct `materialiseSelectIR` is **gone**),
pushfwd (§22 affine fast-path; lowered MvNormal), iid-of-composite (matIid's
repeat axis, §22.4). The cases that stay IR-native are the ones the by-name
handlers can't express: the **dependent-threaded** joint/record walk (matRecord
is independent-product only) and the §21 generative / scalar pushforward
(`evaluateN` of a value-expr; no by-name analogue), plus the iid-of-LEAF
`sampleN repeat=k` fast path. A full collapse of `materialiseMeasure(name)` into
`materialiseMeasureIR(expandMeasure(name))` is **deferred** — the duplication is
already removed by the delegations, so the inversion would add an
`expandMeasure` round-trip per by-name materialisation without removing any
parallel code (and the density/scoring kinds — bayesupdate / logdensityof /
likelihood_density / totalmass — have no `materialiseMeasureIR` coverage).

## Cross-file invariants

These lists must agree across files; drift produces silent runtime failures.
The cleanly-checkable ones ARE enforced by `test/invariants.test.ts`
(distribution ↔ REGISTRY both ways, discrete flag, EVALUABLE_OPS ↔ ARITH_OPS,
BIN_OP_MAP/UN_OP_MAP signatures, REGISTRY params ↔ types kwargs, sampleable/
discrete ⊆ `builtins.DISTRIBUTIONS`) — a failing test names the mirror to
update. The `MEASURE_OP_*` catalogs (`MEASURE_OP_CLASSIFIERS`, the measure walker's
`MEASURE_OP_WALKERS` in `sampler.ts`, density `OP_HANDLERS`, materialiser `KIND_HANDLERS`) are
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
| `sampler.ts` | walker + `MEASURE_OP_WALKERS` (sampling — the in-module measure walker) |
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
vs Placeholder boundary kwarg — feeds the §11 input-entry refs + viewer plot
ranges); `numType` on lit nodes (`1` integer vs `1.0` real);
`originLoc`/`originName` on LoweredBindings (diagnostics/DAG back-refs).

One deliberate divergence from the *syntactic image* of a source (still
valid spec FlatPIR):

- **`kernelof` never exists internally.** Both ingest points apply the
  §04 lowering rule `kernelof(x, kw) ≡ functionof(lawof(x), kw)` —
  `lower._lowerReification` for surface FlatPPL, `pir-sexpr`'s
  `readReificationTail` for `.flatpir` (rust-emitted FlatPIR contains
  `kernelof`). Consequence: JS-exported FlatPIR for a kernelof-authored
  source is the `functionof∘lawof` image. Preserving `kernelof` would
  touch every `op === 'functionof'` consumer for no engine benefit; a
  syntax-preserving round-tripping tool is flatppl-rust's role.

**Reified-body namespaces are spec-shaped** (spec §11): `%local` is the
placeholder namespace ONLY (`_x_`-class formals — lambda / fn-hole params
and identifier boundary kwargs that name no module node, which lower
treats as formals per the §04 lambda rule); body refs to identifier-form
boundary inputs that designate REAL nodes are plain `self`/module refs.
The cut lives in the entry list (`params`/`paramKwargs`/`paramSources`),
and the §04 input substitution is applied by boundary-aware consumers:
`ir-walk`'s `identifierBoundParams` + `walkIRScoped`/`mapIRScoped` give
collectors and rewriters nesting-aware shadowing (a body ref naming an
identifier-bound param is the callable's INPUT, not an outer-scope dep);
application/substitution sites accept the self-form param ref alongside
`%local` (the param list disambiguates from captures). The boundary-feed
nets back this end-to-end (audit §3 — a feed gap is loud, never a silent
re-materialisation of the like-named module binding): feeders declare
`ctx._boundaryNames`, and `collectRefArrays`/`prepareDensityRefs` THROW if
a declared boundary ref reaches getMeasure uncovered (Phase 4); on the
lowering itself, `clm.lowerMeasure` THROWS when a body self-ref is covered
by no declared input (the ⊆ invariant), and `clm.assertFedCoverage` —
called by `matScore` after its overlay merge, main-thread because the
worker cannot distinguish a shared ref from an unfed boundary — THROWS
when a declared boundary the body references has no fed column (Phase 6).
`pir-sexpr` consequently carries bodies VERBATIM (no serialization-
boundary namespace translation).

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

## Fixed-phase values: `fixedValues` (demand-driven, `fixed-values.ts`)

Per spec §04, fixed-phase bindings are compile-time-determinate. Their value
resolution is **demand-driven** (engine-concepts §17.1): `buildDerivations`
returns `fixedValues` as a **`FixedValues`** instance (`fixed-values.ts`) — a
lazy, memoised, cycle-guarded resolver, NOT an eagerly-populated map. A binding's
value is computed only when first asked for, then cached. This matters whenever a
fixed value isn't a per-atom `Float64Array` slice: compile-time-known-length
numeric arrays (`rand(rstate, iid(Normal,10))` → length 10, not `N`),
records/tuples, opaque values (rngstate). The per-binding evaluation logic (walk
self-refs, recursing measure subtrees via `expandMeasure`; skip measure-typed
refs; resolve value-context refs via `this._resolve`; `evaluateExpr`,
catch→UNRESOLVED) is `FixedValues._compute` — moved verbatim from the former
eager `while (progress)` sweep.

**Map-compatible** (`.has`/`.get`/`.set` + iteration) so the ~100 consumers that
read it as a `Map` are untouched. `.has(name)`/`.get(name)` resolve on demand
(compute + cache). **iterate-forces-all**: `size`/`forEach`/`keys`/`values`/
`entries`/`[Symbol.iterator]` force-resolve every fixed-phase binding then
delegate to the cache (iteration means "all fixed values" — can't be lazy; keeps
test harnesses that mirror the old bulk push green). **No negative cache**: an
UNRESOLVED outcome (cycle / evaluator throw / unresolvable dep) is not remembered
— recomputed on next demand (preserves the old fixpoint's retry).

**Who demands (and the one bounded eager site).** `resolveConstant` pulls an iid
count during pass-2 classification (only that subgraph). **Cascade-prune**
(`derivationRefsValid.resolvable`) answers by PHASE for derivation-having
bindings (no resolve — the laziness win for a never-displayed `B = expensive(A)`)
and only force-resolves an *underived* fixed binding to confirm it has a value
(faithful prune of a dead-end's dependents). The **fixed-phase dead-end
diagnostic** force-resolves underived fixed-phase non-object bindings — the one
intentional bounded eager site (a bare such binding is still resolved at build
time, same as before; full zero-eval needs a reachability gate — TODO §06).
Materialisation demands a fixed value via `collectRefArrays`/`prepareDensityRefs`
→ `pushFixedEnv` (per-materialisation, `setEnv merge:true`); the fixed-phase
short-circuit (`fixedPhaseStage`) synthesises the measure for a fixed-phase
target plotted directly.

**Viewer** (no longer bulk-pushes the map): per `rebuildDerivations` it sends one
`setEnv {merge:false}` carrying only `__moduleRegistry` — the stale-env RESET
(always fires). Each materialisation pushes its own fixed refs on demand;
`render-profile` pushes the profile body's fixed self-refs (`setEnv merge:true`)
before `profileN`. is-fixed-phase SIGNALS read `binding.phase === 'fixed'`
(`render-kernel`, `overrides`); `plot-plan` keeps `fixedValues.has` (value-
presence — distinguishes opaque rngstate). `collectRefArrays` drops fixed-phase
refs from refArrays (session env owns them); opaque values aren't plottable
(`buildPlotPlan` returns null for `rngstate`).

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

`packages/engine/test/` (~190 files): per-module unit tests; distributions
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
| "How is this measure-algebra op handled?" | `derivations.ts` classifier + `_expandByName`; `materialiser.KIND_HANDLERS`; `density.OP_HANDLERS`; `sampler.ts` `MEASURE_OP_WALKERS` (the measure walker) |
| "Why does the DAG render this way?" | `dag.ts` |
| "Why did disintegrate produce this Plan?" | `disintegrate.ts` (`disintegratePlan`) |
| "Why is the RNG result this number?" | `rng.ts` (Philox); `sampler.ts` PRNG adapter |
| "How are refs in a measure body resolved?" | `materialiser.ts` `prepareDensityRefs`; `sampler.ts` `evaluateExprN` |
| "Empirical measure / Value shape?" | `engine-types.d.ts`; engine-concepts §2 / §2.1–2.2 |

## Known issues and gaps

Full list + priorities in `flatppl-dev/TODO-flatppl-js.md` (the checkable source
of truth — no status duplication here). Shortlist: multi-file `load_module`
end-to-end (bundle API landed; `.flatppl` cross-module resolution + substitution +
host pre-fetch pending); standard modules (registry + typeinfer/sampler dispatch
wired; the named-module contents + semver matching pending); `PoissonProcess` (needs a
ragged-vector-per-atom measure shape class); multivariate transports beyond
MvNormal (Dirichlet stick-breaking etc.); some newer stdlib fns still type as
`deferred()`; `packages/viewer/src/main.ts` (~69 KB) is the next decomposition
target (the monolithic `viewer.js` is now ~24 split modules).

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
remaining only for symbolic ('%dynamic' / ref-name) sizes. It **descends into
kernel-broadcast bodies** (`_kernelBodyInnerStack`: kernel head ref → `functionof`
body → unwrap `lawof` → recurse), so a kernel-broadcast binding carries the full
parallel-axis ladder — `[kernel_broadcast K_outer, broadcast K_inner]` (nested)
or `[kernel_broadcast K, iid D]` (iid-composite) — the static K_inner the Phase 8
nested/joint/jointchain de-nesting folds consume (§22.4). Composites whose
variate shape isn't a single inner's (superpose / select / joint) carry no
stack by design.

**Runtime fast-path.** For broadcasts the dissolver leaves (or that the user
writes as `broadcasted(op)(args)`), `_maybeFastBroadcasted` recognises the
`broadcast(<scalar op>, args)` shape, coerces args to Values (incl. flat
all-scalar JS arrays), gates on the spec-§04 collection rule — same rank
among rank ≥ 1 inputs, per-axis sizes equal-or-1 — and dispatches through
value-ops' `*Elem` primitives (flat row-major or stride-0-strided for
singleton axes; no per-cell `evaluateExpr` reentry). The equal-or-1 rule has
ONE owner, `value-ops._broadcastOutShape`, shared by the gate and every
elementwise impl (binop factory, complex binop, ifelseElem, complexElem).
The cold path keeps what it alone can express: nested-vector (Ref-wrap)
args, outerRank-tagged vec-of-vec Values (per-cell sees the inner vector
WHOLE — the gate bails on `outerRank < rank` so §04 outer-axis semantics
can't be silently elementwise-dispatched), tables, multi-op bodies, and all
incompatible-shape diagnostics. Spec-`mul` matrix semantics are preserved —
the fast path intercepts only `broadcast(...)` IRs, not direct `mul(A,B)`.
**Constants stay rank-0** (`fixedValueToMeasure` carries the rank-0 Value at
`m.value`; the length-N `.samples` buffer is transitional for legacy consumers);
arithmetic broadcasts rank-0 × rank-N via stride-0 reads. The Klein-4 + struct
tags stay runtime-internal markers behind the IR contract (free transpose via
tag flip; structured-storage exploitation; op-boundary handoffs materialise to
contiguous form). Remaining phases + the cold-path audit are in the TODO.
