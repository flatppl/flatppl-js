# AGENTS.md — orientation for AI coding agents

You're working in **`flatppl-js`**, the JavaScript reference implementation of
**FlatPPL** (the Flat Portable Probabilistic Language). This file is your
entry-point map. Keep it tight; depth lives in linked documents.

## Read the language spec first

Almost every non-trivial task in this repo requires the FlatPPL language
specification. **Read it before making semantic changes.** The spec lives in the
sibling repo **flatppl-design**; resolve in this order:

1. **A directory accessible in your current session** — your AI tool's
   mechanism for session-wide directory access (workspace folders,
   additional-directory grants, or equivalent). If `flatppl-design` is mounted,
   use it from there. The user often edits the spec alongside the engine, so a
   local copy may be ahead of the published version.
2. **Filesystem sibling** — `../flatppl-design/docs/` from this repo, when both
   are checked out under one parent.
3. **GitHub** — https://github.com/flatppl/flatppl-design — fetch read-only as a
   last resort.

The spec docs are numbered: start with `02-overview.md`, then dip into
`03-value-types.md`, `04-design.md`, `05-syntax.md`, `06-measure-algebra.md`,
`07-functions.md`, `08-distributions.md`, `11-flatpir.md`. For something specific,
the file numbers tell you where to go.

## Repo layout

This is an npm workspace monorepo with four packages:

```
flatppl-js/
├── AGENTS.md                            ← you are here
├── packages/
│   ├── engine/                          ← parser, analyzer, IR, orchestrator, sampler
│   │   ├── ARCHITECTURE.md              ← READ THIS before non-trivial engine work
│   │   └── *.ts                         ← ~65 top-level modules, ~60 KLOC
│   ├── viewer/                          ← browser-side DAG + plot rendering
│   │   └── src/                         ← ~24 TS modules, ~9 KLOC
│   ├── vscode-extension/                ← thin VS Code wrapper
│   │   ├── extension.ts                 ← extension host
│   │   ├── src/visualPanel.ts           ← webview panel singleton
│   │   ├── src/lspClient.ts             ← LSP language-client lifecycle (spawns the
│   │   │                                  bundled `flatppl-lsp` binary over stdio);
│   │   │                                  esbuild-bundled (inlines vscode-languageclient)
│   │   │                                  into src/lspClient.js
│   │   └── src/lspHelpers.ts            ← pure helpers: host→triple resolution, binary
│   │                                      resolution, catalogue reading, serial-queue /
│   │                                      debounce utilities; type-stripped to
│   │                                      src/lspHelpers.js; unit-tested directly
│   └── web/                             ← standalone gallery shell for static hosts
│       ├── src/                         ← page entry, app, resolver, router,
│       │                                  manifest loader, syntax highlighter
│       └── demo/                        ← in-package demo .flatppl programs
└── package.json                         ← workspace root
```

**Engine is the bulk of the work.** Sources are TypeScript throughout; `.js`
siblings of the extension sources are git-ignored build artifacts. Viewer is
~24 ES modules (`main.ts` is the orchestrating entry, ~1.6 KLOC) bundled to a
single IIFE at build time; vscode-extension is a thin wrapper around the
engine + viewer; web is a gallery shell built on top of viewer for non-VS-Code
hosts (GitHub Pages, docs sites, web app deploys). The viewer and web hosts
are independent surfaces —
viewer's `embed-test.html` is for embedding only the DAG/plot panes inside a
larger page; `packages/web` adds the file tree, source pane, hash routing, and
manifest-driven model discovery on top.

## Engine pipeline (one-paragraph version)

`tokenizer → parser → analyzer → pir.lowerToModule → typeinfer → orchestrator
→ worker (sampler / density / empirical / histogram / rng)`. The engine's
`processSource(src)` runs tokenize → parse → analyze (lowering and type
inference run *inside* `analyze`) and returns
`{ ast, bindings, loweredModule, symbols, diagnostics, variant, bundle }`;
the orchestrator runs on demand, not in `processSource`. The worker runs in a
separate thread (Web Worker in browser, `worker_threads` in Node) and is
addressed via postMessage; `engine/index.ts` deliberately does **not** re-export
`sampler`/`worker` because they pull in ~1 MB of stdlib distribution code.

Full diagram, per-file responsibilities, IR shapes, and "where to look for X"
table: see [`packages/engine/ARCHITECTURE.md`](packages/engine/ARCHITECTURE.md).

## Critical conventions

These are the things that catch out first-time contributors. Read each one.

- **Read [`flatppl-dev/CONVENTIONS.md`](../flatppl-dev/CONVENTIONS.md)** —
  ecosystem-wide invariants and conventions every contributor should know
  (version state, examples-and-test-fixtures pattern, no-AI-model-identification
  policy). Resolves the same way as `flatppl-design`: workspace folder /
  filesystem sibling / GitHub. Apply unconditionally; the items below are
  flatppl-js-specific additions.

- **FlatPPL is a different language from JavaScript.** It uses JS/Python-compatible
  syntax for cheap parsing, but the semantics are entirely separate. Don't import
  JS reasoning for FlatPPL constructs (no mutation, no loops, no implicit
  elementwise ops, no truthiness coercion).

- **FlatPPL is 1-indexed; this engine stores 0-indexed.** The engine uses
  0-based JS internally for performance; translate at FlatPPL
  `get`/indexing/decomposition boundaries and at display labels (`obs[1]`,
  not `obs[0]`). Other engines (Julia, …) may store 1-indexed natively;
  the spec is 1-indexed regardless.

- **Idiomatic JavaScript (TypeScript-typed), prototype/closure-based.** The
  codebase is intentionally not class-OO. Type representations are
  `{kind, ...}` plain objects with a
  discriminator. State lives in closures or `Map`s, not class instances. Two
  classes exist (`FlatPPLPanel`, the VS Code panel singleton in
  `vscode-extension/src/visualPanel.ts`; `FixedValues`, the demand-driven
  fixed-value resolver in `engine/fixed-values.ts`); both are well-justified.
  Don't introduce more without a reason.

- **Generous code comments.** This codebase deliberately overrides the "minimal
  comments" default. Every non-trivial function has a JSDoc block citing the
  spec section it implements; module headers explain what the file does and does
  not do. When you add code, write comments that explain *why*, not *what* —
  future AI agents (and humans) will thank you.

- **Cross-file invariants: several catalogs must agree across files** (built-in
  name lists, sampleable distribution lists, evaluable op lists); drift produces
  silent runtime failures. The cleanly-checkable ones ARE enforced by
  `packages/engine/test/invariants.test.ts` — a failing test names the mirror
  catalog to update. The `MEASURE_OP_*` catalogs are intentionally hand-watched
  (no clean equivalence holds). See "Cross-file invariants" in
  `packages/engine/ARCHITECTURE.md` before adding distributions, built-in
  functions, or measure-algebra ops.

- **Vectors of vectors are NOT matrices (spec §03).** A bare nested
  literal `[[1, 2], [3, 4]]` is a vector-of-vectors, semantically
  distinct from a 2x2 matrix; only `rowstack(...)` (rows = inner
  vectors) or `colstack(...)` (columns = inner vectors) commits a
  layout interpretation. FlatPPL is row/column-major-agnostic — the
  engine uses flat row-major Float64Array storage as its INTERNAL
  convention (the ArrayOfSimilarArrays-style "vec-of-equal-sized-
  vecs backed by flat n-d storage", cf. Julia ArraysOfArrays.jl), but
  the SEMANTIC distinction is enforced end-to-end:
    - `value.ts` Values carry an optional `outerRank` tag; user
      nested literals get it set via `asValue`; matrix-input ops
      refuse it via `valueLib.requireMatrix(v, opName)`.
    - Typeinfer signature unification rejects vec-of-vec input to
      matrix-input ops; `argError` appends a `rowstack(...)` hint.
    - Aggregate body indexing `A[.i, .j]` stays SPEC-LEGAL on
      vec-of-vec (spec §04 sec:aggregate: `A[.i, .j] ≡ A[.i][.j]`);
      the runtime row-major convention makes the numerical result
      equivalent.
    - When you write a fixture or test that uses a matrix, wrap it
      with `rowstack(...)` explicitly. New fixtures that bypass this
      will fail at typeinfer with the §03 diagnostic.
    - See `test/spec-vec-of-vec-not-matrix.test.ts` for the 29
      pinned invariants.

- **Fixed-phase bindings flow through `fixedValues`, not refArrays.**
  `buildDerivations(...).fixedValues` is a demand-driven, memoised,
  cycle-guarded `FixedValues` resolver (`engine/fixed-values.ts`,
  Map-compatible) — a binding's value is computed on first demand, not
  eagerly. Materialisation pushes the fixed refs each evaluation needs to
  the worker session env via `pushFixedEnv` (per-materialisation `setEnv`
  merge, `materialiser-shared.ts`); evaluators layer session env underneath
  refArrays. When you touch the chain / derivation / refArrays machinery,
  remember that fixed-phase refs are env-resolved, not slice-indexed — see
  ARCHITECTURE.md's "Fixed-phase values: `fixedValues` (demand-driven,
  `fixed-values.ts`)" section.

- **Three bundle deployments.** The engine+viewer bundles are deployed in
  THREE places: `packages/viewer/vendor/` (standalone embed),
  `packages/vscode-extension/lib/` (extension webview), and
  `packages/web/dist/vendor/` (web gallery). Engine and viewer changes that
  affect runtime behaviour require rebuilding **all** of them — `npm run build`
  from the workspace root does that (workspace builds + `build:vendor`).
  `npm run build:vendor` alone only refreshes the extension's `lib/` (it is
  an alias for `npm run --workspace=packages/vscode-extension build:vendor`).
  Don't assume the user is testing one host — rebuild all.

  **LSP provisioning (`build-vendor.mjs`).** As part of `build:vendor`,
  `packages/vscode-extension/build-vendor.mjs` puts a `flatppl-lsp` binary into
  `packages/vscode-extension/bin/`, choosing a source by precedence (decision
  logic in the pure module `build-lsp-source.cjs`): (1) `FLATPPL_LSP_BIN_DIR` —
  copy every `flatppl-lsp-*` found there (CI stages the five per-target binaries
  this way); (2) `FLATPPL_LSP_LOCAL=/path/to/flatppl-lsp` — copy that prebuilt
  host binary; (3) a sibling `flatppl-rust` checkout (`FLATPPL_RUST_DIR`, else
  `../../../flatppl-rust`) — `cargo build --release -p flatppl-lsp`; (4) no
  sibling but `cargo` on PATH — shallow-clone `flatppl-rust@main` to a temp dir,
  build, copy the host binary, delete the clone; (5) no sibling, no `cargo` —
  download the host binary from the `flatppl-rust` nightly release. Routes 2–5
  produce the host triple only; CI (route 1) bundles all five into one universal
  vsix. The binary is copied *into* `bin/`, so the vsix is self-contained — the
  sibling/clone may be deleted afterward. A real cargo failure on routes 3/4
  errors loudly; it never silently falls back to a download.

- **Webview escape traps in `vscode-extension/src/visualPanel.ts`.** The
  webview HTML lives inside the outer template literal returned by
  `_getHtml()`. The bulk of inline JS is now gone (viewer loaded as a
  separate script), but `${nonce}` / `${...Uri}` interpolations and any
  small inline content remain at risk. Two specific failure modes:

  1. **Backticks anywhere inside the template literal.** A JSDoc-style
     reference like `` `prior2` `` closes the outer template literal
     mid-document. Use apostrophes or plain text in any prose inside
     `_getHtml`'s template.
  2. **Backslash escapes in inner string literals** (if you ever add
     inline JS back). `essLabel.title = 'foo\nbar';` looks like a
     string with `\n`, but the outer template literal interprets `\n`
     as a real newline first, leaving an unterminated string in the
     rendered HTML. Write `\\n` in source for a literal `\n` in the
     rendered JS, or avoid multi-line strings entirely.

  **Detection:** `npm run --workspace=packages/vscode-extension typecheck`
  (tsc) catches case (1) on the `.ts` source, as does
  `node --check src/visualPanel.js` on the transpiled artifact after a
  `build:vendor` (`node --check` does not accept `.ts` directly). Neither
  catches case (2) — that breakage manifests only in the rendered HTML the
  webview iframe parses.

  Failure signature is webviews that go silently empty with
  `Uncaught SyntaxError: Invalid or unexpected token` in the iframe,
  with line numbers pointing at innocuous-looking comment lines. Easy
  to lose hours debugging — beware.

- **Test fixtures are at `packages/engine/test/fixtures/`.** Per the
  examples-and-test-fixtures convention in `flatppl-dev/CONVENTIONS.md`,
  fixtures here are *copies* of the upstream `.flatppl` files in the
  sibling `flatppl-examples` repo. When upstream changes, refresh the
  copy and mention the upstream commit in the engine commit message.

- **One commit per significant step.** The user prefers focused per-topic commits
  as steps finish (matching the existing git history). Don't accumulate multiple
  themes into one big commit. Don't commit unless the user asks.

## Build and test

```sh
# from repo root
npm install                          # first time only

npm test                             # run all workspace test suites
                                     # (~190 test files, all under
                                     #  packages/engine/test/)

# rebuild ALL deployed bundles after engine or viewer changes:
npm run build                        # viewer vendor/ + web dist/ + extension lib/

# narrower rebuilds when only one host matters:
npm run --workspace=packages/viewer build   # packages/viewer/vendor/
npm run --workspace=packages/web build      # packages/web/dist/
npm run build:vendor                        # extension lib/ ONLY
npm run watch:vendor                        # extension lib/ watch mode

# run a single test file:
node --test packages/engine/test/orchestrator.test.ts
```

CI on this repo is GitHub Actions; see `.github/workflows/`.

The user runs the VS Code extension via the standard "Launch Extension" debug
profile or by installing the built `.vsix`. The viewer can also be served
standalone (`npm run --workspace=packages/viewer serve` → http://localhost:8000/),
and the web gallery host has its own dev server
(`npm run --workspace=packages/web build && npm run --workspace=packages/web serve`
→ http://localhost:8001/). The two web servers target different concerns —
viewer is the DAG/plot viewer in isolation; `packages/web` is the full
three-pane gallery shell with file tree, source view, syntax highlighting,
and source ↔ DAG cross-pane navigation. CI deploys
`packages/web/dist/` to GitHub Pages on push to main (see
`.github/workflows/pages.yml` — needs the one-time repo setting Pages →
Source: GitHub Actions before its first run takes effect).

## Known issues

A point-in-time architectural review (May 2026) flagged a number of bugs and
gaps. Most of the small consistency bugs have since been fixed; the remaining
items below are larger structural work or open feature gaps.

- **Type signatures cover ~185 ops** (`types.ts` `SIGNATURE_FACTORIES`), plus
  typeinfer special-case handlers — the array-construction shape rules
  (`fill`/`zeros`/`ones`/`eye`/`cartpow`/`cartprod`/`stdsimplex`) have landed.
  Still falling through to `deferred()` silently: the multivariate
  distributions (`MvNormal`, `Wishart`, `Dirichlet`, `Multinomial`, …) and a
  handful of measure-algebra / structural ops (`kernelof`, `disintegrate`,
  `restrict`, `relabel`, …). If you add a new distribution or built-in, also
  add its signature in `types.ts`.
- **`orchestrator.ts` was split** into five facade modules (`ir-shared`,
  `lift`, `derivations`, `signatures`, `profile-plan`); the core is now a
  thin facade (~550 lines). See the "Module map" in `ARCHITECTURE.md` for
  what lives where and the one-way dependency order. **The viewer
  decomposition landed** — `viewer/src/` is ~24 modules; the largest
  remaining file is `main.ts` (~1 600 lines), the next decomposition target.
- **The planning document `flatppl-dev/TODO-flatppl-js.md`** (in the
  sibling `flatppl-dev` repo, resolved the same way as `flatppl-design`)
  tracks the remaining work toward complete spec coverage. `flatppl-dev`
  is the cross-repo development infrastructure for the FlatPPL ecosystem
  — TODO files today, and any future shared AI context, working
  agreements, or engineering meta-docs as the ecosystem grows. **Check
  it before starting feature work** — the TODO file lists what's open,
  what's in progress, and what's blocked. Workflow conventions live in
  `flatppl-dev/AGENTS.md`; in short:
    - Pull `flatppl-dev` at session start.
    - Mark items `[in-progress: <handle>]` when you start them so others
      don't pick them up too.
    - Remove or check off items when you commit the engine change that
      closes them; commit the TODO update either in the same session or
      as a separate commit in `flatppl-dev`.

If you spot a new bug or gap during a task, surface it in the commit message
or, when it's larger than a one-liner, add it to `TODO-flatppl-js.md`.

**The orientation docs are living documents.** When you make a non-trivial
change — new feature, new invariant, fixing one of the known issues, a
structural refactor — update the affected sections of `AGENTS.md` and/or
`packages/engine/ARCHITECTURE.md` in the same or a follow-up commit. Stale
orientation docs teach future cold-session AI agents to expect things that
aren't there.

## Style and process

- **Don't fix things you weren't asked to fix.** Drive-by fixes break focused
  commits and complicate review. If you spot a related bug, mention it; don't
  silently include it.
- **Don't run destructive git commands without asking.** No force-push, no
  `--no-verify`, no `git reset --hard` unless the user explicitly approves.
- **Don't commit unless asked.** When asked, follow the commit conventions in
  this repo's history (one topic per commit; descriptive subject; reference
  the spec section if relevant).
- **Trust but verify the spec.** The spec is the source of truth for
  semantics; the engine is one implementation among several planned across
  language ecosystems (Rust, Julia, Python, …). When you find an
  engine/spec mismatch, the engine is usually the side that's wrong.
- **Per-file headers are the intended primary documentation.** Read them. They
  explain *why* each file exists, what it does NOT do, and how it relates to
  neighbours. After reading the spec, the per-file headers are the next layer
  of context.
