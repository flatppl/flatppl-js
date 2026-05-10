# AGENTS.md — orientation for AI coding agents

You're working in **`flatppl-js`**, the JavaScript reference implementation of
**FlatPPL** (the Flat Portable Probabilistic Language). This file is your
entry-point map. Keep it tight; depth lives in linked documents.

## Read the language spec first

Almost every non-trivial task in this repo requires the FlatPPL language
specification. **Read it before making semantic changes.** The spec lives in the
sibling repo **flatppl-design**; resolve in this order:

1. **A directory accessible in your current session** — VS Code workspace folders,
   Claude Code "additional working directories", Cursor open folders, etc. If
   `flatppl-design` is mounted, use it from there. The user often edits the spec
   alongside the engine, so a local copy may be ahead of the published version.
2. **Filesystem sibling** — `../flatppl-design/docs/` from this repo, when both
   are checked out under one parent.
3. **GitHub** — https://github.com/flatppl/flatppl-design — fetch read-only as a
   last resort.

The spec docs are numbered: start with `02-overview.md`, then dip into
`03-value-types.md`, `04-design.md`, `05-syntax.md`, `06-measure-algebra.md`,
`07-functions.md`, `08-distributions.md`, `11-flatpir.md`. For something specific,
the file numbers tell you where to go.

## Repo layout

This is an npm workspace monorepo with three packages:

```
flatppl-js/
├── AGENTS.md                            ← you are here
├── packages/
│   ├── engine/                          ← parser, analyzer, IR, orchestrator, sampler
│   │   ├── ARCHITECTURE.md              ← READ THIS before non-trivial engine work
│   │   └── *.js                         ← ~13 KLOC
│   ├── viewer/                          ← browser-side DAG + plot rendering
│   │   └── src/viewer.js                ← single-file IIFE, ~5.7 KLOC
│   └── vscode-extension/                ← thin VS Code wrapper
│       ├── extension.js                 ← extension host
│       └── src/visualPanel.js           ← webview panel singleton
└── package.json                         ← workspace root
```

**Engine is the bulk of the work.** Viewer is one large IIFE (no internal modules
yet); vscode-extension is a thin wrapper around the engine + viewer.

## Engine pipeline (one-paragraph version)

`tokenizer → parser → analyzer → pir.lowerToModule → typeinfer → orchestrator
→ worker (sampler / traceeval / empirical / histogram / rng)`. The engine's
`processSource(src)` runs the first six stages and returns
`{ ast, bindings, loweredModule, symbols, diagnostics }`. The worker runs in a
separate thread (Web Worker in browser, `worker_threads` in Node) and is
addressed via postMessage; `engine/index.js` deliberately does **not** re-export
`sampler`/`worker` because they pull in ~1 MB of stdlib distribution code.

Full diagram, per-file responsibilities, IR shapes, and "where to look for X"
table: see [`packages/engine/ARCHITECTURE.md`](packages/engine/ARCHITECTURE.md).

## Critical conventions

These are the things that catch out first-time contributors. Read each one.

- **FlatPPL is a different language from JavaScript.** It uses JS/Python-compatible
  syntax for cheap parsing, but the semantics are entirely separate. Don't import
  JS reasoning for FlatPPL constructs (no mutation, no loops, no implicit
  elementwise ops, no truthiness coercion).

- **FlatPPL is 1-indexed.** The engine uses 0-based JS internally; translate at
  FlatPPL `get`/indexing/decomposition boundaries and at display labels (`obs[1]`,
  not `obs[0]`).

- **FlatPPL is at v0.1, pre-release.** No backward-compat engineering is needed
  yet. The engine, the spec, and standard modules all stay at
  `flatppl_compat = "0.1"` through breaking changes until the first real release.

- **Idiomatic JavaScript, prototype/closure-based.** The codebase is intentionally
  not class-OO. Type representations are `{kind, ...}` plain objects with a
  discriminator. State lives in closures or `Map`s, not class instances. Two
  classes exist (`FlatPPLPanel` for the VS Code panel singleton, one supporting
  class in viewer.js); both are well-justified. Don't introduce more without a
  reason.

- **Generous code comments.** This codebase deliberately overrides the "minimal
  comments" default. Every non-trivial function has a JSDoc block citing the
  spec section it implements; module headers explain what the file does and does
  not do. When you add code, write comments that explain *why*, not *what* —
  future AI agents (and humans) will thank you.

- **Cross-file invariants exist that aren't enforced by tests.** Several catalogs
  must agree across multiple files (built-in name lists, sampleable distribution
  lists, evaluable op lists). Drifting them produces silent runtime failures.
  See "Cross-file invariants" in `packages/engine/ARCHITECTURE.md` before
  adding distributions, built-in functions, or measure-algebra ops.

- **Fixed-phase bindings flow through `fixedValues`, not refArrays.** The
  orchestrator pre-evaluates fixed-phase bindings and exposes the values via
  `buildDerivations(...).fixedValues`. The viewer pushes that map to the
  worker as session env on every rebuild; per-atom evaluators layer session
  env underneath refArrays. When you touch the chain / derivation / refArrays
  machinery, remember that fixed-phase refs are env-resolved, not slice-
  indexed — see ARCHITECTURE.md's "Fixed-phase pre-eval and fixedValues"
  section.

- **Two webview deployments.** The viewer's bundled output lives in BOTH
  `packages/viewer/vendor/` (standalone embed) AND
  `packages/vscode-extension/lib/` (extension webview). Engine and viewer
  changes that affect runtime behaviour require rebuilding **both** bundles
  (`npm run build:vendor` from the workspace root, plus
  `npm run --workspace=packages/vscode-extension build:vendor`). Don't assume
  the user is testing one or the other — rebuild both.

- **Webview escape traps in `vscode-extension/src/visualPanel.js`.** The webview
  HTML lives inside an outer template literal; backticks AND backslash escapes
  (`\n`, `\t`, …) get interpreted by the host parser before the inner JS sees
  them, breaking the rendered HTML. The viewer was moved out of the inline HTML
  into a separately-loaded script for this reason. If you must add inline JS
  back, beware.

- **Examples live in a separate repo.** The `.flatppl` example files belong in
  the **flatppl-examples** repo, not here. The engine's
  `test/fixtures/*.flatppl` are *copies* — update both when an example changes.

- **One commit per significant step.** The user prefers focused per-topic commits
  as steps finish (matching the existing git history). Don't accumulate multiple
  themes into one big commit. Don't commit unless the user asks.

## Build and test

```sh
# from repo root
npm install                          # first time only

npm test                             # run all workspace test suites
                                     # (671 tests, ~3.5 s)

# rebuild bundles after engine or viewer changes:
npm run build:vendor                 # both standalone-vendor and extension lib
npm run watch:vendor                 # watch mode

# run a single test file:
node --test packages/engine/test/orchestrator.test.js
```

CI on this repo is GitHub Actions; see `.github/workflows/`.

The user runs the VS Code extension via the standard "Launch Extension" debug
profile or by installing the built `.vsix`. The viewer can also be served
standalone (`npm run --workspace=packages/viewer serve` → http://localhost:8000/).

## Known issues

A point-in-time architectural review (May 2026) flagged a number of bugs and
gaps. Most of the small consistency bugs have since been fixed; the remaining
items below are larger structural work or open feature gaps.

- **Type system covers ~50 ops.** Many spec ops fall through to `deferred()`
  silently — most multivariate distributions, the array/table-generation
  suite, linear algebra, several measure-algebra ops. If you add a new
  distribution or built-in, also add its signature in `types.js`.
- **`orchestrator.js` and `viewer/src/viewer.js` are oversized** (3 445 and
  5 683 lines). Be aware before opening them; both have natural decomposition
  seams documented in `ARCHITECTURE.md`.
- **The planning document `flatppl-todo/TODO-flatppl-js.md`** (in the
  sibling `flatppl-todo` repo, resolved the same way as `flatppl-design`)
  tracks the remaining work toward complete spec coverage. **Check it
  before starting feature work** — it lists what's open, what's in progress,
  and what's blocked. The file is the lightweight coordination surface for
  multiple contributors (humans and AI agents) until we move to GitHub
  issues, so:
    - Pull `flatppl-todo` at session start.
    - Mark items `[in-progress: <handle>]` when you start them so others
      don't pick them up too.
    - Remove or check off items when you commit the engine change that
      closes them; commit the TODO update either in the same session or
      as a separate commit in `flatppl-todo`.

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
