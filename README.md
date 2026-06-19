# FlatPPL — JavaScript implementation

JavaScript ecosystem for FlatPPL, the Flat Portable Probabilistic Language.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Components

This monorepo contains:

* [The FlatPPL JavaScript reference engine](packages/engine)

* [The FlatPPL viewer](packages/viewer)

* [The FlatPPL Visual Studio Code extension](packages/vscode-extension)

* [The FlatPPL web gallery host](packages/web)

## Live demos

* [Live demo of the FlatPPL web gallery](https://flatppl.github.io/flatppl-js/)

## Building and testing

The monorepo is an npm workspace; root-level scripts fan out to every
package via `--workspaces --if-present`.

**Prerequisites**

- **Node.js + `npm`** — the workspace tooling.
- **A Rust toolchain (`cargo`) on `PATH`** — the build compiles `flatppl-lsp`
  (and the convert wasm) from source; there is no prebuilt-binary download
  fallback. Without `cargo`, `npm run build` stops with an actionable error.
- **The wasm toolchain — for the gallery's _Convert to FlatPPL_ (on by
  default):** `wasm-pack` and the `wasm32-unknown-unknown` target on `PATH`. The
  build uses them but never installs them — toolchain setup affects your whole
  Rust installation, so it is left to you. One-time setup:

  ```sh
  rustup target add wasm32-unknown-unknown
  cargo install wasm-pack
  ```

  Or build the gallery convert-less with `FLATPPL_CONVERT=off` (no wasm toolchain
  needed).

```sh
npm install                  # install all workspace deps
npm run build                # build every package + vendor bundle
npm test                     # run all test suites (today: engine)
npm run test:coverage        # same, plus an LCOV report at
                             # packages/<pkg>/coverage/lcov.info
```

Per-package READMEs cover the workspace-local workflows
(`npm run build`/`serve`/`watch` from inside `packages/<pkg>/`)
and any package-specific setup.

## License

[MIT](LICENSE)
