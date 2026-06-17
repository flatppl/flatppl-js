# FlatPPL — Visual Studio Code Extension

VS Code support for [FlatPPL](https://github.com/flatppl/flatppl-design), the
Flat Portable Probabilistic Language.

> Note: This extension is in early development and may be unstable. It is not
> yet published to the VS Code Marketplace; install nightly builds via the
> command below.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Features

Work in progress. Currently:

- Syntax highlighting for `.flatppl` files (and FlatPPL fenced blocks in Markdown)
- Diagnostics, hover, go-to-definition, document symbols, rename
- Interactive ancestor-DAG visualization with reification scopes,
  per-distribution coloring, and tether annotations

## Language server

The extension bundles a `flatppl-lsp` language server that starts automatically
for any `.flatppl` file. No separate install is needed — a per-platform binary
ships inside the extension.

### What it provides

- **Diagnostics** — type errors and undefined-variable reports as you type
- **Hover** — inferred type of any expression or binding on cursor
- **Completion** — distribution names, built-in functions, and in-scope bindings
- **Go-to-definition** — jumps to the definition site, including across
  `load_module` imports into other files
- **Document and workspace symbols** — outline view and workspace-wide symbol
  search
- **Inlay type hints** — inferred types shown inline next to bindings

### Settings

`flatppl.server.path`
: Absolute path to a `flatppl-lsp` binary to use **instead of** the bundled
  one. Useful when you want to test a local build, for example
  `target/release/flatppl-lsp` from a `cargo build --release -p flatppl-lsp`
  run. Leave empty to use the bundled binary.

`flatppl.catalogues.path`
: Absolute path to a directory of external standard-module catalogue files
  (`*.ron`). Every `*.ron` file there is loaded and supplied to the server so
  that third-party modules' distribution signatures resolve correctly in
  diagnostics, hover, and completion. Leave empty to use the built-in catalogue
  only.

  Changing either setting restarts the server. Catalogue files inside the open
  workspace are watched and trigger a live reload when edited; a catalogue
  directory outside the workspace is read once on server start/restart and is
  not auto-watched.

### Other editors

`flatppl-lsp` is a standalone LSP binary. Per-platform builds are published as
raw assets on the `flatppl-rust` nightly GitHub release, named
`flatppl-lsp-<target-triple>` (e.g. `flatppl-lsp-x86_64-unknown-linux-musl`,
`flatppl-lsp-aarch64-apple-darwin`, `flatppl-lsp-x86_64-pc-windows-msvc.exe`).
Download the binary for your platform, place it on your `PATH`, and configure
your editor as shown below.

#### Helix

In `~/.config/helix/languages.toml`:

```toml
[language-server.flatppl-lsp]
command = "flatppl-lsp"

[[language]]
name = "flatppl"
scope = "source.flatppl"
file-types = ["flatppl"]
language-servers = ["flatppl-lsp"]
```

#### Neovim

Using `vim.lsp.start` (e.g. in `~/.config/nvim/ftplugin/flatppl.lua` or your
LSP config):

```lua
vim.lsp.start({
  name = 'flatppl-lsp',
  cmd = { 'flatppl-lsp' },
  root_dir = vim.fs.dirname(vim.fs.find({ '.git' }, { upward = true })[1]),
})
```

#### Emacs (eglot)

```elisp
(add-to-list 'eglot-server-programs '(flatppl-mode . ("flatppl-lsp")))
```

#### External catalogues in non-VSCode editors

Pass catalogues via the LSP `initializationOptions`, an object of the form:

```json
{ "catalogues": ["<ron source string>", ...] }
```

Each entry is the full text of a catalogue `.ron` file. VSCode does this
automatically from `flatppl.catalogues.path`; other editors must supply the
strings manually in their LSP client configuration.

## Installation

### Nightly build

```sh
curl -L https://github.com/flatppl/flatppl-js/releases/download/nightly/flatppl-vscode-extension-nightly.vsix \
    -o flatppl-vscode-extension-nightly.vsix
code --install-extension flatppl-vscode-extension-nightly.vsix
```

## Development

This package is part of the [`flatppl-js`](https://github.com/flatppl/flatppl-js)
npm workspace monorepo and shares the engine, build script, and grammars
with sibling packages. See the [repo-level README](../../README.md) for the
overall layout.

### Local development (VS Code Desktop)

```sh
git clone https://github.com/flatppl/flatppl-js
cd flatppl-js
npm install
npm run build:vendor
ln -s "$(realpath packages/vscode-extension)" ~/.vscode/extensions/flatppl.flatppl
```

### Remote development (VS Code Remote-SSH)

On the remote host:

```sh
git clone https://github.com/flatppl/flatppl-js
cd flatppl-js
npm install
npm run build:vendor
ln -s "$(realpath packages/vscode-extension)" ~/.vscode-server/extensions/flatppl.flatppl
```

The symlink name *must* be `flatppl.flatppl` (matching `publisher.name` from
[`package.json`](package.json)).

### When to re-run `build:vendor`

- After changes anywhere under [`packages/engine/`](../engine) (the engine
  is bundled into `lib/engine.min.js` for the FlatPPL visualization)
- After dependency changes in any `package.json`
- After grammar changes in
  [`flatppl-grammars`](https://github.com/flatppl/flatppl-grammars)

For active engine development, `npm run watch:vendor` rebuilds the engine
bundle on save (sub-second). Reload the VS Code window after each rebuild.

### Fast grammar iteration

Clone `flatppl-grammars` as a sibling of `flatppl-js`:

```
some/workdir/
├── flatppl-grammars
└── flatppl-js
```

The build script auto-detects the sibling clone and copies grammar files
from there on every `npm run build:vendor`. Without a sibling, it fetches
the pinned ref from GitHub.

### Reloading

After code changes, use VS Code's "Developer: Reload Window" command.

## License

[MIT](LICENSE)
