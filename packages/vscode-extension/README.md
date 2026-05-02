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
