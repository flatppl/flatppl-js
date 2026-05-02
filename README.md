# FlatPPL for Visual Studio Code

VS-Code support for FlatPPL, the Flat Portable Probabilistic Language.

Note: This extension is in early development and may be unstable. It is not published on the VS Code Marketplace yet, see below for installation instructions.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Features

Work in progress.

## Installation

### Install latest nightly build

```sh
curl -L https://github.com/flatppl/flatppl-vscode/releases/download/nightly/flatppl-0.1.0-nightly.vsix -o flatppl-0.1.0-nightly.vsix
code --install-extension flatppl-0.1.0-nightly.vsix
```

### Extension development

#### Local development (VS Code Desktop)

```sh
git clone https://github.com/flatppl/flatppl-vscode
cd flatppl-vscode
npm install
npm run build:vendor
cd ..
ln -s "$(realpath flatppl-vscode)" ~/.vscode/extensions/flatppl.flatppl
```

Re-run `npm run build:vendor` after changing dependencies in `package.json`.
The build script also regenerates `THIRD-PARTY-LICENSES.md` from the actual
bundled deps.

#### Remote development (VS Code Remote-SSH)

On the remote host:

```sh
git clone https://github.com/flatppl/flatppl-vscode
cd flatppl-vscode
npm install
npm run build:vendor
cd ..
ln -s "$(realpath flatppl-vscode)" ~/.vscode-server/extensions/flatppl.flatppl
```

Note: The symlink name must be `flatppl.flatppl` (matching `publisher.name` from
`package.json`).

#### Reloading

Reload the VS Code window via VS-Code command "Developer: Reload Window" after
making changes to the extension code.

#### Running tests

The language engine (tokenizer, parser, analyzer, DAG extraction) lives in
[`engine/`](engine/) and has its own test suite based on the Node.js
built-in test runner (no extra dependencies required).

From the repository root:

```sh
npm test
```

Or directly from inside the engine directory:

```sh
cd engine
node --test 'test/*.test.js'
```

See [`engine/README.md`](engine/README.md) for details.

## License

[MIT](LICENSE)
