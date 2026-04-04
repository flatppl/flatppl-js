# FlatPPL for Visual Studio Code

VS-Code support for FlatPPL, the Flat Portable Probabilistic Language.

Note: This extension is in early development and may be unstable. It is not published on the VS Code Marketplace yet, see below for installation instructions.

## Features

Work in progress.

## Installation

### Install latest dev build

```sh
curl -L https://github.com/flatppl/flatppl-vscode/releases/download/dev/flatppl-0.1.0-dev.vsix -o flatppl-0.1.0-dev.vsix
code --install-extension flatppl-0.1.0-dev.vsix
```

### Extension development

#### Local development (VS Code Desktop)

```sh
git clone https://github.com/flatppl/flatppl-vscode
ln -s "$(realpath flatppl-vscode)" ~/.vscode/extensions/flatppl.flatppl
```

#### Remote development (VS Code Remote-SSH)

On the remote host:

```sh
git clone https://github.com/flatppl/flatppl-vscode
ln -s "$(realpath flatppl-vscode)" ~/.vscode-server/extensions/flatppl.flatppl
```

Note: The symlink name must be `flatppl.flatppl` (matching `publisher.name` from
`package.json`).

#### Reloading

Reload the VS Code window via VS-Code command "Developer: Reload Window" after
making changes to the extension code.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## License

[MIT](LICENSE)
