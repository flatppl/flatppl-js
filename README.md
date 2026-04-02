# FlatPPL for Visual Studio Code

VS-Code support for FlatPPL, the Flat Portable Probabilistic Language.

Note: This extension is in early development and may be unstable. It is not published on the VS Code Marketplace yet, see below for installation instructions.

## Features

- Syntax highlighting for `.flatppl` files
- Highlighting of FlatPPL code blocks in Markdown preview (` ```flatppl `)
- Comment toggling, bracket matching, and auto-closing pairs

## Installation

### Install latest dev build

```sh
curl -L https://github.com/flatppl/flatppl-vscode/releases/download/dev/flatppl-0.1.0-dev.vsix -o flatppl-0.1.0-dev.vsix
code --install-extension flatppl-0.1.0-dev.vsix
```

### Local development

```sh
git clone https://github.com/flatppl/flatppl-vscode
ln -s `realpath flatppl-vscode` ~/.vscode/extensions/flatppl-vscode
```

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## License

[MIT](LICENSE)
