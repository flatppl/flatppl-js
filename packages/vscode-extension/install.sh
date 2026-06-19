#!/bin/sh
# Detect the host platform, download the matching FlatPPL VS Code extension
# vsix from the nightly release, and install it with the `code` CLI.
set -eu

REPO="flatppl/flatppl-js"
BASE="https://github.com/$REPO/releases/download/nightly"

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
  Darwin/arm64)               target="darwin-arm64" ;;
  Darwin/x86_64)              target="darwin-x64" ;;
  Linux/x86_64)               target="linux-x64" ;;
  Linux/aarch64 | Linux/arm64) target="linux-arm64" ;;
  *)
    echo "Unsupported platform: $os/$arch." >&2
    echo "Build flatppl-lsp yourself and set the 'flatppl.server.path' setting (see README)." >&2
    exit 1
    ;;
esac

vsix="flatppl-vscode-$target-nightly.vsix"
echo "Downloading $vsix ..."
curl -fsSL "$BASE/$vsix" -o "$vsix"

if command -v code >/dev/null 2>&1; then
  code --install-extension "$vsix" --force
  echo "Installed. Reload VS Code (Command Palette -> Developer: Reload Window)."
else
  echo "Downloaded $vsix, but 'code' is not on PATH." >&2
  echo "Install manually: code --install-extension $vsix" >&2
  exit 1
fi
