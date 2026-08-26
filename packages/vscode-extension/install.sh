#!/bin/sh
# Detect the host platform, download the matching FlatPPL VS Code extension
# vsix from the nightly release, and install it into one or more detected
# VS Code-compatible editor CLIs (code, cursor, code-insiders, codium,
# windsurf).
# shellcheck disable=SC2154 # name_N/state_N are set dynamically via eval; read via eval too
set -eu

REPO="flatppl/flatppl-js"
BASE="https://github.com/$REPO/releases/download/nightly"

# VS Code-compatible editor CLIs we know how to target, in preference order.
known_clis="code code-insiders cursor codium windsurf"

# A fork's CLI must accept --install-extension, or installing into it is
# a no-op that looks like success.
cli_supports_install() {
  "$1" --help 2>/dev/null | grep -q -- '--install-extension'
}

detected=""
for cli in $known_clis; do
  if command -v "$cli" >/dev/null 2>&1 && cli_supports_install "$cli"; then
    detected="$detected $cli"
  fi
done
detected="${detected# }"

usage() {
  echo "Usage: install.sh [--target CLI]... [--all] [--vsix PATH] [-h|--help]" >&2
  echo "  --target CLI   install into CLI (repeatable); skips the prompt" >&2
  echo "  --all          install into every detected CLI; skips the prompt" >&2
  echo "  --vsix PATH    install this vsix instead of downloading the nightly" >&2
  echo "Known CLIs: $known_clis" >&2
}

targets=""
mode="auto"
vsix_path=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target)
      shift
      [ $# -gt 0 ] || { echo "--target needs a value" >&2; exit 2; }
      targets="$targets $1"
      mode="explicit"
      shift
      ;;
    --all)
      mode="all"
      shift
      ;;
    --vsix)
      shift
      [ $# -gt 0 ] || { echo "--vsix needs a value" >&2; exit 2; }
      vsix_path="$1"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done
targets="${targets# }"

is_detected() {
  case " $detected " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

# Interactive checklist over $known_clis. Detected CLIs start checked;
# undetected ones are shown but cannot be toggled on. Prints the final
# space-separated selection on stdout; everything else goes to stderr so
# it composes with `selected="$(prompt_checklist)"`.
prompt_checklist() {
  i=0
  for cli in $known_clis; do
    i=$((i + 1))
    eval "name_$i=\$cli"
    if is_detected "$cli"; then
      eval "state_$i=1"
    else
      eval "state_$i=0"
    fi
  done
  count=$i

  while :; do
    echo "" >&2
    echo "Select install targets (toggle by number, enter to confirm):" >&2
    n=1
    while [ "$n" -le "$count" ]; do
      eval "nm=\$name_$n"
      eval "st=\$state_$n"
      if [ "$st" = 1 ]; then box="[x]"; else box="[ ]"; fi
      if is_detected "$nm"; then avail=""; else avail=" (not found)"; fi
      echo "  $n $box $nm$avail" >&2
      n=$((n + 1))
    done
    printf "> " >&2
    read -r ans || ans=""
    [ -z "$ans" ] && break
    case "$ans" in
      *[!0-9]*)
        echo "Enter a number to toggle, or blank to confirm." >&2
        continue
        ;;
    esac
    if [ "$ans" -ge 1 ] && [ "$ans" -le "$count" ]; then
      eval "nm=\$name_$ans"
      if is_detected "$nm"; then
        eval "cur=\$state_$ans"
        if [ "$cur" = 1 ]; then eval "state_$ans=0"; else eval "state_$ans=1"; fi
      else
        echo "$nm is not installed; cannot select it." >&2
      fi
    else
      echo "No such option: $ans" >&2
    fi
  done

  out=""
  n=1
  while [ "$n" -le "$count" ]; do
    eval "nm=\$name_$n"
    eval "st=\$state_$n"
    [ "$st" = 1 ] && out="$out $nm"
    n=$((n + 1))
  done
  echo "${out# }"
}

if [ "$mode" = "explicit" ]; then
  selected="$targets"
elif [ "$mode" = "all" ]; then
  selected="$detected"
elif [ -t 0 ]; then
  selected="$(prompt_checklist)"
else
  # Non-interactive (e.g. `curl ... | sh`): install into every detected CLI.
  selected="$detected"
fi

if [ -z "$selected" ]; then
  echo "No install targets selected." >&2
  exit 1
fi

if [ -n "$vsix_path" ]; then
  vsix="$vsix_path"
  [ -f "$vsix" ] || { echo "error: no such vsix: $vsix" >&2; exit 1; }
else
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os/$arch" in
    Darwin/arm64)               target_platform="darwin-arm64" ;;
    Darwin/x86_64)              target_platform="darwin-x64" ;;
    Linux/x86_64)               target_platform="linux-x64" ;;
    Linux/aarch64 | Linux/arm64) target_platform="linux-arm64" ;;
    *)
      echo "Unsupported platform: $os/$arch." >&2
      echo "Build flatppl-lsp yourself and set the 'flatppl.server.path' setting (see README)." >&2
      exit 1
      ;;
  esac
  vsix="flatppl-vscode-$target_platform-nightly.vsix"
  echo "Downloading $vsix ..."
  curl -fsSL "$BASE/$vsix" -o "$vsix"
fi

installed=""
failed=""
for cli in $selected; do
  if ! command -v "$cli" >/dev/null 2>&1; then
    echo "error: '$cli' not found on PATH; skipping." >&2
    failed="$failed $cli"
    continue
  fi
  if "$cli" --install-extension "$vsix" --force; then
    installed="$installed $cli"
  else
    echo "error: '$cli' failed to install the extension." >&2
    failed="$failed $cli"
  fi
done
installed="${installed# }"
failed="${failed# }"

n_installed=0
# shellcheck disable=SC2086 # word-splitting $installed on purpose, to count entries
[ -n "$installed" ] && n_installed=$(set -- $installed; echo $#)
n_failed=0
# shellcheck disable=SC2086 # word-splitting $failed on purpose, to count entries
[ -n "$failed" ] && n_failed=$(set -- $failed; echo $#)

echo ""
echo "Summary: installed $n_installed, failed $n_failed."
[ -n "$installed" ] && echo "  installed: $installed"
[ -n "$failed" ] && echo "  failed: $failed"

if [ "$n_installed" -gt 0 ]; then
  echo "Reload the editor (Command Palette -> Developer: Reload Window) to pick up the extension."
  exit 0
else
  exit 1
fi
