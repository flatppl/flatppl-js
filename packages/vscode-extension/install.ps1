# Detect the host, download + install the matching FlatPPL extension vsix.
# We ship only win32-x64; it runs on Windows arm64 via x64 emulation.
$ErrorActionPreference = "Stop"
$repo = "flatppl/flatppl-js"
$base = "https://github.com/$repo/releases/download/nightly"
$target = "win32-x64"
$vsix = "flatppl-vscode-$target-nightly.vsix"

Write-Host "Downloading $vsix ..."
Invoke-WebRequest "$base/$vsix" -OutFile $vsix

if (Get-Command code -ErrorAction SilentlyContinue) {
  code --install-extension $vsix --force
  Write-Host "Installed. Reload VS Code (Developer: Reload Window)."
} else {
  Write-Error "Downloaded $vsix, but 'code' is not on PATH. Install: code --install-extension $vsix"
}
