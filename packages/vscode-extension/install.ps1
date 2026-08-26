# Detect the host, download the matching FlatPPL extension vsix, and install
# it into one or more detected VS Code-compatible editor CLIs (code, cursor,
# code-insiders, codium, windsurf).
# We ship only win32-x64; it runs on Windows arm64 via x64 emulation.
param(
  [string[]]$Target = @(),
  [switch]$All,
  [string]$Vsix
)

$ErrorActionPreference = "Stop"

$repo = "flatppl/flatppl-js"
$base = "https://github.com/$repo/releases/download/nightly"

# VS Code-compatible editor CLIs we know how to target, in preference order.
$knownClis = @("code", "code-insiders", "cursor", "codium", "windsurf")

# A fork's CLI must accept --install-extension, or installing into it is
# a no-op that looks like success.
function Test-CliSupportsInstall([string]$cli) {
  try {
    $help = & $cli --help 2>$null
    return (($help | Out-String) -match [regex]::Escape("--install-extension"))
  } catch {
    return $false
  }
}

$detected = @()
foreach ($cli in $knownClis) {
  if ((Get-Command $cli -ErrorAction SilentlyContinue) -and (Test-CliSupportsInstall $cli)) {
    $detected += $cli
  }
}

# Interactive checklist over $knownClis. Detected CLIs start checked;
# undetected ones are shown but cannot be toggled on. Returns the final
# selection as a string array.
function Show-Checklist {
  $state = @{}
  foreach ($cli in $knownClis) { $state[$cli] = $detected -contains $cli }

  while ($true) {
    Write-Host ""
    Write-Host "Select install targets (toggle by number, enter to confirm):"
    for ($i = 0; $i -lt $knownClis.Count; $i++) {
      $cli = $knownClis[$i]
      $box = if ($state[$cli]) { "[x]" } else { "[ ]" }
      $avail = if ($detected -contains $cli) { "" } else { " (not found)" }
      Write-Host ("  {0} {1} {2}{3}" -f ($i + 1), $box, $cli, $avail)
    }
    $ans = Read-Host ">"
    if ([string]::IsNullOrEmpty($ans)) { break }
    if ($ans -notmatch '^\d+$') {
      Write-Host "Enter a number to toggle, or blank to confirm."
      continue
    }
    $idx = [int]$ans
    if ($idx -ge 1 -and $idx -le $knownClis.Count) {
      $cli = $knownClis[$idx - 1]
      if ($detected -contains $cli) {
        $state[$cli] = -not $state[$cli]
      } else {
        Write-Host "$cli is not installed; cannot select it."
      }
    } else {
      Write-Host "No such option: $ans"
    }
  }

  return $knownClis | Where-Object { $state[$_] }
}

if ($Target.Count -gt 0) {
  $selected = $Target
} elseif ($All) {
  $selected = $detected
} elseif (-not [Console]::IsInputRedirected) {
  $selected = Show-Checklist
} else {
  # Non-interactive (piped stdin, CI): install into every detected CLI.
  $selected = $detected
}

if (-not $selected -or $selected.Count -eq 0) {
  Write-Error "No install targets selected."
  exit 1
}

if ($Vsix) {
  $vsixPath = $Vsix
  if (-not (Test-Path $vsixPath)) {
    Write-Error "No such vsix: $vsixPath"
    exit 1
  }
} else {
  $targetPlatform = "win32-x64"
  $vsixPath = "flatppl-vscode-$targetPlatform-nightly.vsix"
  Write-Host "Downloading $vsixPath ..."
  Invoke-WebRequest "$base/$vsixPath" -OutFile $vsixPath -UseBasicParsing
}

$installed = @()
$failed = @()
foreach ($cli in $selected) {
  if (-not (Get-Command $cli -ErrorAction SilentlyContinue)) {
    Write-Warning "'$cli' not found on PATH; skipping."
    $failed += $cli
    continue
  }
  try {
    & $cli --install-extension $vsixPath --force
    if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
    $installed += $cli
  } catch {
    Write-Warning "'$cli' failed to install the extension: $_"
    $failed += $cli
  }
}

Write-Host ""
Write-Host ("Summary: installed {0}, failed {1}." -f $installed.Count, $failed.Count)
if ($installed.Count -gt 0) { Write-Host "  installed: $($installed -join ' ')" }
if ($failed.Count -gt 0) { Write-Host "  failed: $($failed -join ' ')" }

if ($installed.Count -gt 0) {
  Write-Host "Installed. Reload the editor (Developer: Reload Window)."
  exit 0
} else {
  exit 1
}
