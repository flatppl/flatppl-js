// Pure, I/O-free LSP-source decision logic for build-vendor.mjs.
//
// CommonJS on purpose: the `node --test` suite (test/*.test.js) is CJS and
// `require`s this directly, while build-vendor.mjs (ESM) imports the same
// named exports. Keeping it free of fs/child_process makes the route choice
// unit-testable; build-vendor.mjs performs the actual side effects.

// Node host -> release triple. Mirror of lspClient.hostTriple; kept in JS
// (not the TS source) so the build script needs no transpile step.
function hostTripleForNode(platform, arch) {
  const exe = platform === 'win32' ? '.exe' : '';
  if (platform === 'darwin' && arch === 'arm64') return { triple: 'aarch64-apple-darwin', exe };
  if (platform === 'darwin' && arch === 'x64')   return { triple: 'x86_64-apple-darwin', exe };
  if (platform === 'linux'  && arch === 'x64')   return { triple: 'x86_64-unknown-linux-musl', exe };
  if (platform === 'linux'  && arch === 'arm64') return { triple: 'aarch64-unknown-linux-musl', exe };
  if (platform === 'win32'  && arch === 'x64')   return { triple: 'x86_64-pc-windows-msvc', exe };
  return null;
}

// Decide where the host LSP binary comes from, by precedence. Pure: callers
// pass already-probed signals (env strings, existsSync result, cargo probe).
//
// A staged (`bin-dir`, CI) or `local` prebuilt wins and needs no toolchain.
// Otherwise the binary is BUILT FROM SOURCE — `sibling` if flatppl-rust is
// checked out next door, else a throwaway `clone` of main — and both require
// cargo, a REQUIRED part of the flatppl-js build toolchain. There is no
// prebuilt-binary download fallback (too magical: you build what you run), so
// without cargo and without a prebuilt the route is `none` and the caller
// stops with an actionable error.
function chooseLspSource({ binDir, localBin, siblingExists, cargoAvailable } = {}) {
  if (binDir)   return { route: 'bin-dir', dir: binDir };
  if (localBin) return { route: 'local', path: localBin };
  if (!cargoAvailable) return { route: 'none' };
  return { route: siblingExists ? 'sibling' : 'clone' };
}

module.exports = { hostTripleForNode, chooseLspSource };
