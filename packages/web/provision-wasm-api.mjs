// Provision the flatppl-wasm-api artifact (wasm-pack `--target web` output:
// the ESM glue `flatppl_wasm_api.js` + `flatppl_wasm_api_bg.wasm`) into a
// host's vendor directory. Shared by the web gallery build (dist/vendor/)
// and the VS Code extension build (lib/): the ONE artifact serves both the
// gallery's Convert command (pyhf/HS3 → FlatPPL) and the viewer's math pane
// (`render_math`) in either host.
//
// Source precedence, same no-magic stance as the LSP build:
//   (a) FLATPPL_WASM_DIR — a CI-staged prebuilt artifact; needs no toolchain.
//   (b) the flatppl-rust sibling (FLATPPL_RUST_DIR, else ../flatppl-rust next
//       to this repo), built with the wasm toolchain ON PATH (cargo +
//       wasm-pack + the wasm32 target). The build NEVER installs any of that —
//       `rustup target add` / `cargo install` mutate the user's whole Rust
//       installation, which is their job, not a project build's — and errors
//       with the exact one-time setup commands when a piece is missing.
//   No download fallback. FLATPPL_CONVERT=off opts out deterministically (the
//   artifact is removed from the destination and `false` is returned), so a
//   build that exits 0 has a PREDICTABLE state for both features.

import { copyFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { spawn } from 'node:child_process';

export const WASM_GLUE = 'flatppl_wasm_api.js';
export const WASM_BIN  = 'flatppl_wasm_api_bg.wasm';

/**
 * @param {object} opts
 * @param {string} opts.destDir   directory that receives the glue + wasm
 * @param {string} opts.repoRoot  flatppl-js checkout root (sibling lookup)
 * @param {(line: string) => void} [opts.log]
 * @returns {Promise<boolean>} whether the artifact is present in destDir
 */
export async function provisionWasmApi({ destDir, repoRoot, log = console.log }) {
  if (process.env.FLATPPL_CONVERT === 'off') {
    await rm(join(destDir, WASM_GLUE), { force: true });
    await rm(join(destDir, WASM_BIN), { force: true });
    log('  flatppl-wasm-api: DISABLED (FLATPPL_CONVERT=off) — no Convert command, no math pane');
    return false;
  }

  const staged = process.env.FLATPPL_WASM_DIR;
  if (staged && existsSync(join(staged, WASM_GLUE))) {
    await copyFile(join(staged, WASM_GLUE), join(destDir, WASM_GLUE));
    await copyFile(join(staged, WASM_BIN), join(destDir, WASM_BIN));
    log(`  flatppl-wasm-api: ENABLED (staged ${staged})`);
    return true;
  }

  const rustSibling = process.env.FLATPPL_RUST_DIR || join(dirname(repoRoot), 'flatppl-rust');
  const crateDir = join(rustSibling, 'crates', 'wasm-api');
  if (!existsSync(crateDir)) {
    throw new Error(
      'flatppl-wasm-api: the wasm-api crate was not found at\n'
      + `      ${crateDir}\n`
      + '  Clone flatppl-rust as a sibling (or set FLATPPL_RUST_DIR), point '
      + 'FLATPPL_WASM_DIR at a prebuilt artifact, or build without it via '
      + 'FLATPPL_CONVERT=off (drops the Convert command and the math pane).');
  }
  if (!(await hasCmd('wasm-pack'))) {
    throw new Error(
      'flatppl-wasm-api: wasm-pack not found on PATH — needed to build the wasm '
      + 'API. Set up the wasm build toolchain once yourself (the build will not '
      + 'mutate your global Rust install):\n'
      + '      rustup target add wasm32-unknown-unknown\n'
      + '      cargo install wasm-pack\n'
      + '  Or build without it via FLATPPL_CONVERT=off.');
  }
  const pkg = join(crateDir, 'pkg');
  await runCmd('wasm-pack', ['build', '--target', 'web', '--release',
    '--no-typescript', '--out-dir', pkg, crateDir]);
  await copyFile(join(pkg, WASM_GLUE), join(destDir, WASM_GLUE));
  await copyFile(join(pkg, WASM_BIN), join(destDir, WASM_BIN));
  log('  flatppl-wasm-api: ENABLED (built from sibling flatppl-rust)');
  return true;
}

function hasCmd(cmd) {
  return new Promise((resolve) => {
    const p = spawn(cmd, ['--version'], { stdio: 'ignore' });
    p.on('error', () => resolve(false));
    p.on('exit', (code) => resolve(code === 0));
  });
}

function runCmd(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit' });
    p.on('error', reject);
    p.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
  });
}
