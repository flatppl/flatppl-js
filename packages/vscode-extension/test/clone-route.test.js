// Network-gated integration test for build-vendor.mjs's clone route (precedence
// step 4): no sibling + cargo present → shallow-clone flatppl-rust@main, build
// the LSP, copy the host binary into bin/, and DELETE the temp clone.
//
// This is the test the design doc names under "Testing": "no sibling, no env,
// cargo present → clone route runs: host triple present afterwards, and the
// temp clone dir is gone." It clones over HTTPS and runs a full cargo release
// build, so it is SKIPPED by default — set FLATPPL_NET_TESTS=1 to run it.
//
// The clone route is forced by pointing FLATPPL_RUST_DIR at a path that does
// not exist (→ siblingExists=false) while cargo is on PATH (→ route 'clone'),
// with FLATPPL_LSP_BIN_DIR and FLATPPL_LSP_LOCAL unset.
const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { hostTripleForNode } = require('../build-lsp-source.cjs');

const RUN = process.env.FLATPPL_NET_TESTS === '1';
const pkgDir = path.join(__dirname, '..');

function tmpCloneDirs() {
  // The clone route uses mkdtemp(join(tmpdir(), 'flatppl-rust-')).
  return fs.readdirSync(os.tmpdir()).filter((n) => n.startsWith('flatppl-rust-'));
}

test(
  'clone route: builds host binary from flatppl-rust@main and removes the temp clone',
  { skip: RUN ? false : 'network+cargo; set FLATPPL_NET_TESTS=1 to run', timeout: 15 * 60 * 1000 },
  () => {
    const host = hostTripleForNode(process.platform, process.arch);
    assert.ok(host, 'host triple must be known to run this test');

    const binPath = path.join(pkgDir, 'bin', `flatppl-lsp-${host.triple}${host.exe}`);
    fs.rmSync(binPath, { force: true });

    const before = new Set(tmpCloneDirs());

    const env = { ...process.env, FLATPPL_RUST_DIR: path.join(os.tmpdir(), 'definitely-no-flatppl-rust-here') };
    delete env.FLATPPL_LSP_BIN_DIR;
    delete env.FLATPPL_LSP_LOCAL;

    const res = spawnSync('node', ['./build-vendor.mjs'], { cwd: pkgDir, env, encoding: 'utf8' });
    assert.equal(res.status, 0, `build-vendor.mjs failed:\n${res.stdout}\n${res.stderr}`);
    assert.match(res.stdout, /cloning flatppl-rust@main/, 'expected the clone route to run');

    // Host binary present and executable.
    assert.ok(fs.existsSync(binPath), `expected ${binPath} after clone-route build`);
    if (!host.exe) {
      assert.ok(fs.statSync(binPath).mode & 0o111, 'host binary should be executable');
    }

    // Temp clone removed — no flatppl-rust-* dir leaked beyond what predated us.
    const leaked = tmpCloneDirs().filter((n) => !before.has(n));
    assert.deepEqual(leaked, [], `clone route leaked temp dir(s): ${leaked.join(', ')}`);
  },
);
