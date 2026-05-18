// One-command developer mode: runs `build.mjs --watch` and `serve.mjs`
// together as a single foreground process, so a single Ctrl+C tears
// both down cleanly. Edit any source file under packages/engine/ or
// packages/viewer/src/viewer.js and the browser auto-reloads via
// serve.mjs's SSE channel (the dist/ watcher catches the rebuild's
// output write and pushes a reload event). Zero new dependencies —
// just Node's built-in child_process with signal forwarding.
//
// Usage:
//   npm run dev              # http://localhost:8001/ (or next free port)
//   PORT=9000 npm run dev    # pin the serve port
//
// stdio is inherited so both children's output interleaves into this
// terminal. If either child exits we tear down the other so the
// developer doesn't end up with half a stack still running.

import { spawn } from 'node:child_process';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

function run(name, args) {
  const p = spawn(process.execPath, args, { cwd: here, stdio: 'inherit' });
  p.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.error(`[dev] ${name} exited (code=${code}, signal=${signal}); stopping the other process.`);
    shutdown();
  });
  return p;
}

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const p of children) {
    if (!p.killed) {
      try { p.kill('SIGTERM'); } catch (_) {}
    }
  }
}

const children = [
  run('build.mjs --watch', ['build.mjs', '--watch']),
  run('serve.mjs',         ['serve.mjs']),
];

process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);
