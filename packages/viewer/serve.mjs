// Tiny zero-dependency static file server for local viewer testing.
//
// Serves packages/viewer/ on http://localhost:8000/ (or PORT, or the
// next free port if 8000 is busy). A bare "/" redirects to
// embed-test.html, the page most users actually want to see.
// Anything else maps directly to a file path under the viewer
// package.
//
// Runs on plain Node — no npm install, no external server tool needed.
// Use `npm run serve` from packages/viewer/ (or directly:
// `node serve.mjs`).

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, relative } from 'node:path';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));   // packages/viewer/

// Port selection: explicit PORT pins one port (single attempt; the
// user picked it deliberately). Otherwise start at the documented
// default 8000 and auto-bump within a small range when the port is
// busy — covers the common case of @flatppl/web's serve.mjs running
// alongside on 8001+, or a stale viewer server holding 8000.
const EXPLICIT_PORT = process.env.PORT ? Number(process.env.PORT) : null;
const START_PORT    = EXPLICIT_PORT != null ? EXPLICIT_PORT : 8000;
const MAX_PORT      = EXPLICIT_PORT != null ? EXPLICIT_PORT : START_PORT + 15;

// Minimal MIME map. Browsers tolerate octet-stream for unknown types
// but JS files need application/javascript so type=module / Worker
// loaders are happy.
const MIME = {
  '.html':  'text/html; charset=utf-8',
  '.js':    'application/javascript; charset=utf-8',
  '.mjs':   'application/javascript; charset=utf-8',
  '.css':   'text/css; charset=utf-8',
  '.json':  'application/json; charset=utf-8',
  '.svg':   'image/svg+xml',
  '.png':   'image/png',
  '.jpg':   'image/jpeg',
  '.ico':   'image/x-icon',
  '.map':   'application/json',
};

const server = createServer(async (req, res) => {
  // Strip query / fragment, decode percent-encoding.
  let urlPath;
  try {
    urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  } catch {
    res.writeHead(400); res.end('400 bad URL'); return;
  }
  if (urlPath === '/') urlPath = '/embed-test.html';

  // Resolve under the package directory. normalize collapses '..';
  // we then ensure the final path stays within `here` to refuse
  // path-traversal.
  const candidate = normalize(join(here, urlPath));
  const rel = relative(here, candidate);
  if (rel.startsWith('..') || rel === '') {
    res.writeHead(403); res.end('403 forbidden'); return;
  }

  try {
    const data = await readFile(candidate);
    const mime = MIME[extname(candidate).toLowerCase()] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch (e) {
    if (e && e.code === 'ENOENT') {
      res.writeHead(404); res.end('404 ' + urlPath);
    } else {
      res.writeHead(500); res.end('500 ' + (e && e.message || 'error'));
    }
  }
});

// On EADDRINUSE: when the user didn't pin a specific port, walk up to
// the next one. When they did, exit cleanly with a hint (no Node stack
// trace).
let attempt = START_PORT;
server.on('error', (err) => {
  if (err && err.code === 'EADDRINUSE') {
    if (attempt < MAX_PORT) {
      attempt++;
      setImmediate(() => server.listen(attempt));
      return;
    }
    if (EXPLICIT_PORT != null) {
      console.error(`@flatppl/viewer test server: port ${EXPLICIT_PORT} is already in use.`);
      console.error('  Pick a different port with: PORT=<n> npm run serve, or unset PORT to auto-select.');
    } else {
      console.error(`@flatppl/viewer test server: no free port in [${START_PORT}, ${MAX_PORT}].`);
      console.error('  Pick one explicitly with: PORT=<n> npm run serve');
    }
    process.exit(1);
  }
  throw err;
});

server.on('listening', () => {
  console.log(`@flatppl/viewer test server: http://localhost:${attempt}/`);
  if (attempt !== START_PORT) {
    console.log(`  (port ${START_PORT} was busy; auto-picked ${attempt})`);
  }
  console.log('  (defaults to embed-test.html; Ctrl+C to stop)');
});

server.listen(attempt);
