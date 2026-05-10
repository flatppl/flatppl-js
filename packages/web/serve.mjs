// Tiny zero-dependency static file server for local development of
// the @flatppl/web gallery. Serves dist/ on http://localhost:8001/.
//
// Distinct from packages/viewer/serve.mjs (port 8000), which serves
// the viewer-isolated embed-test page. The two test surfaces stay
// independent: the viewer one tests DAG/plot rendering against an
// inline source; this one tests the gallery shell over real model
// files. Run them simultaneously when iterating on cross-package
// concerns.
//
// Run `npm run build` (or `npm run watch`) first so dist/ exists.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, relative } from 'node:path';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here   = dirname(fileURLToPath(import.meta.url));   // packages/web/
const root   = join(here, 'dist');                        // packages/web/dist/

// Port selection: if PORT is set explicitly, honour exactly that one
// (a single attempt; the user picked it deliberately). Otherwise
// start at the documented default 8001 and auto-bump within a small
// range when the port is busy — covers the common case of the
// viewer's serve.mjs running on 8000 while iterating, or a previous
// `npm run serve` that didn't exit cleanly.
const EXPLICIT_PORT = process.env.PORT ? Number(process.env.PORT) : null;
const START_PORT    = EXPLICIT_PORT != null ? EXPLICIT_PORT : 8001;
const MAX_PORT      = EXPLICIT_PORT != null ? EXPLICIT_PORT : START_PORT + 15;

const MIME = {
  '.html':    'text/html; charset=utf-8',
  '.js':      'application/javascript; charset=utf-8',
  '.mjs':     'application/javascript; charset=utf-8',
  '.css':     'text/css; charset=utf-8',
  '.json':    'application/json; charset=utf-8',
  '.svg':     'image/svg+xml',
  '.png':     'image/png',
  '.jpg':     'image/jpeg',
  '.ico':     'image/x-icon',
  '.map':     'application/json',
  '.flatppl': 'text/plain; charset=utf-8',
  '.csv':     'text/csv; charset=utf-8',
  '.wsv':     'text/plain; charset=utf-8',
};

const server = createServer(async (req, res) => {
  let urlPath;
  try {
    urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  } catch {
    res.writeHead(400); res.end('400 bad URL'); return;
  }
  if (urlPath === '/') urlPath = '/index.html';

  const candidate = normalize(join(root, urlPath));
  const rel = relative(root, candidate);
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
      console.error(`@flatppl/web dev server: port ${EXPLICIT_PORT} is already in use.`);
      console.error('  Pick a different port with: PORT=<n> npm run serve, or unset PORT to auto-select.');
    } else {
      console.error(`@flatppl/web dev server: no free port in [${START_PORT}, ${MAX_PORT}].`);
      console.error('  Pick one explicitly with: PORT=<n> npm run serve');
    }
    process.exit(1);
  }
  throw err;
});

server.on('listening', () => {
  console.log(`@flatppl/web dev server: http://localhost:${attempt}/`);
  if (attempt !== START_PORT) {
    console.log(`  (port ${START_PORT} was busy; auto-picked ${attempt})`);
  }
  console.log(`  serving: ${root}`);
  console.log('  (run "npm run build" or "npm run watch" first; Ctrl+C to stop)');
});

server.listen(attempt);
