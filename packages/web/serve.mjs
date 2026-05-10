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
const PORT   = Number(process.env.PORT || 8001);

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

server.listen(PORT, () => {
  console.log(`@flatppl/web dev server: http://localhost:${PORT}/`);
  console.log(`  serving: ${root}`);
  console.log('  (run "npm run build" or "npm run watch" first; Ctrl+C to stop)');
});
