// Tiny zero-dependency static file server for local viewer testing.
//
// Serves packages/viewer/ on http://localhost:8000/ (or PORT). A bare
// "/" redirects to embed-test.html, the page most users actually want
// to see. Anything else maps directly to a file path under the viewer
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
const PORT = Number(process.env.PORT || 8000);

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

server.listen(PORT, () => {
  console.log(`@flatppl/viewer test server: http://localhost:${PORT}/`);
  console.log('  (defaults to embed-test.html; Ctrl+C to stop)');
});
