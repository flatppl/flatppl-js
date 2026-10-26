// The dev server's live-reload client must reach the page as a same-origin
// script file: the gallery's CSP (`script-src 'self'`) blocks inline scripts.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');

async function freePort() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, resolve));
  const { port } = server.address();
  await new Promise(resolve => server.close(resolve));
  return port;
}

test('the dev server serves live reload as a script the CSP allows', {
  skip: !existsSync(join(web, 'dist', 'index.html')) && 'needs a built dist/',
}, async (t) => {
  const port = await freePort();
  const child = spawn(process.execPath, [join(web, 'serve.mjs')], {
    env: { ...process.env, PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  t.after(() => child.kill());
  await new Promise((resolve, reject) => {
    child.stdout.on('data', data => { if (String(data).includes('dev server:')) resolve(); });
    child.on('exit', code => reject(new Error(`serve.mjs exited with ${code}`)));
  });

  const html = await (await fetch(`http://localhost:${port}/`)).text();
  const policy = html.match(/Content-Security-Policy" content="([^"]+)"/)[1];
  assert.doesNotMatch(policy, /unsafe-inline/);
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
  assert.ok(scripts.some(([, attributes]) => attributes.includes('src="/__livereload.js"')));
  for (const [, attributes, body] of scripts) {
    assert.ok(/\bsrc=/.test(attributes) && !body.trim(), `inline script: ${body.trim().slice(0, 80)}`);
  }

  const client = await fetch(`http://localhost:${port}/__livereload.js`);
  assert.equal(client.status, 200);
  assert.match(client.headers.get('content-type'), /javascript/);
  assert.match(await client.text(), /new EventSource\("\/__livereload"\)/);
});
