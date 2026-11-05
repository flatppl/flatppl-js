import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { LINK_LIMITS, MAX_SHARE_URL_LENGTH, shareAddress } from '../src/format.ts';
import { GOLDEN, comparable } from './golden.mjs';
import { readZstdWasm } from '../zstd-wasm.mjs';

const nodeModules = new URL('../../../node_modules/', import.meta.url).pathname;
let temporary, core;

before(async () => {
  temporary = await mkdtemp(join(tmpdir(), 'flatppl-share-'));
  // Hosts bundle the core with the dictionary, as here.
  await build({ entryPoints: [new URL('../src/core.ts', import.meta.url).pathname], bundle: true, platform: 'node',
    format: 'esm', outfile: join(temporary, 'core.mjs'), target: 'node16', loader: { '.zdict': 'binary' } });
  const { createShareCore } = await import(pathToFileURL(join(temporary, 'core.mjs')).href);
  core = createShareCore(() => readZstdWasm(nodeModules));
});

after(() => rm(temporary, { recursive: true, force: true }));

test('golden tokens decode to their container and re-encode to the same tokens', async () => {
  for (const golden of GOLDEN) {
    for (const token of [golden.link, golden.qr]) {
      const decoded = await core.decode(token);
      assert.equal(decoded.link, golden.link, golden.name);
      assert.deepEqual(decoded.view, golden.view, golden.name);
      assert.deepEqual(comparable(decoded.members), comparable(golden.members), golden.name);
    }
    assert.deepEqual(await core.encode(golden.view, golden.members), { link: golden.link, qr: golden.qr }, golden.name);
  }
});

test('refusals name the limit', async () => {
  const main = bytes => [{ name: 'main.flatppl', kind: 'file', bytes }];
  const view = { model: '', target: '' };
  await assert.rejects(core.encode(view, main(new Uint8Array(8 * 1024 ** 2 + 1))), /8 MiB/);
  const random = createHash('shake256', { outputLength: 64 * 1024 }).update('incompressible').digest();
  await assert.rejects(core.encode(view, main(random)), /49,152 bytes/);
  await assert.rejects(core.encode(view, [{ name: 'main.flatppl', kind: 'symlink', bytes: new Uint8Array([97]) }]),
    /main\.flatppl as a file/);
  await assert.rejects(core.encode(view, [...main(new Uint8Array()), { name: 'x'.repeat(1025), kind: 'file', bytes: new Uint8Array() }]),
    /1,024 bytes/);
});

test('the link limits are the browser limits of share-links.md', () => {
  assert.deepEqual({ ...LINK_LIMITS }, { maxArchiveBytes: 9 * 2 ** 20, maxMemberBytes: 2 * 2 ** 20,
    maxTotalBytes: 8 * 2 ** 20, maxEntries: 256, maxDepth: 4, maxPathBytes: 1024 });
});

test('share addresses accept links that carry the bundle and nothing else', () => {
  const token = GOLDEN[0].link, qr = GOLDEN[0].qr;
  const accepted = {
    [token]: { token },
    [qr]: { token: qr },
    [`  https://live.flatppl.org/#share=${token}  `]: { token },
    [`https://example.org/sub/#share=${qr}`]: { token: qr },
    [`HTTPS://LIVE.FLATPPL.ORG/#${qr}`]: { token: qr },
    [`#share=${token}&target=y`]: { token, target: 'y' },
    [`https://live.flatppl.org/#/files/new/shared-model/modules/prior.flatppl?target=mu&share=${token}`]:
      { token, model: 'modules/prior.flatppl', target: 'mu' },
    [`https://live.flatppl.org/#/files/new/shared-model/main.flatppl?share=${token}`]:
      { token, model: 'main.flatppl', target: null },
  };
  for (const [input, address] of Object.entries(accepted)) assert.deepEqual(shareAddress(input), address, input);
  for (const input of [
    'https://live.flatppl.org/',
    'https://live.flatppl.org/#/files/examples/eight-schools.flatppl?target=theta',
    'https://live.flatppl.org/#model=examples/eight-schools.flatppl',
    'https://live.flatppl.org/#archive=https%3A%2F%2Fexample.org%2Fm.zip',
    `https://live.flatppl.org/#share=${token}&archive=m.zip`,
    'examples/eight-schools.flatppl',
    'not a link',
  ]) assert.equal(shareAddress(input), null, input);
  assert.throws(() => shareAddress(`#share=${token}&share=${token}`), /Duplicate/);
  assert.throws(() => shareAddress('#share=' + 'A'.repeat(MAX_SHARE_URL_LENGTH)), /too long/);
});
