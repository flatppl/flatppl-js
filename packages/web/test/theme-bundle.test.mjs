import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { verifyThemeBundle } from '../scripts/verify-theme.mjs';

test('the pinned theme bundle passes its manifest guard and tampering does not', async (t) => {
  assert.deepEqual(await verifyThemeBundle(), []);
  const pinnedManifest = JSON.parse(await readFile(
    new URL('../../../vendor/flatppl-theme/manifest.json', import.meta.url),
    'utf8',
  ));
  assert.equal(pinnedManifest.version, '0.1.6');
  assert.equal(pinnedManifest.source.commit, 'af60af0344e458ff23bb2af9a81f0aedef9c58cd');
  assert.equal(pinnedManifest.source.release, 'v0.1.6');

  const root = await mkdtemp(join(tmpdir(), 'flatppl-theme-test-'));
  t.after(() => import('node:fs/promises').then(({ rm }) =>
    rm(root, { recursive: true, force: true })));
  const bundle = join(root, 'flatppl-theme');
  await cp(new URL('../../../vendor/flatppl-theme', import.meta.url), bundle, { recursive: true });
  await writeFile(join(bundle, 'tokens.css'), 'tampered\n');

  assert.deepEqual(await verifyThemeBundle(bundle), [
    'tokens.css: size mismatch',
    'tokens.css: SHA-256 mismatch',
  ]);
});
