// The libzstd binary that src/zstd.ts drives, for the playground and VS Code
// builds. src/zstd.ts calls this binary's minified exports directly. After
// upgrading @bokuweb/zstd-wasm, check those names and the imports, then
// update the hash.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const ZSTD_WASM_SHA256 = '80260eee5795ee8072bb5821e203730c74dca3909938f5ac1128da67f96223df';

/** `nodeModules` is the workspace's hoisted node_modules directory. */
export async function readZstdWasm(nodeModules) {
  const bytes = await readFile(join(nodeModules, '@bokuweb/zstd-wasm/dist/web/zstd.wasm'));
  if (createHash('sha256').update(bytes).digest('hex') !== ZSTD_WASM_SHA256) {
    throw new Error('@bokuweb/zstd-wasm ships an unexpected zstd.wasm. See ZSTD_WASM_SHA256 in packages/share/zstd-wasm.mjs.');
  }
  return bytes;
}
