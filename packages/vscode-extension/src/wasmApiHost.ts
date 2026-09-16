'use strict';
// The flatppl-wasm-api artifact as the EXTENSION HOST sees it.
//
// build-vendor.mjs provisions wasm-pack's `--target web` glue and wraps it
// as lib/flatppl_wasm_api.cjs (VS Code loads the extension as CommonJS).
// Every host-side caller of that artifact goes through here: the math
// export commands (`export_math`) and the convert command (`convert`).
//
// Two things this module owns and the callers must not repeat:
//
//   - The glue derives the .wasm location from `import.meta.url`, which the
//     CJS wrap empties, and Node cannot fetch() a file: URL anyway. So init
//     always gets the wasm BYTES, read from the file beside the glue.
//   - The ~2 MB module is instantiated once per extension host, and two
//     first calls share that one instantiation.
//
// No vscode import: the loader, the byte read and the existence check are
// injected, so the unit tests require this module with no host stub — same
// pattern as dependencyPrefetch.ts.

const path = require('path');

/** The CommonJS wrap of the wasm-pack glue, in the extension's lib/. */
export const WASM_API_GLUE = 'flatppl_wasm_api.cjs';
/** The wasm binary wasm-pack emits beside the glue. */
export const WASM_API_BINARY = 'flatppl_wasm_api_bg.wasm';

export interface WasmApiPaths {
  glue: string;
  binary: string;
}

/** Where the build puts the two artifacts for an installed extension. */
export function wasmApiPaths(extensionPath: string): WasmApiPaths {
  return {
    glue: path.join(extensionPath, 'lib', WASM_API_GLUE),
    binary: path.join(extensionPath, 'lib', WASM_API_BINARY),
  };
}

export interface WasmApiHostDeps {
  paths: WasmApiPaths;
  /** `fs.existsSync`. */
  exists: (p: string) => boolean;
  /** Node `require` of the CJS glue. */
  loadGlue: (glue: string) => any;
  /** `fs.readFileSync` of the wasm binary. */
  readBinary: (binary: string) => Uint8Array;
}

export interface WasmApiHost {
  /** Whether the build provisioned the artifact at all. FLATPPL_CONVERT=off
   *  and a build with no flatppl-rust sibling both leave it absent. */
  available(): boolean;
  /** The initialised module, instantiated on first use. */
  api(): Promise<any>;
  /** One named export, or an error naming what the build is missing. */
  entry(name: string): Promise<(...args: any[]) => any>;
}

export function createWasmApiHost(deps: WasmApiHostDeps): WasmApiHost {
  let mod: any = null;
  let loading: Promise<any> | null = null;

  function available(): boolean {
    return deps.exists(deps.paths.glue) && deps.exists(deps.paths.binary);
  }

  async function api(): Promise<any> {
    if (mod) return mod;
    if (!loading) {
      loading = (async () => {
        const m = deps.loadGlue(deps.paths.glue);
        const init = m && (m.default || m);
        if (typeof init !== 'function') {
          throw new Error('the bundled wasm API has no init entry (rebuild lib/ from a current flatppl-rust)');
        }
        await init({ module_or_path: deps.readBinary(deps.paths.binary) });
        mod = m;
        return m;
      })().catch((e: any) => {
        // A failed init must not poison the cache: the next call retries.
        loading = null;
        throw e;
      });
    }
    return loading;
  }

  async function entry(name: string): Promise<(...args: any[]) => any> {
    const m = await api();
    if (typeof m[name] !== 'function') {
      throw new Error('the bundled wasm API has no ' + name + ' (rebuild lib/ from a current flatppl-rust)');
    }
    return m[name];
  }

  return { available, api, entry };
}
