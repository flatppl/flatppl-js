// Ambient declaration for the vendored FlatPPL engine bundle.
//
// build-vendor.mjs emits `./lib/engine.min.js` and
// `../lib/engine.min.js` (relative to extension.ts and
// src/visualPanel.ts respectively) as a single CJS-wrapped IIFE whose
// `module.exports` mirrors @flatppl/engine's exports. The bundle is
// gitignored and only exists after a build, so tsc has nothing to
// resolve from the on-disk file; this module declaration is the
// stand-in.
//
// The surface is intentionally typed as `any` — the engine's full
// type surface lives in packages/engine and is large/dynamic; the
// extension only needs the entry shape (a CJS module whose imports
// are callables / value-bags) to satisfy `const { foo, bar } =
// require('./lib/engine.min.js')` destructure sites.

declare module '*lib/engine.min.js' {
  const value: any;
  export = value;
}
