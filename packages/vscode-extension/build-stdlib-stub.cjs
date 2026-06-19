// esbuild plugin: resolve every `@stdlib/*` import to a stub that THROWS ONLY
// IF CALLED. Used for the main-thread engine bundle only (NOT the sampler
// worker, which runs the real numerics). The main thread delegates all
// sampling/density to the worker via sendWorker, so it never invokes @stdlib;
// stubbing it keeps ~2.4 MB of @stdlib source out of engine.min.js.
//
// Why not `external: ['@stdlib/*']`? sampler-registry.ts has top-level
// `require('@stdlib/...')` that run at IIFE init — externalizing throws
// "Dynamic require not supported" the moment the bundle loads in the webview.
// Why not lazy require-on-call? esbuild treats `require` as a static edge
// regardless of lexical position, so it pulls the bytes anyway. A stub at
// resolve-time is the only approach that drops the source AND loads cleanly.
const stubStdlibPlugin = {
  name: 'stub-stdlib',
  setup(build) {
    build.onResolve({ filter: /^@stdlib\// }, (args) => ({
      path: args.path,
      namespace: 'stdlib-stub',
    }));
    build.onLoad({ filter: /.*/, namespace: 'stdlib-stub' }, (args) => ({
      contents:
        `const t = () => { throw new Error(` +
        '`@stdlib is not available in the main-thread engine bundle ' +
        '(' + args.path + '); sampling/density must route through the sampler worker`' +
        `); };\nmodule.exports = new Proxy(t, { get: () => t, apply: () => t() });`,
      loader: 'js',
    }));
  },
};

module.exports = { stubStdlibPlugin };
