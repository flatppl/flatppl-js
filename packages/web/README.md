# `@flatppl/web`

Standalone web host for the FlatPPL viewer.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Local development

```sh
# from the flatppl-js workspace root, first time only (or after a repo clean):
npm install

# then in this package:
cd packages/web
npm run dev     # one-command dev mode (recommended) — http://localhost:8001/
```

`npm run dev` runs `build.mjs --watch` and `serve.mjs` together in a
single foreground process (one Ctrl+C tears both down):

- **Engine / sampler-worker / CodeMirror / viewer** bundles all
  re-build via esbuild whenever any of their source modules change
  (`packages/engine/**` for the first three, `packages/viewer/src/**`
  for the viewer).
- **The browser auto-reloads** on any change under `dist/`: `serve.mjs`
  exposes an SSE channel at `/__livereload`, injects a tiny
  `EventSource` listener into served HTML, and pushes a `reload` event
  on every `fs.watch` notification from `dist/` (debounced 80 ms). No
  manual refresh required — edit source, save, the page reloads.

Pin a port if 8001 is taken: `PORT=8002 npm run dev`.

The underlying scripts are still available individually if you'd
rather run them in separate terminals:

- `npm run build` — one-shot build of `dist/` (vendor + page sources + demo).
- `npm run watch` — esbuild watch on engine, sampler-worker,
  CodeMirror, and viewer bundles.
- `npm run serve` — static server on http://localhost:8001/ with
  the SSE live-reload channel.

Live-reload is a local-dev-only feature; the SSE channel and the
injected listener script are added by `serve.mjs` at request time and
are not present in deployed builds.

## Demo and example content

Demo content is composed of `web/demo` and
[flatppl-examples](https://github.com/flatppl/flatppl-examples)

## License

[MIT](LICENSE)
