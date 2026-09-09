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

## Visual theme (flatppl-theme)

The page shell, tokens and brand assets come from
[flatppl-theme](https://github.com/flatppl/flatppl-theme), provisioned
into the git-ignored `vendor/flatppl-theme/` on every build
(`scripts/fetch-theme.mjs`), in this order:

1. `FLATPPL_THEME_DIR`, else a sibling `../flatppl-theme` checkout: its
   source files are copied as they are, **unverified** (the build says
   so). Edit `tokens.css` there, then run the build again (or restart
   `npm run dev`; nothing watches the checkout). `FLATPPL_THEME_NO_SIBLING=1`
   ignores a sibling, so the release path can be tested; an explicit
   `FLATPPL_THEME_DIR` still wins.
2. Otherwise the pinned release (`THEME_PIN` in `fetch-theme.mjs`,
   `FLATPPL_THEME_REF` overrides it for a candidate): the release archive
   is downloaded from GitHub and verified against its own manifest. A
   verified copy already in place is reused. This is what CI does.

Adopting a new theme release is a one-line bump of `THEME_PIN`.
`npm run verify:theme` provisions and reports the state of the copy.

## Syntax highlighting (flatppl-grammars)

The editor's lexical highlighting comes from the canonical TextMate
grammar in [flatppl-grammars](https://github.com/flatppl/flatppl-grammars)
(its `textmate/flatppl.tmLanguage.json` plus the optional CodeMirror
module `codemirror/textmate-highlight.ts`), vendored at build time. The
build resolves the grammar source in this order:

1. `GRAMMARS_DIR` env var (CI checks out flatppl-grammars and points this
   at it for deterministic, offline builds);
2. a sibling `../flatppl-grammars` clone (the natural local-dev layout);
3. otherwise the pinned `main` tarball is fetched from GitHub.

The `flatppl.tmLanguage.json` is required; the CodeMirror module is
optional — if absent, the build writes a no-op stub and the editor
degrades to plain text plus the engine's binding overlay. Same
sibling-first / tarball-fallback pattern as the flatppl-examples sync.

## License

[MIT](LICENSE)
