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
npm run build   # produces dist/ with vendor + page sources + demo
npm run serve   # http://localhost:8001/
```

Pick a different port if 8001 is taken: `PORT=8002 npm run serve`.

`npm run watch` keeps esbuild in watch mode for the engine and
sampler-worker bundles, so edits to `packages/engine/` re-bundle
automatically.

## Demo and example content

Demo content is composed of `web/demo` and
[flatppl-examples](https://github.com/flatppl/flatppl-examples)

## License

[MIT](LICENSE)
