# @flatppl/engine

The JavaScript reference engine for [FlatPPL](https://github.com/flatppl/flatppl-design),
the Flat Portable Probabilistic Language.

> Note: The FlatPPL JavaScript engine is in early development and may be
> unstable. It is not yet published to the npm registry; it currently
> ships only as part of the [`flatppl-js`](https://github.com/flatppl/flatppl-js)
> monorepo and is consumed via npm workspace symlinks by sibling packages.

The engine uses `@stdlib` distribution and random-sampling packages at runtime.
The current workspace runs TypeScript sources directly. Package publication
still requires a complete JavaScript build and a consumer installation check.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models. The language is host-language-neutral; this package is
one implementation among several planned across language ecosystems (Rust,
Julia, Python, …). Cross-implementation conformance is anchored by
[`flatppl-design`](https://github.com/flatppl/flatppl-design) (the spec) and
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples) (the shared
example suite).

## Modules

- [`tokenizer.ts`](tokenizer.ts) — source text → token stream
- [`ast.ts`](ast.ts) — AST node constructors
- [`parser.ts`](parser.ts) — recursive-descent parser → AST + diagnostics
- [`analyzer.ts`](analyzer.ts) — scope, classification, dependencies, diagnostics
- [`dag.ts`](dag.ts) — ancestor sub-DAG extraction
- [`disintegrate.ts`](disintegrate.ts) — structural disintegration rewriter
- [`builtins.ts`](builtins.ts) — catalog of known FlatPPL names
- [`index.ts`](index.ts) — public API

## Testing

```sh
npm test                          # via the workspace's test script
```

Or run directly from inside this package:

```sh
node --test 'test/**/*.test.ts'
```

These commands require a Node version that runs the repository's TypeScript
sources. The audit checks used Node 26.3.0.

## License

[MIT](LICENSE)
