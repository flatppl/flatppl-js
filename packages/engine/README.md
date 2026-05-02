# @flatppl/engine

The JavaScript reference engine for [FlatPPL](https://github.com/flatppl/flatppl-design),
the Flat Portable Probabilistic Language.

> Note: The FlatPPL JavaScript engine is in early development and may be
> unstable. It is not yet published to the npm registry; it currently
> ships only as part of the [`flatppl-js`](https://github.com/flatppl/flatppl-js)
> monorepo and is consumed via npm workspace symlinks by sibling packages.

This package has no runtime dependencies and uses only Node.js built-ins.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models. The language is host-language-neutral; this package is
one implementation among several planned across language ecosystems (Rust,
Julia, Python, …). Cross-implementation conformance is anchored by
[`flatppl-design`](https://github.com/flatppl/flatppl-design) (the spec) and
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples) (the shared
example suite).

## Modules

- [`tokenizer.js`](tokenizer.js) — source text → token stream
- [`ast.js`](ast.js) — AST node constructors
- [`parser.js`](parser.js) — recursive-descent parser → AST + diagnostics
- [`analyzer.js`](analyzer.js) — scope, classification, dependencies, diagnostics
- [`dag.js`](dag.js) — ancestor sub-DAG extraction for visualization
- [`disintegrate.js`](disintegrate.js) — structural disintegration rewriter
- [`builtins.js`](builtins.js) — catalog of known FlatPPL names
- [`index.js`](index.js) — public API

## Testing

```sh
npm test                          # via the workspace's test script
```

Or run directly from inside this package:

```sh
node --test 'test/*.test.js'
```

## License

[MIT](LICENSE)
