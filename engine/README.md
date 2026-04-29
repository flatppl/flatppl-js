# flatppl-engine

A reusable JavaScript engine for the [FlatPPL](https://github.com/flatppl/flatppl-design)
probabilistic language. Provides tokenization, recursive-descent parsing,
semantic analysis (binding classification, dependency tracking, diagnostics),
and ancestor sub-DAG computation.

This package has zero runtime dependencies and uses only Node.js built-ins.
It is designed to be extractable as an independent package — currently it
lives inside the `flatppl-vscode` extension repo, but is structured so
nothing inside this directory depends on `vscode` or any code outside
`engine/`.

## Usage

```js
const { processSource, computeSubDAG } = require('flatppl-engine');

const source = `
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
m = lawof(x)
`;

const { ast, bindings, symbols, diagnostics } = processSource(source);

// Diagnostics list parse errors, undefined references, illegal forms, etc.
for (const d of diagnostics) {
  console.log(`${d.severity} at line ${d.loc.start.line + 1}: ${d.message}`);
}

// Compute the ancestor sub-DAG of a node.
const dag = computeSubDAG(bindings, 'm');
console.log(dag.nodes.map(n => n.id), dag.edges);
```

## Modules

- [`tokenizer.js`](tokenizer.js) — source text → token stream
- [`ast.js`](ast.js) — AST node constructors
- [`parser.js`](parser.js) — recursive-descent parser → AST + diagnostics
- [`analyzer.js`](analyzer.js) — scope, classification, dependencies, diagnostics
- [`dag.js`](dag.js) — ancestor sub-DAG extraction
- [`builtins.js`](builtins.js) — catalog of known FlatPPL names
- [`index.js`](index.js) — public API

## Testing

```sh
node --test test/
```

## License

MIT — see the parent repository for details.
