# Markdown preview surface

This file is a **Markdown** model entry. The gallery's right pane is a
file-type-dependent *surface*: a `.flatppl` file shows the FlatPPL graph
viewer, while this `.md` file renders here as a live preview through the
viewer's `renderDoc` pipeline (the same Markdown + Temml-MathML renderer
that powers DAG-node tooltips).

## Why this exists

It exercises the Phase-1 surface registry end-to-end:

- selecting a `.flatppl` model mounts the FlatPPL viewer surface;
- selecting this `.md` file **disposes** that surface (tearing down its
  sampler worker, cytoscape, and echarts instances) and mounts the
  Markdown surface;
- switching back re-mounts a fresh FlatPPL viewer.

## A little of everything

Inline math renders too: the Normal density is
$\frac{1}{\sigma\sqrt{2\pi}}\,e^{-(x-\mu)^2 / 2\sigma^2}$, and a display
block:

$$
\log p(x \mid \mu, \sigma) = -\tfrac{1}{2}\log(2\pi\sigma^2)
  - \frac{(x-\mu)^2}{2\sigma^2}.
$$

A fenced code block:

```flatppl
mu = elementof(reals)
x ~ Normal(mu = mu, sigma = 1.0)
```

A small table:

| Surface     | File type   | Right pane            |
|-------------|-------------|-----------------------|
| FlatPPL     | `.flatppl`  | graph + plots         |
| Markdown    | `.md`       | this preview          |
| placeholder | other       | "no visualization yet"|
