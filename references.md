# Citations

Foundational and supporting literature for the FlatPPL VS Code extension.
Each entry has a `Relevant for:` tag identifying which aspect of this
software the work informs (`disintegration`, `phase analysis`,
`reification`, …). Add new entries under the existing topical sections, or
introduce new sections as the codebase grows. Each entry's `Relevance to
this software:` note explains specifically what the work contributes.

All references were consulted directly (not just abstracts) unless stated
otherwise.

## Categorical and symbolic foundations

### Cho & Jacobs 2017/2019 — *Disintegration and Bayesian Inversion via String Diagrams*
- **Citation:** Kenta Cho and Bart Jacobs. "Disintegration and Bayesian inversion via string diagrams." *Mathematical Structures in Computer Science*, 29(7):938–971, 2019. arXiv:1709.00322.
- **URL:** https://arxiv.org/abs/1709.00322
- **Relevant for:** disintegration
- **Relevance to this software:** The clean categorical statement of disintegration that exactly matches FlatPPL's spec. A joint state $\omega: I \to X \otimes Y$ disintegrates into a pair $(\omega_1, c_1)$ where $\omega_1$ is the marginal and $c_1: X \to Y$ is the conditional channel; the equation $\omega = (\mathrm{id}_X \otimes c_1) \circ \Delta_X \circ \omega_1$ is precisely "jointchain(prior, kernel) = joint" in FlatPPL terms. Section 6.1 generalises to multipartite states and gives us the rule that any disjoint split of the wires $(X, Y)$ admits a disintegration of $\omega$ once the marginals are chosen — this is the formal justification for the "split a record-typed joint into selected and unselected fields" rule.

### Shan & Ramsey 2017 — *Exact Bayesian Inference by Symbolic Disintegration*
- **Citation:** Chung-chieh Shan and Norman Ramsey. "Exact Bayesian inference by symbolic disintegration." *POPL 2017*.
- **URL:** https://homes.luddy.indiana.edu/ccshan/rational/disintegrator.pdf
- **Relevant for:** disintegration
- **Relevance to this software:** The reference implementation strategy for *general* disintegration — a lazy partial evaluator with three intertwined operators (`constrain-outcome`, `constrain-value`, `perform`, `evaluate`) over a heap. Read carefully to confirm what we are *not* doing: their machinery is needed to invert arbitrary deterministic expressions (e.g. observe `x + y`) and to handle `lebesgue` against compound expressions. FlatPPL's spec explicitly opts out of this generality and asks for "structural disintegration" only, so we skip the constraint-propagation machinery.

### Narayanan, Carette, Romano, Shan, Zinkov 2016 — *Probabilistic Inference by Program Transformation in Hakaru (System Description)*
- **Citation:** Praveen Narayanan, Jacques Carette, Wren Romano, Chung-chieh Shan, Robert Zinkov. "Probabilistic inference by program transformation in Hakaru (system description)." *FLOPS 2016*.
- **URL:** https://www.cs.tufts.edu/comp/150PLD/Papers/Hakaru.pdf
- **Relevant for:** disintegration
- **Relevance to this software:** Confirms the broader Hakaru architecture: disintegration is one of two source-to-source transformations (with simplification), implemented as a syntactic pass that preserves measure semantics. The compositional view — produce a transformed source program, then compose with other transformations — is the architecture we want in FlatPPL: a `disintegrate` plan should yield FlatPPL terms (or equivalent IR) so downstream passes can keep transforming.

### Stein & Staton 2021 — *Compositional Semantics for Probabilistic Programs with Exact Conditioning*
- **Citation:** Dario Stein and Sam Staton. "Compositional Semantics for Probabilistic Programs with Exact Conditioning." *LICS 2021*. arXiv:2101.11351.
- **URL:** https://arxiv.org/abs/2101.11351
- **Relevant for:** disintegration
- **Relevance to this software:** Background for why exact conditioning on continuous observables (Borel's paradox) needs a categorical framing. Mostly relevant as a guard rail against syntactically-plausible local rewrites that break exact-conditioning semantics. We don't need their `Cond` construction directly because FlatPPL's structural fragment doesn't expose continuous equality observations as a primitive — but the paper validates our caution about non-projection pushforwards and weighted/normalised measures.

## Probabilistic graphical models

### Pearl 1988 — *Probabilistic Reasoning in Intelligent Systems*
- **Citation:** Judea Pearl. *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, 1988.
- **Relevant for:** disintegration
- **Relevance to this software:** The Bayesian-network factorization theorem: a distribution that factors as $P(X_1, \dots, X_n) = \prod_i P(X_i \mid \mathrm{Pa}(X_i))$ is exactly what FlatPPL's `lawof(record(...))` joints encode. Reading any standard PGM reference (Pearl, Koller & Friedman, etc.) is enough; what matters is that the factorization is *structural* — it lives in the DAG topology — which is exactly the regime FlatPPL targets.

### Koller & Friedman 2009 — *Probabilistic Graphical Models: Principles and Techniques*
- **Citation:** Daphne Koller and Nir Friedman. *Probabilistic Graphical Models: Principles and Techniques*. MIT Press, 2009.
- **URL:** http://mcb111.org/w06/KollerFriedman.pdf
- **Relevant for:** disintegration
- **Relevance to this software:** Same textbook story as Pearl, with more formalism. Chapters on factorization and on conditional independence (d-separation) cover the prerequisites for our admissibility test ("base-side factor must not depend on kernel-side stochastic node").

## Compiler-pass framings

### Holtzen, Van den Broeck, Millstein 2018 — *Sound Abstraction and Decomposition of Probabilistic Programs*
- **Citation:** Steven Holtzen, Guy Van den Broeck, Todd Millstein. "Sound abstraction and decomposition of probabilistic programs." *ICML 2018*.
- **URL:** https://proceedings.mlr.press/v80/holtzen18a/holtzen18a.pdf
- **Relevant for:** disintegration
- **Relevance to this software:** Confirms that splitting a probabilistic program along causal cuts is a tractable compiler pass for discrete programs and gives a rigorous notion of "sound decomposition". Our structural disintegration is a different cut (latent vs. observed, not abstraction levels), but the pass shape — classify nodes, then rebuild two subprograms — is the same.

### Weiser 1981 — *Program Slicing*
- **Citation:** Mark Weiser. "Program slicing." *ICSE 1981*.
- **Relevant for:** disintegration
- **Relevance to this software:** The classical reference for backward slicing on a DAG with respect to a chosen output. Our admissibility-and-cut step is exactly program slicing applied to a stochastic-graph fragment: "give me the smallest subgraph that produces these selected variates." Cited because it's the right canonical reference for the static-analysis half of the algorithm, not because we need anything from it that isn't in any compiler textbook.

## Honourable mentions (read but not directly used)

- **Chang & Pollard 1997**, "Conditioning as disintegration." *Statistica Neerlandica* 51(3):287–317. *Relevant for:* disintegration. The measure-theoretic origin story; FlatPPL's `disintegrate` directly cites the structural decomposition this paper formalises. Worth reading for the underlying probability theory, not for compiler-side rules.
- **Narayanan & Shan 2017**, "Symbolic conditioning of arrays in probabilistic programs." *ICFP 2017*. *Relevant for:* disintegration. Adds array support to Hakaru disintegration — relevant if FlatPPL ever needs structural disintegration through `iid` along the array axis (we currently treat that as "supported only when the split is uniform per replicated element").
- **Holtzen, Qian, Millstein, Van den Broeck 2019**, "Factorized Exact Inference for Discrete Probabilistic Programs." *LAFI 2019*. *Relevant for:* disintegration. Discrete-only, but the worked examples for "compile to a factorisation" mirror the rebuild step in our pass.
