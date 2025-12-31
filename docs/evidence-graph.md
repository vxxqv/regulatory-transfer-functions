# Typed evidence graph

Every node and edge records provenance. Edges additionally record evidence tier, effect score, uncertainty, and biological state. The graph is constructed only after the primary response vectors are frozen, so external annotations cannot alter response classification.

## Node types

Variant, locus, regulatory element, perturbation, gene, state, module, trait, and study.

## Edge types

Contains, targets, regulates, responds in, loads on, associates with, and supported by. Each relation has an explicit allowed source and target type signature enforced in code.

## Evidence tiers

- D0: primary perturbation response or independent natural-genetic evidence used for the core claim.
- D1: orthogonal experimental evidence used for replication or mechanism.
- D2: curated functional annotation used for interpretation.
- D3: hypothesis-generating computational support.

Scores are not pooled across tiers. Missing evidence remains missing and is never converted into a negative edge.
