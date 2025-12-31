# Molecular cascade analysis

This extension verifies frozen transfer results against state-compatible regulatory evidence. Rules, resources, thresholds, exclusions, and denominators are fixed in `config/molecular_cascade.yaml` and recorded by SHA-256 in `freeze_manifest.json` before regulatory annotations are joined to the existing outcomes.

The extension is retrospective because the primary transfer results already existed. It is not described as a prospective preregistration. Overlapping TF databases are treated as corroborating provenance, not independent observations. Direct-regulation claims require the complete Tier 1 gate. Missing occupancy, guide-resolved vectors, and donor-by-guide effects remain unavailable rather than being imputed.
