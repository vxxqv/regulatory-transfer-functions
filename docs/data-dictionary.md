# Data dictionary

## Core identifiers

| Field | Meaning |
|---|---|
| `target_contrast` | Stable perturbation contrast identifier from the source study |
| `target_contrast_gene_name` | Perturbed gene symbol |
| `culture_condition` | Rest, 8-hour stimulation, or 48-hour stimulation |
| `target` | Harmonized target symbol used across datasets |
| `locus` | Frozen disease-locus identifier |
| `variant` or `rsid` | Harmonized genetic variant identifier |

## Primary quantities

| Field | Meaning |
|---|---|
| `ontarget_effect_size` | Released on-target differential-expression z score |
| `n_de_genes` | Significant downstream genes after excluding the perturbed target |
| `predicted_log_response` | Target-held-out expected log1p distal burden |
| `transfer_residual` | Observed minus expected log1p distal burden |
| `transfer_z` | Standardized transfer residual |
| `transfer_class` | Frozen lower-decile buffered, upper-decile amplified, or intermediate label |
| `signed_response_norm` | Magnitude of the signed distal response vector |
| `module_energy` | Fraction of signed response energy assigned to a reference component |
| `rerouting` | Change in downstream module composition across contexts |

## Statistical fields

| Field | Meaning |
|---|---|
| `estimate` | Endpoint-specific point estimate |
| `ci_low`, `ci_high` | Clustered or target-bootstrap 95% interval unless otherwise labeled |
| `p_value` | Raw test probability |
| `q_value` | False-discovery adjusted probability within the prespecified family |
| `decision` | Passed, failed, unresolved, unavailable, underpowered, or not tested |
| `reason` | Explicit exclusion or unavailability explanation |

Units, transformations, grouping variables, and endpoint-specific definitions are specified in `docs/protocol.md` and each analysis README.
