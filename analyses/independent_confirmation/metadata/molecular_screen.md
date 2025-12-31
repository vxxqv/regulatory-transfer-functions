# Molecular evidence metadata screen

## Locked scope

This screen used only experiment, study, release, sample-design, identifier, build, replicate, QC, license, and checksum metadata. It did not open signal tracks, peaks, count or expression matrices, perturbation effects, enrichment outputs, response vectors, or outcome figures. The screen is locked to the 284 frozen eligible perturbed TFs, 133 frozen secondary TFs, and the states Rest, Stim8hr, and Stim48hr.

The screened denominator is 27 candidate records. It comprises three TF-specific occupancy records, seven chromatin-context records, two enhancer-gene physical-link records, four independent perturbation records, one permanently excluded development record, two motif releases, seven curated-network records, and one software-wrapper record. The denominator includes the prespecified named resources and the primary-CD4 assay records retained during portal screening; unavailable fields are not imputed. The 27 matching authoritative source records are listed separately.

## Metadata-only compatibility results

TF-specific occupancy is separate from chromatin context. ENCSR470KCE and ENCSR806WWX are CTCF-only ENCODE TF ChIP-seq records. CTCF has exact coverage of 0/284 frozen perturbed TFs and 0/133 frozen secondary TFs. ENCSR093CTT is likewise CTCF-based physical linking with 0/284 and 0/133 coverage. GSE271089 supplies Rest/48-hour occupancy for MED12, CXXC1, and RNA polymerase II, but the enumerated factors also have 0/284 and 0/133 coverage. There is no eligible state-compatible occupancy factor and no Stim8hr TF-specific occupancy record.

ATAC-seq and H3K27ac records establish only chromatin-context availability. They do not count as TF-specific occupancy and cannot advance a mechanistic claim alone. ENCODE activated records use 36 hours, not the frozen Stim48hr state. GSE244035 has strong Rest/48-hour primary-CD4 H3K27ac CUT&RUN metadata, including two donors per KO-cell-state stratum, matched IgG, SEACR, blacklist removal, and spike-in information, but its enumerated perturbation targets have 0/284 and 0/133 overlap. The GSE271090 bundle is therefore ranked but stopped.

The Javierre promoter-capture Hi-C map is independent physical evidence, but it is GRCh37, uses Ensembl release 75 baits, and includes 0/4-hour rather than 0/8/48-hour activation. GSE171737 and GSE190604 are independent perturbation candidates, but their frozen-state, target-overlap, donor, build, or timing metadata do not satisfy the gate. GSE314342 remains permanently excluded because outcomes were inspected during candidate development before freeze.

JASPAR 2026 and HOCOMOCO v12 are versioned motif resources, not physical or perturbational confirmation. DoRothEA, TRRUST, RegNetwork, CollecTRI, TFTG, and OmniPath are one correlated curated-network evidence family. decoupleR is software, not a separate evidence family. Frozen source coverage is recorded without privileging any named cascade: DoRothEA-A 120/284 and 84/133; TRRUST v2 268/284 and 119/133; historical RegNetwork 43/284 and 40/133; CollecTRI 272/284 and 121/133. Current-resource coverage is unavailable where no immutable release-specific mapping was locked.

No experiment-level provider checksum was exposed in the screened metadata records, so every provider checksum field is explicitly unavailable and unverified. Dataset reuse licenses are also marked unavailable unless the official release record supplied an explicit license. These unavailable fields stop the corresponding source-record gate rather than being inferred from an article license.

## Ranking rule

Selection rank uses only locked metadata. The score is the sum of: +8 for verified nonzero frozen factor coverage, +4 each for passing state and time gates, +1 for a screened provider record, +3 each for independent physical and perturbational evidence, +2 for alternative-cascade capability, and +1 for a primary-cell record; with -3 for chromatin-context-only, -5 for motif-only, and -5 for the correlated curated-network family. The permanently excluded record scores -999. Ties sort by accession. Outcomes are absent from both the score and tie-break.

## Gate decisions

| Hypothesis | Decision | Metadata-only reason |
| --- | --- | --- |
| H25 | STOP | No state-compatible TF-specific occupancy target overlaps either frozen factor list. The only ENCODE TF occupancy target is CTCF at 0/284 and 0/133; the exact Rest/48-hour GSE271089 targets are also 0/284 and 0/133. Chromatin context and motifs cannot substitute. |
| H26 | STOP | No source bundle establishes the ordered direct edge to secondary-TF activity to downstream response across the frozen 0/8/48-hour states. Stim8hr occupancy is absent, ENCODE activation is 36 hours, and the exact Rest/48-hour perturbation bundle has no frozen factor overlap. |

No candidate advances to downstream molecular outcome analysis.
