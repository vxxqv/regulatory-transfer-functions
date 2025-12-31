"""Test frozen credible variants against published CD4 promoter capture Hi-C calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/cd4_pchic_interactions.tsv.gz"
VARIANTS = ROOT / "analyses/loci/results/credible_set_variants.csv"
OUTPUT = ROOT / "analyses/loci/results"
MAPPING = {
    "gata3": {"grch38_start": 8059464, "grch37_start": 8101427},
    "stat3": {"grch38_start": 42340522, "grch37_start": 40492540},
    "ptpn22": {"grch38_start": 113834946, "grch37_start": 114377568},
}
CONTACT_COLUMNS = [
    "locus", "gene", "variant_id", "rsids", "posterior_probability", "grch37_position",
    "bait_chr", "bait_start", "bait_length", "prey_chr", "prey_start", "prey_end",
    "chicago_activated", "chicago_nonactivated", "called_activated", "called_nonactivated",
    "differential_logfc", "differential_fdr",
]


def main() -> None:
    variants = pd.read_csv(VARIANTS)
    interactions = pd.read_csv(SOURCE, sep=r"\s+", compression="gzip")
    interactions["baitChr"] = interactions["baitChr"].astype(str).str.removeprefix("chr")
    interactions["oeChr"] = interactions["oeChr"].astype(str).str.removeprefix("chr")
    interactions["oeEnd"] = interactions["oeStart"] + interactions["oeLength"]
    interactions["called_activated"] = interactions["Total_CD4_Activated"] > 5
    interactions["called_nonactivated"] = interactions["Total_CD4_NonActivated"] > 5

    variant_rows = []
    contact_rows = []
    for variant in variants.itertuples(index=False):
        mapping = MAPPING[variant.locus]
        grch37_position = int(mapping["grch37_start"] + (variant.position - mapping["grch38_start"]))
        gene_interactions = interactions[
            (interactions["gene"] == variant.gene)
            & (interactions["oeChr"] == str(variant.chromosome))
        ].copy()
        called_gene = gene_interactions[
            gene_interactions["called_activated"] | gene_interactions["called_nonactivated"]
        ].copy()
        candidates = gene_interactions[
            (gene_interactions["oeStart"] <= grch37_position)
            & (gene_interactions["oeEnd"] >= grch37_position)
        ].copy()
        called = candidates[candidates["called_activated"] | candidates["called_nonactivated"]]
        if len(called_gene):
            distance = np.maximum(
                np.maximum(called_gene["oeStart"].to_numpy() - grch37_position, 0),
                np.maximum(grch37_position - called_gene["oeEnd"].to_numpy(), 0),
            )
            nearest_called_contact_bp = int(distance.min())
        else:
            nearest_called_contact_bp = None
        variant_rows.append(
            {
                "locus": variant.locus,
                "gene": variant.gene,
                "variant_id": variant.variant_id,
                "rsids": variant.rsids,
                "posterior_probability": variant.posterior_probability,
                "grch38_position": int(variant.position),
                "grch37_position": grch37_position,
                "gene_interaction_rows": len(gene_interactions),
                "called_gene_interaction_rows": len(called_gene),
                "nearest_called_contact_bp": nearest_called_contact_bp,
                "overlapping_candidate_rows": len(candidates),
                "overlapping_called_contacts": len(called),
                "called_contact": bool(len(called)),
            }
        )
        for contact in called.itertuples(index=False):
            contact_rows.append(
                {
                    "locus": variant.locus,
                    "gene": variant.gene,
                    "variant_id": variant.variant_id,
                    "rsids": variant.rsids,
                    "posterior_probability": variant.posterior_probability,
                    "grch37_position": grch37_position,
                    "bait_chr": contact.baitChr,
                    "bait_start": contact.baitStart,
                    "bait_length": contact.baitLength,
                    "prey_chr": contact.oeChr,
                    "prey_start": contact.oeStart,
                    "prey_end": contact.oeEnd,
                    "chicago_activated": contact.Total_CD4_Activated,
                    "chicago_nonactivated": contact.Total_CD4_NonActivated,
                    "called_activated": contact.called_activated,
                    "called_nonactivated": contact.called_nonactivated,
                    "differential_logfc": contact.logFC,
                    "differential_fdr": contact.FDR,
                }
            )

    variant_table = pd.DataFrame(variant_rows)
    contacts = pd.DataFrame(contact_rows, columns=CONTACT_COLUMNS)
    variant_table.to_csv(OUTPUT / "credible_variant_pchic_tests.csv", index=False)
    contacts.to_csv(OUTPUT / "credible_variant_pchic_contacts.csv", index=False)
    summary = variant_table.groupby(["locus", "gene"], as_index=False).agg(
        credible_variants=("variant_id", "size"),
        variants_with_called_contact=("called_contact", "sum"),
        posterior_probability_with_called_contact=(
            "posterior_probability",
            lambda values: float(values[variant_table.loc[values.index, "called_contact"]].sum()),
        ),
        nearest_called_contact_bp=("nearest_called_contact_bp", "min"),
    )
    summary.to_csv(OUTPUT / "pchic_locus_summary.csv", index=False)
    audit = {
        "source": "Burren et al. 2017 Additional file 5 Table S4",
        "doi": "10.1186/s13059-017-1285-0",
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_assembly": "GRCh37",
        "mapping_service": "Ensembl REST GRCh38 to GRCh37",
        "interaction_call_rule": "CHiCAGO score greater than 5 in either CD4 state",
        "source_interaction_rows": len(interactions),
        "frozen_credible_variants": len(variant_table),
        "called_contact_rows": len(contacts),
    }
    (OUTPUT / "pchic_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
