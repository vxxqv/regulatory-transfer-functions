"""Test frozen credible variants against published CD4 promoter capture Hi-C calls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from pyliftover import LiftOver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.common.strict_liftover import lift_interval


SOURCE = ROOT / "work/cd4_pchic_interactions.tsv.gz"
CHAIN = ROOT / "work/regulatory_inputs/hg19ToHg38.over.chain.gz"
VARIANTS = ROOT / "analyses/loci/results/credible_set_variants.csv"
OUTPUT = ROOT / "analyses/loci/results"
CONTACT_COLUMNS = [
    "locus", "gene", "variant_id", "rsids", "posterior_probability",
    "grch38_position", "grch38_position_0based", "bait_chr_grch37",
    "bait_start_grch37", "bait_end_grch37", "prey_chr_grch37",
    "prey_start_grch37", "prey_end_grch37", "bait_chr_grch38",
    "bait_start_grch38", "bait_end_grch38", "prey_chr_grch38",
    "prey_start_grch38", "prey_end_grch38", "chicago_activated",
    "chicago_nonactivated", "called_activated", "called_nonactivated",
    "differential_logfc", "differential_fdr",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval_distance(position_0based: int, start: int, end: int) -> int:
    if start <= position_0based < end:
        return 0
    if position_0based < start:
        return int(max(0, start - (position_0based + 1)))
    return int(max(0, position_0based - end))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(VARIANTS)
    interactions = pd.read_csv(SOURCE, sep=r"\s+", compression="gzip")
    interactions["gene"] = interactions["gene"].astype(str).str.upper()
    interactions["baitChr"] = interactions["baitChr"].astype(str).str.removeprefix("chr")
    interactions["oeChr"] = interactions["oeChr"].astype(str).str.removeprefix("chr")
    interactions["baitEnd"] = interactions["baitStart"] + interactions["baitLength"]
    interactions["oeEnd"] = interactions["oeStart"] + interactions["oeLength"]
    interactions["called_activated"] = interactions["Total_CD4_Activated"] > 5
    interactions["called_nonactivated"] = interactions["Total_CD4_NonActivated"] > 5

    eligible_genes = set(variants["gene"].astype(str).str.upper())
    relevant = interactions[interactions["gene"].isin(eligible_genes)].copy()
    converter = LiftOver(str(CHAIN))
    lifted_rows: list[dict[str, object]] = []
    for contact in relevant.reset_index().itertuples(index=False):
        bait = lift_interval(
            converter, contact.baitChr, int(contact.baitStart), int(contact.baitEnd)
        )
        prey = lift_interval(
            converter, contact.oeChr, int(contact.oeStart), int(contact.oeEnd)
        )
        row: dict[str, object] = {
            "source_row": int(contact.index),
            "gene": str(contact.gene),
            "bait_chr_grch37": str(contact.baitChr),
            "bait_start_grch37": int(contact.baitStart),
            "bait_end_grch37": int(contact.baitEnd),
            "prey_chr_grch37": str(contact.oeChr),
            "prey_start_grch37": int(contact.oeStart),
            "prey_end_grch37": int(contact.oeEnd),
            "bait_liftover_status": bait["status"],
            "prey_liftover_status": prey["status"],
            "bait_first_candidates": bait.get("first_candidates", 0),
            "bait_last_candidates": bait.get("last_candidates", 0),
            "prey_first_candidates": prey.get("first_candidates", 0),
            "prey_last_candidates": prey.get("last_candidates", 0),
            "chicago_activated": float(contact.Total_CD4_Activated),
            "chicago_nonactivated": float(contact.Total_CD4_NonActivated),
            "called_activated": bool(contact.called_activated),
            "called_nonactivated": bool(contact.called_nonactivated),
            "differential_logfc": float(contact.logFC),
            "differential_fdr": float(contact.FDR),
        }
        if bait["status"] == "mapped":
            row.update(
                {
                    "bait_chr_grch38": bait["chromosome"],
                    "bait_start_grch38": bait["start"],
                    "bait_end_grch38": bait["end"],
                    "bait_strand": bait["strand"],
                }
            )
        if prey["status"] == "mapped":
            row.update(
                {
                    "prey_chr_grch38": prey["chromosome"],
                    "prey_start_grch38": prey["start"],
                    "prey_end_grch38": prey["end"],
                    "prey_strand": prey["strand"],
                }
            )
        row["liftover_status"] = (
            "mapped"
            if bait["status"] == "mapped" and prey["status"] == "mapped"
            else "failed"
        )
        lifted_rows.append(row)

    liftover = pd.DataFrame(lifted_rows)
    mapped = liftover[liftover["liftover_status"] == "mapped"].copy()
    called_mapped = mapped[mapped["called_activated"] | mapped["called_nonactivated"]].copy()

    locus_by_gene = (
        variants[["locus", "gene", "chromosome"]]
        .drop_duplicates("gene")
        .set_index("gene")
        .to_dict("index")
    )
    gene_contact_rows: list[dict[str, object]] = []
    for contact in called_mapped.itertuples(index=False):
        meta = locus_by_gene[str(contact.gene)]
        gene_contact_rows.append(
            {
                "locus": meta["locus"],
                "gene": contact.gene,
                "chromosome": str(meta["chromosome"]),
                "source_row": contact.source_row,
                "bait_start_grch38": int(contact.bait_start_grch38),
                "bait_end_grch38": int(contact.bait_end_grch38),
                "prey_start_grch38": int(contact.prey_start_grch38),
                "prey_end_grch38": int(contact.prey_end_grch38),
                "chicago_activated": contact.chicago_activated,
                "chicago_nonactivated": contact.chicago_nonactivated,
                "called_activated": contact.called_activated,
                "called_nonactivated": contact.called_nonactivated,
                "differential_logfc": contact.differential_logfc,
                "differential_fdr": contact.differential_fdr,
            }
        )

    variant_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    for variant in variants.itertuples(index=False):
        position_0based = int(variant.position) - 1
        source_gene = relevant[relevant["gene"] == variant.gene]
        lifted_gene = mapped[
            (mapped["gene"] == variant.gene)
            & (mapped["prey_chr_grch38"].astype(str) == str(variant.chromosome))
        ]
        called_gene = lifted_gene[
            lifted_gene["called_activated"] | lifted_gene["called_nonactivated"]
        ]
        candidates = lifted_gene[
            (lifted_gene["prey_start_grch38"] <= position_0based)
            & (position_0based < lifted_gene["prey_end_grch38"])
        ]
        called = candidates[candidates["called_activated"] | candidates["called_nonactivated"]]
        distances = [
            interval_distance(
                position_0based,
                int(contact.prey_start_grch38),
                int(contact.prey_end_grch38),
            )
            for contact in called_gene.itertuples(index=False)
        ]
        nearest_called_contact_bp = min(distances) if distances else None
        variant_rows.append(
            {
                "locus": variant.locus,
                "gene": variant.gene,
                "variant_id": variant.variant_id,
                "rsids": variant.rsids,
                "posterior_probability": variant.posterior_probability,
                "grch38_position": int(variant.position),
                "grch38_position_0based": position_0based,
                "gene_source_interaction_rows": len(source_gene),
                "gene_lifted_interaction_rows": len(lifted_gene),
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
                    "grch38_position": int(variant.position),
                    "grch38_position_0based": position_0based,
                    "bait_chr_grch37": contact.bait_chr_grch37,
                    "bait_start_grch37": contact.bait_start_grch37,
                    "bait_end_grch37": contact.bait_end_grch37,
                    "prey_chr_grch37": contact.prey_chr_grch37,
                    "prey_start_grch37": contact.prey_start_grch37,
                    "prey_end_grch37": contact.prey_end_grch37,
                    "bait_chr_grch38": contact.bait_chr_grch38,
                    "bait_start_grch38": contact.bait_start_grch38,
                    "bait_end_grch38": contact.bait_end_grch38,
                    "prey_chr_grch38": contact.prey_chr_grch38,
                    "prey_start_grch38": contact.prey_start_grch38,
                    "prey_end_grch38": contact.prey_end_grch38,
                    "chicago_activated": contact.chicago_activated,
                    "chicago_nonactivated": contact.chicago_nonactivated,
                    "called_activated": contact.called_activated,
                    "called_nonactivated": contact.called_nonactivated,
                    "differential_logfc": contact.differential_logfc,
                    "differential_fdr": contact.differential_fdr,
                }
            )

    variant_table = pd.DataFrame(variant_rows)
    contacts = pd.DataFrame(contact_rows, columns=CONTACT_COLUMNS)
    gene_contacts = pd.DataFrame(gene_contact_rows)
    liftover.to_csv(OUTPUT / "pchic_liftover_rows.csv", index=False)
    variant_table.to_csv(OUTPUT / "credible_variant_pchic_tests.csv", index=False)
    contacts.to_csv(OUTPUT / "credible_variant_pchic_contacts.csv", index=False)
    gene_contacts.to_csv(OUTPUT / "target_gene_pchic_contacts.csv", index=False)
    summary = variant_table.groupby(["locus", "gene"], as_index=False).agg(
        credible_variants=("variant_id", "size"),
        variants_with_called_contact=("called_contact", "sum"),
        posterior_probability_with_called_contact=(
            "posterior_probability",
            lambda values: float(
                values[variant_table.loc[values.index, "called_contact"]].sum()
            ),
        ),
        nearest_called_contact_bp=("nearest_called_contact_bp", "min"),
    )
    summary.to_csv(OUTPUT / "pchic_locus_summary.csv", index=False)
    audit = {
        "source": "Burren et al. 2017 Additional file 5 Table S4",
        "doi": "10.1186/s13059-017-1285-0",
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": sha256(SOURCE),
        "source_assembly": "GRCh37",
        "target_assembly": "GRCh38",
        "liftover_method": "UCSC hg19ToHg38 chain via pyliftover",
        "chain_path": str(CHAIN.relative_to(ROOT)).replace("\\", "/"),
        "chain_bytes": CHAIN.stat().st_size,
        "chain_sha256": sha256(CHAIN),
        "liftover_helper_path": "analyses/common/strict_liftover.py",
        "liftover_helper_sha256": sha256(ROOT / "analyses/common/strict_liftover.py"),
        "source_interval_convention": "zero-based half-open",
        "variant_position_convention": "one-based; converted to zero-based before overlap",
        "distance_definition": "number of intervening bases; adjacent intervals have distance zero",
        "interaction_call_rule": "CHiCAGO score greater than 5 in either CD4 state",
        "source_interaction_rows": len(interactions),
        "eligible_gene_source_rows": len(relevant),
        "eligible_gene_lifted_rows": len(mapped),
        "failed_liftover_rows": int((liftover["liftover_status"] != "mapped").sum()),
        "frozen_credible_variants": len(variant_table),
        "called_contact_rows": len(contacts),
    }
    (OUTPUT / "pchic_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
