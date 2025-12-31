"""Prepare compact eQTLGen cis-to-trans validation records from frozen releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMNS = [
    "Pvalue",
    "SNP",
    "SNPChr",
    "SNPPos",
    "AssessedAllele",
    "OtherAllele",
    "Zscore",
    "Gene",
    "GeneSymbol",
    "NrCohorts",
    "NrSamples",
    "FDR",
    "BonferroniP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cis", type=Path, required=True)
    parser.add_argument("--trans", type=Path, required=True)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/interim/eqtlgen"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trans = pd.read_csv(args.trans, sep="\t", usecols=COLUMNS)
    trans_snps = set(trans["SNP"].astype(str))
    targets = pd.read_csv(args.target_summary, usecols=["target_contrast"])
    target_ids = set(targets["target_contrast"].astype(str))

    candidates: list[pd.DataFrame] = []
    source_rows = 0
    for chunk in pd.read_csv(args.cis, sep="\t", usecols=COLUMNS, chunksize=args.chunk_size):
        source_rows += len(chunk)
        keep = chunk["SNP"].astype(str).isin(trans_snps) & chunk["Gene"].astype(str).isin(target_ids)
        if keep.any():
            candidates.append(chunk.loc[keep].copy())
    if not candidates:
        raise ValueError("No cis mediators overlap the profiled perturbation targets")
    cis = pd.concat(candidates, ignore_index=True)
    cis = cis.sort_values(["SNP", "Pvalue", "Gene"]).reset_index(drop=True)
    cis["candidate_rank"] = cis.groupby("SNP").cumcount() + 1
    strongest = cis.loc[cis["candidate_rank"] == 1].copy()
    retained_trans = trans[trans["SNP"].isin(strongest["SNP"])].copy()

    cis.to_parquet(args.output / "cis_mediator_candidates.parquet", index=False)
    strongest.to_parquet(args.output / "cis_mediators.parquet", index=False)
    retained_trans.to_parquet(args.output / "trans_associations.parquet", index=False)
    audit = {
        "cis_source_rows": int(source_rows),
        "trans_source_rows": int(len(trans)),
        "trans_source_snps": int(trans["SNP"].nunique()),
        "profiled_target_ids": int(len(target_ids)),
        "cis_candidate_rows": int(len(cis)),
        "cis_candidate_genes": int(cis["Gene"].nunique()),
        "retained_snps": int(strongest["SNP"].nunique()),
        "retained_trans_rows": int(len(retained_trans)),
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
