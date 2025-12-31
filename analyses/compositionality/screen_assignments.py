"""Apply frozen cell-count and overlap gates to assignment metadata."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.compositionality.audit_metadata import ROOT, SOURCE, OUT

RESULTS = ROOT / "analyses/compositionality/results"


def pair_id(a, b, accession):
    return "|".join(sorted([a, b]) + [accession])


def fold(key, n=5):
    return 1 + int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % n


def summarize(labels, accession, name, cd4):
    labels = labels.map(lambda value: tuple(sorted(set(value))))
    counts = labels.value_counts()
    singles = {key[0]: int(n) for key, n in counts.items() if len(key) == 1}
    ntc = int(counts.get((), 0))
    rows = []
    for targets, cells in counts.items():
        if len(targets) != 2:
            continue
        a, b = targets
        reason = []
        if cells < 50:
            reason.append("fewer_than_50_pair_cells")
        if singles.get(a, 0) < 50:
            reason.append("component_a_fewer_than_50_single_cells")
        if singles.get(b, 0) < 50:
            reason.append("component_b_fewer_than_50_single_cells")
        if ntc < 500:
            reason.append("fewer_than_500_control_cells")
        key = pair_id(a, b, accession)
        rows.append({"dataset": name, "accession": accession, "observation_key": name + "|" + key,
                     "target_a": a, "target_b": b,
                     "pair_id": key, "pair_fold": fold(key), "pair_cells": int(cells),
                     "single_a_cells": singles.get(a, 0), "single_b_cells": singles.get(b, 0),
                     "control_cells": ntc, "target_a_cd4": a in cd4, "target_b_cd4": b in cd4,
                     "cell_gate_pass": not reason, "exclusion_reason": ";".join(reason)})
    frame = pd.DataFrame(rows).sort_values("pair_id") if rows else pd.DataFrame()
    eligible = frame[frame.cell_gate_pass] if len(frame) else frame
    targets = set(eligible.target_a) | set(eligible.target_b) if len(eligible) else set()
    fold_counts = {str(i): int((eligible.pair_fold == i).sum()) if len(eligible) else 0 for i in range(1, 6)}
    report = {"dataset": name, "accession": accession, "assignment_rows": len(labels),
              "controls": ntc, "unique_single_targets": len(singles), "unique_double_targets": len(frame),
              "cell_eligible_pairs": len(eligible), "eligible_targets": len(targets),
              "overlapping_cd4_targets": len(targets & cd4),
              "both_components_cd4_pairs": int((eligible.target_a_cd4 & eligible.target_b_cd4).sum()) if len(eligible) else 0,
              "higher_order_cells": int(sum(n for key, n in counts.items() if len(key) > 2)),
              "pair_count_gate": len(eligible) >= 40, "target_count_gate": len(targets) >= 15,
              "cd4_overlap_gate": len(targets & cd4) >= 20,
              "pair_model_power_gate": len(eligible) >= 100 and len(targets) >= 30 and min(fold_counts.values()) >= 15,
              "eligible_pair_fold_counts": fold_counts,
              "targets": sorted(targets)}
    return frame, report


def extra_assignments(cd4):
    outputs, reports = [], []
    data = pd.read_csv(OUT / "wessels_metadata.tsv.gz", sep="\t")
    labels = data.GenePair.map(lambda value: [t for t in value.split("_") if t != "NT"])
    table, report = summarize(labels, "GSE213957", "WesselsSatija2023", cd4)
    report["intervention"] = "Cas13"
    outputs.append(table)
    reports.append(report)
    data = pd.read_csv(OUT / "yao_kd_perturbations.tsv.gz", sep="\t", index_col=0)
    targets = data.drop(index=["non-targeting", "safe-targeting"])
    identifiers = targets.index.to_numpy()
    matrix = targets.to_numpy() > 0
    keep = (data.loc["safe-targeting"].to_numpy() == 0) & ((matrix.sum(axis=0) > 0) | (data.loc["non-targeting"].to_numpy() > 0))
    labels = pd.Series([identifiers[matrix[:, col]].tolist() for col in np.flatnonzero(keep)])
    table, report = summarize(labels, "GSE221321", "YaoCleary2023_KD_guide_pooled", cd4)
    report["input_cells"] = data.shape[1]
    report["excluded_safe_targeting_or_unassigned"] = int((~keep).sum())
    outputs.append(table)
    reports.append(report)
    data = pd.read_csv(OUT / "pacalin_metadata.csv.gz")
    for intervention, suffix in [("CRISPRi", "i"), ("CRISPRa", "a")]:
        def compatible(label):
            return label == "Non-Targeting" or all(t.endswith("-" + suffix) and "-e" not in t for t in label.split("|"))
        selected = data[data.guide_group2.map(compatible)]
        labels = selected.guide_group2.map(lambda v: [] if v == "Non-Targeting" else [t[:-2] for t in v.split("|")])
        table, report = summarize(labels, "GSE220974", "Pacalin2024_" + intervention, cd4)
        report["intervention"] = intervention
        report["excluded_incompatible_intervention_or_enhancer_cells"] = len(data) - len(selected)
        outputs.append(table)
        reports.append(report)
    data = pd.read_csv(OUT / "gasperini_lowmoi_pheno.tsv.gz", sep=r"\s+", header=None, usecols=[0, 1, 4, 5, 10])
    labels = []
    for label in data[4].fillna("unassigned").astype(str):
        genes = re.findall(r"([A-Za-z0-9.-]+)_TSS", label)
        remainder = re.sub(r"[A-Za-z0-9.-]+_TSS", "", label)
        remainder = re.sub(r"scrambled_[0-9]+", "", remainder).strip("_")
        if not remainder:
            labels.append(genes)
    table, report = summarize(pd.Series(labels), "GSE120861", "Gasperini2019_lowMOI_TSS_only", cd4)
    report["input_cells"] = len(data)
    report["excluded_enhancer_intergenic_or_unassigned_cells"] = len(data) - len(labels)
    outputs.append(table)
    reports.append(report)
    return outputs, reports


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = pd.read_parquet(SOURCE / "analyses/primary/results/transfer_phenotypes.parquet",
                              columns=["target_contrast_gene_name"])
    cd4 = set(primary["target_contrast_gene_name"])
    outputs, reports = [], []
    for name, accession in [("TianKampmann2019_iPSC", "GSE124703"),
                            ("TianKampmann2019_day7neuron", "GSE124703"),
                            ("SunshineHein2023", "GSE208240")]:
        data = pd.read_csv(OUT / (name + "_obs.tsv.gz"), sep="\t")
        keep = data.perturbation.notna() & ~data.perturbation.isin(["unassigned", "None"])
        labels = data.loc[keep, "perturbation"].map(lambda x: [t for t in x.split("_") if t != "control"])
        table, report = summarize(labels, accession, name, cd4)
        outputs.append(table)
        reports.append(report)
    data = pd.read_csv(OUT / "replogle_exp6_cell_identities.csv.gz")
    keep = data.good_coverage & data.number_of_cells.eq(1) & data.num_guides.eq(2)
    data = data.loc[keep].copy()
    mapping = pd.concat([data[["gene_A", "target_A"]].rename(columns={"gene_A": "symbol", "target_A": "ensembl"}),
                         data[["gene_B", "target_B"]].rename(columns={"gene_B": "symbol", "target_B": "ensembl"})]).drop_duplicates()
    mapping["canonical_symbol"] = mapping.symbol.str.replace(r"_2$", "", regex=True)
    assert mapping[mapping.ensembl != "Non-Targeting"].groupby("ensembl").canonical_symbol.nunique().max() == 1
    mapping.to_csv(RESULTS / "replogle_guide_aliases.tsv", sep="\t", index=False)
    labels = data.apply(lambda r: [re.sub(r"_2$", "", str(r[col]).strip()) for col in ["gene_A", "gene_B"]
                                  if not str(r[col]).strip().startswith("NegCtrl")], axis=1)
    table, report = summarize(labels, "GSE146194", "Replogle2020_exp6", cd4)
    outputs.append(table)
    reports.append(report)
    data = pd.read_csv(OUT / "norman_cell_identities.csv.gz")
    keep = data.good_coverage & data.number_of_cells.eq(1)
    data = data.loc[keep]
    def norman_targets(label):
        pieces = label.strip().split("__")
        assert len(pieces) == 2 and (pieces[0] == pieces[1] or pieces[0] == re.sub(r"_[12]$", "", pieces[1]))
        return [part for part in pieces[0].split("_") if not part.startswith("NegCtrl")]
    labels = data.guide_identity.map(norman_targets)
    table, report = summarize(labels, "GSE133344", "Norman2019", cd4)
    outputs.append(table)
    reports.append(report)
    extra_tables, extra_reports = extra_assignments(cd4)
    outputs.extend(extra_tables)
    reports.extend(extra_reports)
    pd.concat(outputs, ignore_index=True).to_csv(RESULTS / "candidate_pair_metadata.tsv", sep="\t", index=False)
    (RESULTS / "assignment_denominators.json").write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run()
