"""Update finalized molecular evidence and add target-controlled estimates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

from freeze import sha256
from run_analysis import ROOT, OUT, bh, cluster_fit, molecular_support, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    spec_path = ROOT / "analyses/guide_concordance/target_control_spec.json"
    spec = json.loads(spec_path.read_text())
    edge_path = args.source_root / "analyses/molecular_cascade/results/edge_evidence_matrix.parquet"
    assert sha256(edge_path) == spec["final_molecular_sha256"]
    cfg = yaml.safe_load((ROOT / "config/guide_concordance.yaml").read_text())
    rng = np.random.default_rng(spec["seed"])
    frozen_outputs = ["state_summaries.csv", "denominators.csv", "matched_state_comparisons.csv", "matched_state_nulls.parquet", "measurement_error_sensitivity.csv", "availability_gates.csv"]
    before_hashes = {name: sha256(OUT / name) for name in frozen_outputs}
    original_tests = pd.read_csv(OUT / "hypothesis_tests.csv", float_precision="round_trip")
    original_null = pd.read_parquet(OUT / "permutation_nulls.parquet")
    original_predictions = pd.read_parquet(OUT / "target_held_out_predictions.parquet")
    original_influence = pd.read_csv(OUT / "target_influence.csv")
    rows = pd.read_parquet(OUT / "all_target_states.parquet")
    molecular_columns = [c for c in rows.columns if c.startswith("molecular_") or c == "evidence_response_edges"]
    rows = rows.drop(columns=molecular_columns)
    rows, available = molecular_support(args.source_root, rows)
    assert available
    write(rows, "all_target_states.parquet")
    eligible = rows.loc[rows.eligible_pair].copy()
    eligible["significance_status"] = np.where(eligible.significant_union_eligible, "supported_union_size", "small_or_empty_union")
    write(eligible, "eligible_pairs.parquet")
    kept = original_tests.loc[original_tests.analysis.eq("paired_cis_count")].copy()
    assert len(kept) == 1
    kept["estimator"] = "pooled"
    kept["evaluation_scale"] = "absolute outcome"
    records = kept.to_dict("records")
    nulls = [original_null.loc[original_null.analysis.eq("paired_cis_count")]]
    predictions = [original_predictions.loc[original_predictions.analysis.eq("paired_cis_count")].assign(estimator="pooled", evaluation_scale="absolute outcome")]
    influences = [original_influence.loc[original_influence.analysis.eq("paired_cis_count")]]
    qualities = cfg["quality_covariates"] + ["log1p_proximal_expression", "log1p_response_degree", "cis_magnitude"]
    for predictor in ["molecular_support_fraction", "molecular_any_fraction"]:
        for outcome in ["correlation_all", "correlation_signif", "frac_sign_agreement_signif_union"]:
            subset = eligible.loc[eligible.in_primary_study & eligible.quality_complete].copy()
            if outcome != "correlation_all":
                subset = subset.loc[subset.significant_union_eligible]
            for target_control in [False, True]:
                name = f"{predictor}:{outcome}" + (":target_controlled" if target_control else "")
                result, null, pred, influence = cluster_fit(subset, predictor, outcome, qualities, cfg, rng, name, target_control=target_control)
                records.append(result)
                nulls.append(null)
                predictions.append(pred)
                influences.append(influence)
    covars = ["delta_gc_fraction", "delta_log1p_tss_distance", "delta_library_flag", "delta_bidirectional_promoter", "delta_secondary_alignment", "delta_log1p_cells"]
    result, null, pred, influence = cluster_fit(eligible, "cis_difference", "count_difference", covars, cfg, rng, "paired_cis_count:target_controlled", permutation="guide_swap", target_control=True)
    records.append(result)
    nulls.append(null)
    predictions.append(pred)
    influences.append(influence)
    tests = pd.DataFrame(records)
    tests["q_value"] = bh(tests.permutation_p)
    tests["decision"] = np.where(tests.q_value.lt(cfg["fdr"]) & tests.ci_low.gt(0), "supported_positive", np.where(tests.q_value.lt(cfg["fdr"]) & tests.ci_high.lt(0), "supported_negative", "unresolved"))
    tests.loc[tests.status.ne("estimated"), "decision"] = tests.loc[tests.status.ne("estimated"), "status"]
    write(tests, "hypothesis_tests.csv")
    write(pd.concat(nulls, ignore_index=True), "permutation_nulls.parquet")
    pred = pd.concat(predictions, ignore_index=True)
    write(pred, "target_held_out_predictions.parquet")
    write(pd.concat(influences, ignore_index=True), "target_influence.csv")
    calibration = []
    for (analysis, model), data in pred.groupby(["analysis", "model"]):
        slope, intercept = np.polyfit(data.prediction, data.observed, 1)
        calibration.append({"analysis": analysis, "model": model, "n_rows": len(data), "r2": r2_score(data.observed, data.prediction), "mae": mean_absolute_error(data.observed, data.prediction), "calibration_slope": slope, "calibration_intercept": intercept, "evaluation_scale": data.evaluation_scale.iloc[0]})
    write(pd.DataFrame(calibration), "calibration.csv")
    manifest_path = ROOT / "analyses/guide_concordance/freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replaced = []
    for record in manifest["inputs"]:
        if record["path"].startswith("analyses/molecular_cascade/results/"):
            replaced.append(record.copy())
            record["sha256"] = sha256(args.source_root / record["path"])
    manifest["molecular_update"] = {"source_commit": spec["source_commit"], "reason": "Final molecular-cascade evidence replaces preliminary matrix after independent audit", "superseded_inputs": replaced, "target_control_spec_sha256": sha256(spec_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path = OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["tests"] = tests[["analysis", "decision"]].to_dict("records")
    audit["final_molecular_sha256"] = spec["final_molecular_sha256"]
    audit["primary_estimator"] = "target_controlled"
    audit["target_control_spec_sha256"] = sha256(spec_path)
    audit["unchanged_output_hashes"] = before_hashes
    for name, expected in before_hashes.items():
        assert sha256(OUT / name) == expected, name
    pooled = tests.loc[tests.analysis.eq("paired_cis_count")].iloc[0]
    old_pooled = original_tests.loc[original_tests.analysis.eq("paired_cis_count")].iloc[0]
    for field in ["coefficient", "ci_low", "ci_high", "permutation_p"]:
        assert pooled[field] == old_pooled[field]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(tests[["analysis", "n_targets", "varying_targets", "coefficient", "ci_low", "ci_high", "q_value", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
