import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analyses.compositionality.audit_metadata import OUT, ROOT, SOURCE
from analyses.compositionality.screen_assignments import RESULTS, fold, pair_id, summarize


@pytest.fixture(scope="module")
def pairs():
    return pd.read_csv(RESULTS / "candidate_pair_metadata.tsv", sep="\t")


@pytest.fixture(scope="module")
def reports():
    return {r["dataset"]: r for r in json.loads((RESULTS / "assignment_denominators.json").read_text())}


def test_frozen_inputs():
    manifest = json.loads((SOURCE / "analyses/cell_systems_expansion/freeze_manifest.json").read_text())
    verified = json.loads((OUT / "input_verification.json").read_text())
    assert len(verified) == len(manifest["artifacts"]) == 46
    assert all(r["verified"] and r["actual_sha256"] == r["sha256"] for r in verified)
    for row in manifest["artifacts"]:
        assert hashlib.sha256((SOURCE / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_metadata_checksums():
    sources = pd.read_csv(RESULTS / "source_manifest.tsv", sep="\t")
    assert sources.path.is_unique
    for row in sources.itertuples():
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256


def test_frozen_thresholds():
    config = yaml.safe_load((SOURCE / "config/cell_systems_expansion.yaml").read_text())
    gate = config["external_dataset_eligibility"]["confirmatory"]
    assert [gate[k] for k in ["minimum_cells_per_single", "minimum_cells_per_pair", "minimum_control_cells",
                              "minimum_unique_pairs", "minimum_unique_targets", "minimum_overlapping_cd4_targets"]] == [50, 50, 500, 40, 15, 20]
    assert config["compositionality"]["minimum_power"]["pair_model"] == {"eligible_pairs": 100, "unique_targets": 30, "unique_pairs_per_test_fold": 15}


@pytest.mark.parametrize("single,pair,control,expected", [(50, 50, 500, True), (49, 50, 500, False), (50, 49, 500, False), (50, 50, 499, False)])
def test_count_gate_boundaries(single, pair, control, expected):
    labels = pd.Series([[]] * control + [["A"]] * single + [["B"]] * single + [["B", "A"]] * pair)
    result, _ = summarize(labels, "GSE_TEST", "test", {"A", "B"})
    assert bool(result.cell_gate_pass.iloc[0]) is expected


def test_unique_pairs_and_folds(pairs):
    assert pairs.observation_key.is_unique
    assert (pairs.target_a < pairs.target_b).all()
    for row in pairs.itertuples():
        key = "|".join([row.target_a, row.target_b, row.accession])
        assert row.pair_id == key == pair_id(row.target_b, row.target_a, row.accession)
        expected = 1 + int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 5
        assert row.pair_fold == expected == fold(key)
    assert pairs.groupby("pair_id").pair_fold.nunique().max() == 1


def test_cell_gates_and_denominators(pairs, reports):
    expected = pairs.pair_cells.ge(50) & pairs.single_a_cells.ge(50) & pairs.single_b_cells.ge(50) & pairs.control_cells.ge(500)
    pd.testing.assert_series_equal(expected, pairs.cell_gate_pass, check_names=False)
    for name, subset in pairs.groupby("dataset"):
        report = reports[name]
        eligible = subset[subset.cell_gate_pass]
        targets = set(eligible.target_a) | set(eligible.target_b)
        assert len(subset) == report["unique_double_targets"]
        assert len(eligible) == report["cell_eligible_pairs"]
        assert len(targets) == report["eligible_targets"]
        counts = {str(i): int(eligible.pair_fold.eq(i).sum()) for i in range(1, 6)}
        assert counts == report["eligible_pair_fold_counts"]
        assert report["pair_model_power_gate"] == (len(eligible) >= 100 and len(targets) >= 30 and min(counts.values()) >= 15)


def test_cd4_overlap(pairs, reports):
    primary = pd.read_parquet(SOURCE / "analyses/primary/results/transfer_phenotypes.parquet", columns=["target_contrast_gene_name"])
    cd4 = set(primary.target_contrast_gene_name)
    assert len(cd4) == 6105
    assert pairs.target_a.isin(cd4).equals(pairs.target_a_cd4)
    assert pairs.target_b.isin(cd4).equals(pairs.target_b_cd4)
    for name, subset in pairs.groupby("dataset"):
        eligible = subset[subset.cell_gate_pass]
        targets = set(eligible.target_a) | set(eligible.target_b)
        assert len(targets & cd4) == reports[name]["overlapping_cd4_targets"]
        assert int((eligible.target_a_cd4 & eligible.target_b_cd4).sum()) == reports[name]["both_components_cd4_pairs"]


def test_replogle_alias_collapse(pairs, reports):
    aliases = pd.read_csv(RESULTS / "replogle_guide_aliases.tsv", sep="\t")
    for target in ["FDPS", "HUS1"]:
        rows = aliases[aliases.symbol.isin([target, target + "_2"])]
        assert set(rows.symbol) == {target, target + "_2"}
        assert rows.ensembl.nunique() == rows.canonical_symbol.nunique() == 1
    raw = pd.read_csv(OUT / "replogle_exp6_cell_identities.csv.gz")
    raw = raw[raw.good_coverage & raw.number_of_cells.eq(1) & raw.num_guides.eq(2)]
    lookup = dict(zip(aliases.symbol, aliases.canonical_symbol))
    controls = set(aliases.loc[aliases.ensembl.eq("Non-Targeting"), "canonical_symbol"])
    independent = raw.apply(lambda r: tuple(sorted({lookup[r.gene_A], lookup[r.gene_B]} - controls)), axis=1).value_counts()
    observed = pairs[pairs.dataset.eq("Replogle2020_exp6")]
    for row in observed.itertuples():
        assert row.pair_cells == independent[(row.target_a, row.target_b)]
    assert len(raw) == reports["Replogle2020_exp6"]["assignment_rows"] == 37238
    assert reports["Replogle2020_exp6"]["unique_single_targets"] == 41
    assert len(observed) == 37
    assert observed.cell_gate_pass.sum() == 36


def test_norman_independent_pair_counts(pairs, reports):
    raw = pd.read_csv(OUT / "norman_cell_identities.csv.gz")
    raw = raw[raw.good_coverage & raw.number_of_cells.eq(1)]
    independent = raw.guide_identity.map(lambda v: tuple(sorted(set(t for t in v.strip().split("__")[0].split("_") if not t.startswith("NegCtrl"))))).value_counts()
    observed = pairs[pairs.dataset.eq("Norman2019")]
    for row in observed.itertuples():
        assert row.pair_cells == independent[(row.target_a, row.target_b)]
        assert row.single_a_cells == independent[(row.target_a,)]
        assert row.single_b_cells == independent[(row.target_b,)]
    assert len(raw) == 91168
    assert len(observed) == 131
    assert observed.cell_gate_pass.sum() == 127
    assert reports["Norman2019"]["both_components_cd4_pairs"] == 25


def test_inventory_denominator_and_selection():
    catalog = pd.read_csv(OUT / "scperturb_website.csv")
    release = json.loads((OUT / "scperturb_zenodo.json").read_text())
    table = pd.read_csv(RESULTS / "dataset_eligibility.tsv", sep="\t")
    assert len(catalog) == 58 and len(release["files"]) == 54
    assert set(catalog["Full index"]) <= set(table.dataset)
    assert {r["key"].removesuffix(".h5ad") for r in release["files"]} <= set(table.dataset)
    assert table.dataset.is_unique
    assert table.exclusion_reasons.notna().all()
    assert not table.selected_for_outcomes.any()
    assert not table.full_eligibility_pass.any()
    assert not table.provider_checksum_verified_on_full_release.eq(True).any()


def test_stopping_rule_and_no_estimates():
    summary = json.loads((RESULTS / "stopping_rule.json").read_text())
    assert summary["external_expression_matrices_downloaded"] == summary["external_expression_arrays_read"] == summary["models_fitted"] == 0
    assert summary["norman_expression_downloaded"] is False
    assert summary["thresholds_changed"] is False
    decisions = pd.read_csv(RESULTS / "hypothesis_decisions.tsv", sep="\t")
    assert list(decisions.hypothesis) == [f"H{i}" for i in range(1, 7)]
    assert decisions.decision.eq("unavailable").all()
    assert decisions[["effect_size", "ci_low", "ci_high", "p_value", "q_value"]].isna().all().all()
    models = pd.read_csv(RESULTS / "model_availability.tsv", sep="\t")
    assert len(models) == 12 and models.fit_count.sum() == 0


def test_remote_metadata_scope():
    audits = list(OUT.glob("*_audit.json"))
    assert len(audits) == 4
    for path in audits:
        report = json.loads(path.read_text())
        assert report["outcome_arrays_read"] is False
        assert report["bytes_transferred"] < 25000000
        assert report["status"] == "metadata_retrieved"
        metadata = pd.read_csv(OUT / (report["dataset"] + "_obs.tsv.gz"), sep="\t")
        assert len(metadata) == report["cells"]
        assert set(metadata.columns) <= {"perturbation", "perturbation_type", "batch", "ngenes"}
    assert not list(OUT.glob("*.h5ad"))
    assert not list(OUT.glob("*.h5"))
    assert not list(OUT.glob("*.npz"))


def test_authored_text_style():
    paths = list((ROOT / "analyses/compositionality").glob("*.py")) + list((ROOT / "analyses/compositionality").glob("*.md")) + [Path(__file__)]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert chr(8211) not in text and chr(8212) not in text
