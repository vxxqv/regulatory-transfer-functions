from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_verified_manifest_rows_have_checksums() -> None:
    manifest = pd.read_csv(ROOT / "data_manifest" / "sources.tsv", sep="\t", dtype=str).fillna("")
    assert manifest["source_id"].is_unique
    verified = manifest["status"] == "verified"
    assert manifest.loc[verified, "sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (manifest.loc[verified, "url"].str.startswith("https://")).all()


def test_figure_specification_has_every_planned_panel() -> None:
    specification = pd.read_csv(ROOT / "docs" / "figure-specification.tsv", sep="\t")
    expected = {1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 6, 7: 5}
    observed = specification.groupby("figure")["panel"].nunique().to_dict()
    assert observed == expected
    assert not specification.duplicated(["figure", "panel"]).any()


def test_manuscript_uses_final_figure_architecture() -> None:
    manuscript = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")
    for number in range(1, 8):
        assert f"### Figure {number}." in manuscript
    assert "Corresponding author details to be supplied" not in manuscript
    assert "A permanent archive DOI will be added" not in manuscript


def test_seed_and_state_dictionary_are_frozen() -> None:
    analysis = yaml.safe_load((ROOT / "config" / "analysis.yaml").read_text(encoding="utf-8"))
    colors = yaml.safe_load((ROOT / "config" / "colors.yaml").read_text(encoding="utf-8"))
    assert isinstance(analysis["study"]["seed"], int)
    assert set(colors["states"]) == set(analysis["primary_resource"]["conditions"])
