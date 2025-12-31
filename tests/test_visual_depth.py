from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/visual_depth/results"


def test_volcano_tables_retain_all_tested_genes_and_fixed_rule():
    expected = {"volcano_gata3_rest.csv": 9752, "volcano_relb_stim48hr.csv": 10282, "volcano_nfat5_rest.csv": 7473}
    for name, denominator in expected.items():
        table = pd.read_csv(RESULTS / name)
        assert int(table.tested.sum()) == denominator
        assert (table.loc[table.significant, "q_value"] <= 0.10).all()
        assert (~table.loc[table.q_value > 0.10, "significant"]).all()


def test_network_and_locus_denominators_are_complete():
    edges = pd.read_csv(RESULTS / "signed_bipartite_edges.csv")
    contact = pd.read_csv(ROOT / "analyses/loci/results/pchic_locus_summary.csv")
    assert len(edges) == 121
    assert set(edges.locus) == {"gata3", "stat3", "ptpn22"}
    assert (contact.variants_with_called_contact == 0).all()
    assert contact.credible_variants.sum() == 50


def test_new_figure_exports_and_sources_are_vector_safe():
    for index in range(17, 21):
        name = f"S{index:02d}"
        directory = ROOT / "figures/supplement" / name
        assert (directory / f"{name}.png").stat().st_size > 10000
        assert (directory / f"{name}.pdf").read_bytes().startswith(b"%PDF-")
        svg = directory / f"{name}.svg"
        tree = ET.parse(svg)
        assert not tree.getroot().findall(".//{http://www.w3.org/2000/svg}image")
        assert len(list((directory / "figure_data").glob("*.tsv"))) == 4
        text = svg.read_text(encoding="utf-8")
        for label in "ABCD":
            assert f">{label}<" in text
