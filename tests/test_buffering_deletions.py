from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/primary/results"
STATES = ["Rest", "Stim8hr", "Stim48hr"]


def test_buffering_deletion_point_estimates_reproduce():
    frame = pd.read_parquet(RESULTS / "transfer_phenotypes.parquet")
    wide = frame.pivot(index="target_contrast", columns="culture_condition", values="transfer_residual")
    wide = wide.reindex(columns=STATES).dropna()
    values = wide.to_numpy(float)
    estimates = []
    counts = []
    outcomes = []
    for heldout in range(3):
        training = np.delete(values, heldout, axis=1).mean(axis=1)
        selected = training <= np.quantile(training, 0.10)
        local = values[selected, heldout]
        estimates.append(local.mean())
        counts.append(int(selected.sum()))
        outcomes.append(local)
    estimates.append(np.concatenate(outcomes).mean())
    counts.append(sum(counts))
    summary = pd.read_csv(RESULTS / "buffering_deletion_summary.csv")
    assert len(wide) == 4399
    assert np.allclose(summary["estimate"], estimates, atol=1e-12)
    assert summary["selected_rows"].tolist() == counts
    assert summary["decision"].eq("supported").all()


def test_buffering_deletion_resampling_and_influence_are_complete():
    summary = pd.read_csv(RESULTS / "buffering_deletion_summary.csv")
    null = pd.read_parquet(RESULTS / "buffering_deletion_null.parquet")
    influence = pd.read_csv(RESULTS / "buffering_leave_one_target.csv")
    unavailable = pd.read_csv(RESULTS / "buffering_unavailable_deletions.csv")
    assert len(null) == 8000
    assert null.groupby("heldout_state").size().eq(2000).all()
    assert len(influence) == 4399
    assert influence["target"].is_unique
    assert influence["change_from_full"].abs().max() < 0.005
    assert summary["ci_high"].lt(0).all()
    assert unavailable.set_index("analysis").loc["leave_one_guide", "status"] == "unavailable"
    assert unavailable.set_index("analysis").loc["leave_one_donor", "status"] == "unavailable"

