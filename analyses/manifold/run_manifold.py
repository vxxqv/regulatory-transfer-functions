"""Construct UMAP and PHATE response-state manifolds with grouped validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import phate
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor", type=Path, default=Path("analyses/vectors/results/contextual_transfer_tensor.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/manifold/results"))
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = pd.read_parquet(args.tensor)
    wide = tensor.pivot(
        index=["target_contrast", "target_gene", "condition"], columns="module", values="score"
    ).reset_index()
    module_columns = sorted(column for column in wide.columns if isinstance(column, int))
    raw_features = wide[module_columns].to_numpy(dtype=float)
    wide["response_score_norm"] = np.linalg.norm(raw_features, axis=1)
    evaluable = wide["response_score_norm"].to_numpy() > 0
    features = StandardScaler().fit_transform(raw_features[evaluable])
    umap_coordinates = umap.UMAP(
        n_neighbors=30,
        min_dist=0.2,
        n_components=2,
        metric="cosine",
        random_state=args.seed,
        low_memory=True,
        n_jobs=1,
    ).fit_transform(features)
    phate_coordinates = phate.PHATE(
        n_components=2,
        knn=15,
        decay=40,
        t="auto",
        n_landmark=2000,
        random_state=args.seed,
        n_jobs=1,
        verbose=0,
    ).fit_transform(features)
    for name in ["umap_1", "umap_2", "phate_1", "phate_2"]:
        wide[name] = np.nan
    wide.loc[evaluable, ["umap_1", "umap_2"]] = umap_coordinates
    wide.loc[evaluable, ["phate_1", "phate_2"]] = phate_coordinates

    analysis_rows = wide.loc[evaluable].reset_index(names="source_row")
    labels = analysis_rows["condition"].astype("category")
    groups = analysis_rows["target_contrast"].astype(str).to_numpy()
    predicted = np.empty(len(analysis_rows), dtype=object)
    fold_records = []
    splitter = GroupKFold(n_splits=10)
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups), start=1):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed + fold)
        model.fit(features[train], labels.iloc[train])
        predicted[test] = model.predict(features[test])
        fold_records.append(
            {
                "fold": fold,
                "rows": len(test),
                "accuracy": accuracy_score(labels.iloc[test], predicted[test]),
                "balanced_accuracy": balanced_accuracy_score(labels.iloc[test], predicted[test]),
            }
        )
    wide["oof_predicted_condition"] = None
    wide.loc[analysis_rows["source_row"], "oof_predicted_condition"] = predicted
    args.output.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(args.output / "response_manifold.parquet", index=False)
    pd.DataFrame(fold_records).to_csv(args.output / "state_prediction_folds.csv", index=False)
    audit = {
        "rows": len(wide),
        "evaluable_nonzero_rows": int(evaluable.sum()),
        "zero_response_rows": int((~evaluable).sum()),
        "targets": int(wide["target_contrast"].nunique()),
        "components": len(module_columns),
        "oof_state_accuracy": float(accuracy_score(labels, predicted)),
        "oof_state_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "chance_balanced_accuracy": float(1.0 / labels.nunique()),
        "umap_neighbors": 30,
        "phate_neighbors": 15,
        "phate_landmarks": 2000,
    }
    (args.output / "manifold_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
