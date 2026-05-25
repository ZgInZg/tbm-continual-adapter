import numpy as np
import pandas as pd

from config import INPUT_SPACES, LAMBDA_GRID, SEEDS, TARGETS
from shared_utils import (
    anchor_mask,
    choose_subset_adapter,
    ensure_output_dir,
    fit_from_data,
    fit_initial_model,
    load_shared_domains,
    make_design,
    metrics,
    split_initial,
)


FRACTIONS = [0.10, 0.20, 0.33, 0.50]
METHODS = ["source_only", "full_rehearsal", "generic_adapter", "graph_adapter", "random_adapter"]


def evaluate(model, data, columns):
    return metrics(model, data, columns)


def main():
    out = ensure_output_dir("05_lowshot_s3_adaptation")
    domains = load_shared_domains()
    rows = []

    for input_space, variables in INPUT_SPACES.items():
        for target in TARGETS:
            designs = {}
            columns = None
            for domain, data in domains.items():
                designs[domain], columns = make_design(data, target, variables)
            penalties = anchor_mask(target, columns, include_pressure="pressure_state" in variables)
            graph_columns = [column for column, weight in zip(columns, penalties) if weight == 1.0]

            for source in ["S1", "S2"]:
                train, _, source_test = split_initial(designs[source])
                anchor = fit_initial_model(train, columns)
                source_reference = metrics(anchor, source_test, columns)[0]

                for fraction in FRACTIONS:
                    cut = max(10, int(round(len(designs["S3"]) * fraction)))
                    observed = designs["S3"].iloc[:cut].copy()
                    target_test = designs["S3"].iloc[cut:].copy()
                    for seed in SEEDS:
                        for method in METHODS:
                            if method == "source_only":
                                model = anchor
                                parameters = 0
                                stored_rows = 0
                            elif method == "full_rehearsal":
                                model = fit_from_data(anchor, pd.concat([train, observed], ignore_index=True), columns)
                                parameters = 0
                                stored_rows = len(train)
                            else:
                                if method == "generic_adapter":
                                    selected = columns
                                elif method == "graph_adapter":
                                    selected = graph_columns
                                else:
                                    rng = np.random.default_rng(seed)
                                    selected = rng.choice(columns, size=len(graph_columns), replace=False).tolist()
                                model, _ = choose_subset_adapter(anchor, observed, columns, selected, LAMBDA_GRID)
                                parameters = len(selected) + 1
                                stored_rows = 0

                            current_r2, current_rmse = evaluate(model, target_test, columns)
                            if method.endswith("adapter"):
                                source_retention = source_reference
                                forgetting = 0.0
                            else:
                                source_retention = evaluate(model, source_test, columns)[0]
                                forgetting = source_reference - source_retention

                            rows.append(
                                {
                                    "input_space": input_space,
                                    "target": target,
                                    "source": source,
                                    "observed_fraction": fraction,
                                    "observed_rows": len(observed),
                                    "method": method,
                                    "seed": seed,
                                    "graph_channels": len(graph_columns),
                                    "adapter_parameters": parameters,
                                    "stored_rows": stored_rows,
                                    "target_r2": current_r2,
                                    "target_rmse": current_rmse,
                                    "forgetting": forgetting,
                                    "balanced_score": current_r2 - 0.2 * max(0.0, forgetting),
                                }
                            )

    results = pd.DataFrame(rows)
    results.to_csv(out / "all_results.csv", index=False)
    summary = results.groupby(["input_space", "observed_fraction", "target", "method"], as_index=False).agg(
        observed_rows=("observed_rows", "mean"),
        adapter_parameters=("adapter_parameters", "mean"),
        stored_rows=("stored_rows", "mean"),
        target_r2=("target_r2", "mean"),
        target_r2_std=("target_r2", "std"),
        forgetting=("forgetting", "mean"),
        balanced_score=("balanced_score", "mean"),
    )
    summary["rank"] = summary.groupby(["input_space", "observed_fraction", "target"])["balanced_score"].rank(
        ascending=False, method="min"
    )
    by_fraction = summary.groupby(["input_space", "observed_fraction", "method"], as_index=False).agg(
        mean_rank=("rank", "mean"),
        wins=("rank", lambda values: int((values == 1).sum())),
        observed_rows=("observed_rows", "mean"),
        adapter_parameters=("adapter_parameters", "mean"),
        stored_rows=("stored_rows", "mean"),
        target_r2=("target_r2", "mean"),
        forgetting=("forgetting", "mean"),
        balanced_score=("balanced_score", "mean"),
    )
    summary.to_csv(out / "summary_by_task.csv", index=False)
    by_fraction.to_csv(out / "summary_by_fraction.csv", index=False)
    with open(out / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 05 Low-shot S3 Adaptation\n\n")
        handle.write(by_fraction.round(4).to_markdown(index=False))
        handle.write("\n")


if __name__ == "__main__":
    main()
