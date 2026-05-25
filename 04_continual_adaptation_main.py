import numpy as np
import pandas as pd

from config import INPUT_SPACES, LAMBDA_GRID, SEEDS, TARGETS
from shared_utils import (
    adapt_global_model,
    anchor_mask,
    choose_adapter,
    ensure_output_dir,
    fit_initial_model,
    load_shared_domains,
    make_design,
    metrics,
    split_adaptation,
    split_initial,
)


METHODS = [
    "source_only",
    "global_target_only",
    "global_full_rehearsal",
    "global_mechanism_anchor",
    "generic_adapter_bank",
    "mechanism_adapter_bank",
]


def global_adapt(method, reference, observed, memory, prior_validation, columns, penalties, seed):
    mapped = {
        "source_only": "source_only",
        "global_target_only": "target_only",
        "global_full_rehearsal": "full_rehearsal",
        "global_mechanism_anchor": "mechanism_anchor",
    }[method]
    return adapt_global_model(mapped, reference, observed, memory, prior_validation, columns, penalties, seed, LAMBDA_GRID)


def main():
    out = ensure_output_dir("04_continual_adaptation_main")
    domains = load_shared_domains()
    paths = [("S1", "S2", "S3"), ("S2", "S1", "S3")]
    rows = []

    for space, variables in INPUT_SPACES.items():
        for target in TARGETS:
            designs = {}
            columns = None
            for name, data in domains.items():
                designs[name], columns = make_design(data, target, variables)
            mechanism_penalties = anchor_mask(target, columns, include_pressure="pressure_state" in variables)
            generic_penalties = np.ones(len(columns))

            for initial_domain, second_domain, third_domain in paths:
                initial_train, initial_validation, initial_test = split_initial(designs[initial_domain])
                second_observed, second_test = split_adaptation(designs[second_domain])
                third_observed, third_test = split_adaptation(designs[third_domain])

                for seed in SEEDS:
                    anchor = fit_initial_model(initial_train, columns)
                    initial_r2 = metrics(anchor, initial_test, columns)[0]
                    for method in METHODS:
                        if method in {"generic_adapter_bank", "mechanism_adapter_bank"}:
                            penalties = generic_penalties if method == "generic_adapter_bank" else mechanism_penalties
                            second_model, lambda2 = choose_adapter(anchor, second_observed, columns, penalties, LAMBDA_GRID)
                            second_r2, second_rmse = metrics(second_model, second_test, columns)
                            rows.append(
                                {
                                    "input_space": space,
                                    "target": target,
                                    "path": f"{initial_domain}->{second_domain}->{third_domain}",
                                    "stage": f"adapt_to_{second_domain}",
                                    "method": method,
                                    "seed": seed,
                                    "lambda": lambda2,
                                    "stored_rows": 0,
                                    "adapter_parameters": len(columns) + 1,
                                    "current_r2": second_r2,
                                    "current_rmse": second_rmse,
                                    "prior_retention_r2": initial_r2,
                                    "forgetting": 0.0,
                                    "balanced_score": second_r2,
                                }
                            )
                            third_model, lambda3 = choose_adapter(anchor, third_observed, columns, penalties, LAMBDA_GRID)
                            third_r2, third_rmse = metrics(third_model, third_test, columns)
                            rows.append(
                                {
                                    "input_space": space,
                                    "target": target,
                                    "path": f"{initial_domain}->{second_domain}->{third_domain}",
                                    "stage": "adapt_to_S3",
                                    "method": method,
                                    "seed": seed,
                                    "lambda": lambda3,
                                    "stored_rows": 0,
                                    "adapter_parameters": 2 * (len(columns) + 1),
                                    "current_r2": third_r2,
                                    "current_rmse": third_rmse,
                                    "prior_retention_r2": np.mean([initial_r2, second_r2]),
                                    "forgetting": 0.0,
                                    "balanced_score": third_r2,
                                }
                            )
                            continue

                        model2, lambda2, rows2 = global_adapt(
                            method,
                            anchor,
                            second_observed,
                            initial_train,
                            [initial_validation],
                            columns,
                            mechanism_penalties,
                            seed,
                        )
                        second_r2, second_rmse = metrics(model2, second_test, columns)
                        first_after2 = metrics(model2, initial_test, columns)[0]
                        forgetting2 = initial_r2 - first_after2
                        rows.append(
                            {
                                "input_space": space,
                                "target": target,
                                "path": f"{initial_domain}->{second_domain}->{third_domain}",
                                "stage": f"adapt_to_{second_domain}",
                                "method": method,
                                "seed": seed,
                                "lambda": lambda2,
                                "stored_rows": rows2,
                                "adapter_parameters": 0,
                                "current_r2": second_r2,
                                "current_rmse": second_rmse,
                                "prior_retention_r2": first_after2,
                                "forgetting": forgetting2,
                                "balanced_score": second_r2 - 0.2 * max(0.0, forgetting2),
                            }
                        )

                        memory = pd.concat([initial_train, second_observed], ignore_index=True)
                        model3, lambda3, rows3 = global_adapt(
                            method,
                            model2,
                            third_observed,
                            memory,
                            [initial_validation, second_observed],
                            columns,
                            mechanism_penalties,
                            seed + 100,
                        )
                        third_r2, third_rmse = metrics(model3, third_test, columns)
                        first_after3 = metrics(model3, initial_test, columns)[0]
                        second_after3 = metrics(model3, second_test, columns)[0]
                        forgetting3 = np.mean([initial_r2 - first_after3, second_r2 - second_after3])
                        rows.append(
                            {
                                "input_space": space,
                                "target": target,
                                "path": f"{initial_domain}->{second_domain}->{third_domain}",
                                "stage": "adapt_to_S3",
                                "method": method,
                                "seed": seed,
                                "lambda": lambda3,
                                "stored_rows": rows3,
                                "adapter_parameters": 0,
                                "current_r2": third_r2,
                                "current_rmse": third_rmse,
                                "prior_retention_r2": np.mean([first_after3, second_after3]),
                                "forgetting": forgetting3,
                                "balanced_score": third_r2 - 0.2 * max(0.0, forgetting3),
                            }
                        )

    results = pd.DataFrame(rows)
    results.to_csv(out / "all_results.csv", index=False)
    summary = results.groupby(["input_space", "target", "stage", "method"], as_index=False).agg(
        stored_rows=("stored_rows", "mean"),
        adapter_parameters=("adapter_parameters", "mean"),
        current_r2=("current_r2", "mean"),
        current_r2_std=("current_r2", "std"),
        prior_retention_r2=("prior_retention_r2", "mean"),
        forgetting=("forgetting", "mean"),
        balanced_score=("balanced_score", "mean"),
    )
    summary["rank"] = summary.groupby(["input_space", "target", "stage"])["balanced_score"].rank(
        ascending=False, method="min"
    )
    ranking = summary.groupby(["input_space", "method"], as_index=False).agg(
        mean_rank=("rank", "mean"),
        wins=("rank", lambda values: int((values == 1).sum())),
        stored_rows=("stored_rows", "mean"),
        adapter_parameters=("adapter_parameters", "mean"),
        current_r2=("current_r2", "mean"),
        forgetting=("forgetting", "mean"),
        balanced_score=("balanced_score", "mean"),
    )
    summary.to_csv(out / "summary_by_task.csv", index=False)
    ranking.to_csv(out / "overall_ranking.csv", index=False)
    with open(out / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 04 Continual Adaptation Main Experiment\n\n")
        handle.write("## Overall ranking\n\n")
        handle.write(ranking.round(4).to_markdown(index=False))
        handle.write("\n\n## Detailed summary\n\n")
        handle.write(summary.round(4).to_markdown(index=False))
        handle.write("\n")


if __name__ == "__main__":
    main()
