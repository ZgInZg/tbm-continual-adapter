from pathlib import Path

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from config import COMMON_FEATURES, COMMON_DISPLAY, LAYER_ORDER
from shared_utils import continuous_segments, ensure_output_dir, load_shared_domains


TAU_MAX = 2
PC_ALPHA = 0.01
EDGE_ALPHA = 0.01
GAMMA = 0.60
N_SUBSAMPLES = 30
WINDOW_FRACTION = 0.80
MIN_SEGMENT_LENGTH = 20


def sample_windows(segments, rng):
    output = {}
    for index, segment in enumerate(segments):
        size = max(MIN_SEGMENT_LENGTH, int(len(segment) * WINDOW_FRACTION))
        size = min(size, len(segment))
        start = 0 if size == len(segment) else int(rng.integers(0, len(segment) - size + 1))
        output[index] = segment.iloc[start : start + size].to_numpy()
    return output


def discover_pcmci(segments, condition, rng):
    records = {}
    for _ in range(N_SUBSAMPLES):
        data = pp.DataFrame(
            sample_windows(segments, rng),
            analysis_mode="multiple",
            var_names=COMMON_FEATURES,
        )
        result = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0).run_pcmci(
            tau_max=TAU_MAX, pc_alpha=PC_ALPHA
        )
        seen = set()
        for i, cause in enumerate(COMMON_FEATURES):
            for j, effect in enumerate(COMMON_FEATURES):
                for lag in range(TAU_MAX + 1):
                    if lag == 0 and i == j:
                        continue
                    if result["p_matrix"][i, j, lag] <= EDGE_ALPHA:
                        edge = (cause, lag, effect)
                        seen.add(edge)
                        item = records.setdefault(
                            edge,
                            {"cause": cause, "lag": lag, "effect": effect, "count": 0},
                        )
                        item["count"] += 0
        for edge in seen:
            records[edge]["count"] += 1
    table = pd.DataFrame(
        [
            {
                "condition": condition,
                "cause": item["cause"],
                "lag": item["lag"],
                "effect": item["effect"],
                "frequency": item["count"] / N_SUBSAMPLES,
            }
            for item in records.values()
        ]
    )
    stable = table[table["frequency"] >= GAMMA].copy()
    stable["keep_after_prior"] = stable.apply(
        lambda row: int(row["lag"]) > 0 or LAYER_ORDER[row["cause"]] <= LAYER_ORDER[row["effect"]],
        axis=1,
    )
    return table, stable[stable["keep_after_prior"]].copy()


def discover_pcmciplus(segments, condition, rng):
    directed = {}
    adjacencies = {}
    for _ in range(N_SUBSAMPLES):
        data = pp.DataFrame(
            sample_windows(segments, rng),
            analysis_mode="multiple",
            var_names=COMMON_FEATURES,
        )
        result = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0).run_pcmciplus(
            tau_min=0, tau_max=TAU_MAX, pc_alpha=PC_ALPHA
        )
        seen_directed = set()
        seen_adj = set()
        for i, cause in enumerate(COMMON_FEATURES):
            for j, effect in enumerate(COMMON_FEATURES):
                for lag in range(TAU_MAX + 1):
                    if lag == 0 and i == j:
                        continue
                    orientation = str(result["graph"][i, j, lag])
                    if lag == 0 and i < j and orientation not in {"", "None"}:
                        seen_adj.add(tuple(sorted((cause, effect))))
                    if orientation in {"-->", "o->"} and result["p_matrix"][i, j, lag] <= EDGE_ALPHA:
                        seen_directed.add((cause, lag, effect))
        for edge in seen_directed:
            directed[edge] = directed.get(edge, 0) + 1
        for edge in seen_adj:
            adjacencies[edge] = adjacencies.get(edge, 0) + 1
    directed_table = pd.DataFrame(
        [
            {
                "condition": condition,
                "cause": cause,
                "lag": lag,
                "effect": effect,
                "frequency": count / N_SUBSAMPLES,
            }
            for (cause, lag, effect), count in directed.items()
            if count / N_SUBSAMPLES >= GAMMA
        ]
    )
    if not directed_table.empty:
        directed_table["keep_after_prior"] = directed_table.apply(
            lambda row: int(row["lag"]) > 0 or LAYER_ORDER[row["cause"]] <= LAYER_ORDER[row["effect"]],
            axis=1,
        )
        directed_table = directed_table[directed_table["keep_after_prior"]].copy()
    adjacency_table = pd.DataFrame(
        [
            {
                "condition": condition,
                "variable_1": first,
                "variable_2": second,
                "frequency": count / N_SUBSAMPLES,
            }
            for (first, second), count in adjacencies.items()
            if count / N_SUBSAMPLES >= GAMMA
        ]
    )
    return directed_table, adjacency_table


def edge_set(table: pd.DataFrame):
    if table.empty:
        return set()
    return set(zip(table["cause"], table["lag"].astype(int), table["effect"]))


def adjacency_set(table: pd.DataFrame):
    if table.empty or "variable_1" not in table.columns or "variable_2" not in table.columns:
        return set()
    return set(zip(table["variable_1"], table["variable_2"]))


def main():
    out = ensure_output_dir("01_shared_structure_discovery")
    rng = np.random.default_rng(20260523)
    domains = load_shared_domains()

    prior_tables = {}
    strict_tables = {}
    adjacency_tables = {}
    segment_rows = []

    for condition, domain in domains.items():
        segments, metadata = continuous_segments(domain, min_segment_length=MIN_SEGMENT_LENGTH)
        for item in metadata:
            item["condition"] = condition
        segment_rows.extend(metadata)

        all_edges, prior_kept = discover_pcmci(segments, condition, rng)
        strict_edges, adjacency_edges = discover_pcmciplus(segments, condition, rng)

        all_edges.to_csv(out / f"{condition.lower()}_pcmci_edges.csv", index=False)
        prior_kept.to_csv(out / f"{condition.lower()}_shared_candidate_edges.csv", index=False)
        strict_edges.to_csv(out / f"{condition.lower()}_pcmci_plus_directed_edges.csv", index=False)
        adjacency_edges.to_csv(out / f"{condition.lower()}_pcmci_plus_adjacencies.csv", index=False)

        prior_tables[condition] = prior_kept
        strict_tables[condition] = strict_edges
        adjacency_tables[condition] = adjacency_edges

    pd.DataFrame(segment_rows).to_csv(out / "continuous_segments.csv", index=False)

    shared_candidates = edge_set(prior_tables["S1"]) & edge_set(prior_tables["S2"]) & edge_set(prior_tables["S3"])
    shared_directed = edge_set(strict_tables["S1"]) & edge_set(strict_tables["S2"]) & edge_set(strict_tables["S3"])
    shared_adjacency = (
        adjacency_set(adjacency_tables["S1"])
        & adjacency_set(adjacency_tables["S2"])
        & adjacency_set(adjacency_tables["S3"])
    )

    pd.DataFrame(sorted(shared_candidates), columns=["cause", "lag", "effect"]).to_csv(
        out / "shared_candidate_edges.csv", index=False
    )
    pd.DataFrame(sorted(shared_directed), columns=["cause", "lag", "effect"]).to_csv(
        out / "shared_strict_directed_edges.csv", index=False
    )
    pd.DataFrame(sorted(shared_adjacency), columns=["variable_1", "variable_2"]).to_csv(
        out / "shared_strict_adjacencies.csv", index=False
    )

    with open(out / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 01 Shared Structure Discovery\n\n")
        handle.write(
            "This script discovers shared pressure-load structure across S1, S2, and S3 "
            "using PCMCI and then checks directional robustness using PCMCI+.\n\n"
        )
        handle.write("## Shared candidate edges after engineering prior\n\n")
        if shared_candidates:
            for cause, lag, effect in sorted(shared_candidates):
                handle.write(f"- {COMMON_DISPLAY[cause]}(t-{lag}) -> {COMMON_DISPLAY[effect]}\n")
        else:
            handle.write("None\n")
        handle.write("\n## Shared strict directed edges from PCMCI+\n\n")
        if shared_directed:
            for cause, lag, effect in sorted(shared_directed):
                handle.write(f"- {COMMON_DISPLAY[cause]}(t-{lag}) -> {COMMON_DISPLAY[effect]}\n")
        else:
            handle.write("None\n")
        handle.write("\n## Shared strict contemporaneous adjacencies from PCMCI+\n\n")
        if shared_adjacency:
            for first, second in sorted(shared_adjacency):
                handle.write(f"- {COMMON_DISPLAY[first]} -- {COMMON_DISPLAY[second]}\n")
        else:
            handle.write("None\n")


if __name__ == "__main__":
    main()
