import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from config import COMMON_FEATURES, COMMON_DISPLAY, LAYER_ORDER
from shared_utils import continuous_segments, ensure_output_dir, load_shared_domains


N_SUBSAMPLES = 20
TAU_MAX = 2
PC_ALPHA = 0.01
EDGE_ALPHA = 0.01
MIN_SEGMENT_LENGTH = 20
WINDOW_FRACTIONS = [0.60, 0.70, 0.80, 0.90]
GAMMAS = [0.50, 0.60, 0.70]


def sample_windows(segments, fraction, rng):
    windows = {}
    for index, segment in enumerate(segments):
        length = len(segment)
        size = min(length, max(MIN_SEGMENT_LENGTH, int(length * fraction)))
        start = 0 if size == length else int(rng.integers(0, length - size + 1))
        windows[index] = segment.iloc[start : start + size].to_numpy()
    return windows


def edge_frequency(segments, fraction, rng):
    records = {}
    for _ in range(N_SUBSAMPLES):
        dataframe = pp.DataFrame(
            sample_windows(segments, fraction, rng), analysis_mode="multiple", var_names=COMMON_FEATURES
        )
        result = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0).run_pcmci(
            tau_max=TAU_MAX, pc_alpha=PC_ALPHA
        )
        seen = set()
        for i, cause in enumerate(COMMON_FEATURES):
            for j, effect in enumerate(COMMON_FEATURES):
                for lag in range(TAU_MAX + 1):
                    if lag == 0 and i == j:
                        continue
                    if result["p_matrix"][i, j, lag] <= EDGE_ALPHA:
                        seen.add((cause, lag, effect))
        for key in seen:
            records[key] = records.get(key, 0) + 1
    return {key: count / N_SUBSAMPLES for key, count in records.items()}


def keep_prior(edge):
    cause, lag, effect = edge
    return lag > 0 or LAYER_ORDER[cause] <= LAYER_ORDER[effect]


def main():
    out = ensure_output_dir("02_shared_structure_sensitivity")
    rng = np.random.default_rng(20260523)
    segments = {
        condition: continuous_segments(domain, min_segment_length=MIN_SEGMENT_LENGTH)[0]
        for condition, domain in load_shared_domains().items()
    }
    baseline = {
        ("pressure_state", 0, "cutterhead_torque"),
        ("pressure_state", 0, "total_thrust"),
        ("pressure_state", 1, "pressure_state"),
        ("cutterhead_torque", 0, "total_thrust"),
        ("cutterhead_torque", 1, "cutterhead_torque"),
        ("total_thrust", 0, "cutterhead_torque"),
        ("total_thrust", 1, "total_thrust"),
    }
    rows = []
    for fraction in WINDOW_FRACTIONS:
        frequencies = {
            condition: edge_frequency(condition_segments, fraction, rng)
            for condition, condition_segments in segments.items()
        }
        for gamma in GAMMAS:
            sets = {
                condition: {
                    edge for edge, frequency in table.items() if frequency >= gamma and keep_prior(edge)
                }
                for condition, table in frequencies.items()
            }
            shared = sets["S1"] & sets["S2"] & sets["S3"]
            rows.append(
                {
                    "window_fraction": fraction,
                    "gamma": gamma,
                    "edges_S1": len(sets["S1"]),
                    "edges_S2": len(sets["S2"]),
                    "edges_S3": len(sets["S3"]),
                    "shared_three_edges": len(shared),
                    "baseline_recall": len(shared & baseline) / len(baseline),
                    "shared_edges": "; ".join(
                        f"{COMMON_DISPLAY[cause]}(t-{lag})->{COMMON_DISPLAY[effect]}"
                        for cause, lag, effect in sorted(shared)
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "sensitivity_grid.csv", index=False)
    with open(out / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 02 Shared Structure Sensitivity\n\n")
        handle.write(
            "This script repeats shared-structure discovery under different contiguous window fractions "
            "and stability thresholds.\n\n"
        )
        handle.write(summary.round(4).to_markdown(index=False))
        handle.write("\n")


if __name__ == "__main__":
    main()
