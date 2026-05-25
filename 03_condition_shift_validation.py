import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import COMMON_DISPLAY, COMMON_FEATURES
from shared_utils import ensure_output_dir, load_shared_domains


COLORS = {"S1": "#245B74", "S2": "#E07A4E", "S3": "#4F8A5B"}
SEED = 20260525


def confidence_ellipse(points, axis, color):
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 1.65 * np.sqrt(np.maximum(eigenvalues, 0))
    center = points.mean(axis=0)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.5,
        alpha=0.12,
    )
    axis.add_patch(ellipse)


def standardized_frames(frames):
    pooled = pd.concat([frames[name] for name in ["S1", "S2", "S3"]], ignore_index=True)
    scaler = StandardScaler().fit(pooled[COMMON_FEATURES])
    result = {}
    for name, frame in frames.items():
        standardized = frame.copy()
        standardized[COMMON_FEATURES] = scaler.transform(frame[COMMON_FEATURES])
        result[name] = standardized
    return result


def create_figure(out, frames, standardized):
    rng = np.random.default_rng(SEED)
    common_n = min(len(frames[name]) for name in ["S1", "S2", "S3"])
    balanced = []
    for name in ["S1", "S2", "S3"]:
        choices = rng.choice(len(frames[name]), size=common_n, replace=False)
        sampled = frames[name].iloc[choices].copy()
        balanced.append(sampled)
    balanced = pd.concat(balanced, ignore_index=True)
    x = StandardScaler().fit_transform(balanced[COMMON_FEATURES])
    pca = PCA(n_components=2, random_state=SEED)
    scores = pca.fit_transform(x)
    balanced["PC1"] = scores[:, 0]
    balanced["PC2"] = scores[:, 1]
    balanced.to_csv(out / "balanced_pca_coordinates.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(12.8, 5.2), facecolor="#FAF8F3")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.2], wspace=0.26)
    ax_pca = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    for axis in [ax_pca, ax_box]:
        axis.set_facecolor("#FAF8F3")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for name in ["S1", "S2", "S3"]:
        points = balanced.loc[balanced["condition"] == name, ["PC1", "PC2"]].to_numpy()
        ax_pca.scatter(points[:, 0], points[:, 1], s=22, color=COLORS[name], alpha=0.56, edgecolors="none", label=name)
        confidence_ellipse(points, ax_pca, COLORS[name])
        centroid = points.mean(axis=0)
        ax_pca.scatter(centroid[0], centroid[1], marker="X", s=95, color=COLORS[name], edgecolor="white")
    explained = pca.explained_variance_ratio_ * 100
    ax_pca.set_title("Balanced PCA view", loc="left", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel(f"PC1 ({explained[0]:.1f}% explained variance)")
    ax_pca.set_ylabel(f"PC2 ({explained[1]:.1f}% explained variance)")
    ax_pca.legend(frameon=False, loc="best")

    positions = []
    box_data = []
    box_colors = []
    for feature_index, feature in enumerate(COMMON_FEATURES):
        base = feature_index * 4.5
        for domain_index, name in enumerate(["S1", "S2", "S3"]):
            positions.append(base + domain_index + 1)
            box_data.append(standardized[name][feature].to_numpy())
            box_colors.append(COLORS[name])
    boxes = ax_box.boxplot(
        box_data,
        positions=positions,
        widths=0.74,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1E1E1E", "linewidth": 1.25},
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(color)
    ax_box.set_title("Shared-state distributions", loc="left", fontsize=12, fontweight="bold")
    ax_box.set_ylabel("Pooled standardized value")
    ax_box.set_xticks([2, 6.5, 11])
    ax_box.set_xticklabels([COMMON_DISPLAY[feature] for feature in COMMON_FEATURES])
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS[name], label=name, markersize=8)
        for name in ["S1", "S2", "S3"]
    ]
    ax_box.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.savefig(out / "condition_shift_overview.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out / "condition_shift_overview.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return common_n, explained


def compute_distances(out, standardized):
    rows = []
    for left_index, left in enumerate(["S1", "S2", "S3"]):
        for right in ["S1", "S2", "S3"][left_index + 1 :]:
            distances = []
            for feature in COMMON_FEATURES:
                distance = wasserstein_distance(
                    standardized[left][feature].to_numpy(),
                    standardized[right][feature].to_numpy(),
                )
                distances.append(distance)
                rows.append(
                    {
                        "condition_pair": f"{left}-{right}",
                        "variable": COMMON_DISPLAY[feature],
                        "standardized_wasserstein": distance,
                    }
                )
            rows.append(
                {
                    "condition_pair": f"{left}-{right}",
                    "variable": "mean_distance",
                    "standardized_wasserstein": float(np.mean(distances)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "pairwise_wasserstein_distances.csv", index=False)
    return table


def blocked_classification(out, frames):
    data = pd.concat([frames[name].sort_values("ring_number") for name in ["S1", "S2", "S3"]], ignore_index=True)
    folds = []
    for fold in range(5):
        test_indices = []
        for name in ["S1", "S2", "S3"]:
            condition_indices = data.index[data["condition"] == name].to_numpy()
            blocks = np.array_split(condition_indices, 5)
            test_indices.extend(blocks[fold].tolist())
        test_indices = np.array(test_indices)
        train_indices = data.index.difference(test_indices).to_numpy()
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classify", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        )
        model.fit(data.loc[train_indices, COMMON_FEATURES], data.loc[train_indices, "condition"])
        prediction = model.predict(data.loc[test_indices, COMMON_FEATURES])
        truth = data.loc[test_indices, "condition"]
        folds.append(
            {
                "fold": fold + 1,
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, average="macro"),
            }
        )
    table = pd.DataFrame(folds)
    table.to_csv(out / "blocked_condition_classification.csv", index=False)
    return table


def main():
    out = ensure_output_dir("03_condition_shift_validation")
    frames = load_shared_domains()
    standardized = standardized_frames(frames)
    common_n, explained = create_figure(out, frames, standardized)
    distances = compute_distances(out, standardized)
    classification = blocked_classification(out, frames)
    with open(out / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 03 Condition Shift Validation\n\n")
        handle.write(
            f"PCA uses {common_n} sampled records from each condition. "
            f"PC1 and PC2 explain {explained[0]:.2f}% and {explained[1]:.2f}% of variance.\n\n"
        )
        handle.write("## Pairwise standardized Wasserstein distance\n\n")
        handle.write(distances.round(4).to_markdown(index=False))
        handle.write("\n\n## Blocked condition classification\n\n")
        handle.write(classification.round(4).to_markdown(index=False))
        handle.write(
            f"\n\nMean balanced accuracy: {classification['balanced_accuracy'].mean():.4f}\n"
        )


if __name__ == "__main__":
    main()
