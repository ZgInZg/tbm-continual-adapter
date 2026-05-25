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

from config import COMMON_FEATURES
from shared_utils import ensure_output_dir, load_shared_domains


COLORS = {"S1": "#2A5C7A", "S2": "#D97941", "S3": "#4E8B57"}
DISPLAY_CONDITION = {"S1": "S1", "S2": "S2", "S3": "S3"}
DISPLAY_FEATURE = {
    "pressure_state": "土舱压力",
    "total_thrust": "总推力",
    "cutterhead_torque": "刀盘扭矩",
}
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
        alpha=0.10,
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
    balanced.to_csv(out / "balanced_pca_coordinates_zh.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(12.8, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.2], wspace=0.26)
    ax_pca = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    for axis in [ax_pca, ax_box]:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for name in ["S1", "S2", "S3"]:
        points = balanced.loc[balanced["condition"] == name, ["PC1", "PC2"]].to_numpy()
        ax_pca.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            color=COLORS[name],
            alpha=0.58,
            edgecolors="none",
            label=f"{DISPLAY_CONDITION[name]} (n={common_n})",
        )
        confidence_ellipse(points, ax_pca, COLORS[name])
        centroid = points.mean(axis=0)
        ax_pca.scatter(centroid[0], centroid[1], marker="X", s=95, color=COLORS[name], edgecolor="white")
    explained = pca.explained_variance_ratio_ * 100
    ax_pca.set_title("等量样本 PCA 可视化", loc="left", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel(f"主成分 1（解释方差 {explained[0]:.1f}%）")
    ax_pca.set_ylabel(f"主成分 2（解释方差 {explained[1]:.1f}%）")
    ax_pca.axhline(0, color="#DDDDDD", linewidth=0.8)
    ax_pca.axvline(0, color="#DDDDDD", linewidth=0.8)
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
        whiskerprops={"linewidth": 1.0, "color": "#4B4B4B"},
        capprops={"linewidth": 1.0, "color": "#4B4B4B"},
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(color)
    ax_box.set_title("共享状态分布对比", loc="left", fontsize=12, fontweight="bold")
    ax_box.set_ylabel("池化标准化数值")
    ax_box.set_xticks([2, 6.5, 11])
    ax_box.set_xticklabels([DISPLAY_FEATURE[feature] for feature in COMMON_FEATURES])
    ax_box.axhline(0, color="#DDDDDD", linewidth=0.8)
    ax_box.grid(axis="y", color="#EEEEEE", linewidth=0.7)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS[name], label=DISPLAY_CONDITION[name], markersize=8)
        for name in ["S1", "S2", "S3"]
    ]
    ax_box.legend(handles=handles, frameon=False, ncol=3, loc="upper right")

    fig.suptitle(
        "共享压力-载荷状态空间下的工况分布差异",
        x=0.055,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.055,
        -0.02,
        "左图采用各工况等量样本进行 PCA；右图采用全部有效样本绘制箱线图。",
        fontsize=9.5,
        color="#555555",
    )
    fig.savefig(out / "condition_shift_overview_zh.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "condition_shift_overview_zh.pdf", bbox_inches="tight", facecolor="white")
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
                        "工况对": f"{DISPLAY_CONDITION[left]}-{DISPLAY_CONDITION[right]}",
                        "变量": DISPLAY_FEATURE[feature],
                        "标准化 Wasserstein 距离": distance,
                    }
                )
            rows.append(
                {
                    "工况对": f"{DISPLAY_CONDITION[left]}-{DISPLAY_CONDITION[right]}",
                    "变量": "平均距离",
                    "标准化 Wasserstein 距离": float(np.mean(distances)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "pairwise_wasserstein_distances_zh.csv", index=False)
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
                "折次": fold + 1,
                "平衡准确率": balanced_accuracy_score(truth, prediction),
                "宏平均 F1": f1_score(truth, prediction, average="macro"),
            }
        )
    table = pd.DataFrame(folds)
    table.to_csv(out / "blocked_condition_classification_zh.csv", index=False)
    return table


def main():
    out = ensure_output_dir("03_condition_shift_validation")
    frames = load_shared_domains()
    standardized = standardized_frames(frames)
    common_n, explained = create_figure(out, frames, standardized)
    distances = compute_distances(out, standardized)
    classification = blocked_classification(out, frames)
    with open(out / "summary_zh.md", "w", encoding="utf-8") as handle:
        handle.write("# 03 工况分布差异验证（中文图版）\n\n")
        handle.write(
            f"PCA 每个工况采用 {common_n} 条等量样本。主成分 1 和主成分 2 的解释方差分别为 "
            f"{explained[0]:.2f}% 和 {explained[1]:.2f}%。\n\n"
        )
        handle.write("## 工况间标准化 Wasserstein 距离\n\n")
        handle.write(distances.round(4).to_markdown(index=False))
        handle.write("\n\n## 工况可辨识性检验\n\n")
        handle.write(classification.round(4).to_markdown(index=False))
        handle.write(
            f"\n\n平均平衡准确率：{classification['平衡准确率'].mean():.4f}\n"
        )


if __name__ == "__main__":
    main()
