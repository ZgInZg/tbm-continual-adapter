from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import COMMON_FEATURES, DISPLAY_TO_RAW, MAIN_DATA, RIDGE, RETENTION_WEIGHT, S3_DATA


def ensure_output_dir(name: str) -> Path:
    from config import RESULTS_DIR

    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_data_files():
    missing = [path for path in [MAIN_DATA, S3_DATA] if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required data file(s): "
            f"{names}. Place the CSV files in course_assignment_code/data/."
        )


def load_shared_domains():
    require_data_files()
    raw = pd.read_csv(MAIN_DATA)
    s3 = pd.read_csv(S3_DATA)

    def condition(display_name: str) -> pd.DataFrame:
        raw_name = DISPLAY_TO_RAW[display_name]
        data = raw[raw["project_id"] == raw_name].copy()
        data = data.loc[~data["ring_number"].duplicated(keep=False)].sort_values("ring_number")
        frame = pd.DataFrame(
            {
                "ring_number": pd.to_numeric(data["ring_number"], errors="coerce"),
                "pressure_state": pd.to_numeric(
                    data["foam_chamber_pressure__W19020205"], errors="coerce"
                ),
                "total_thrust": pd.to_numeric(data["total_thrust_kn__W16010002"], errors="coerce"),
                "cutterhead_torque": pd.to_numeric(data["torque_mnm__W15000105"], errors="coerce"),
            }
        ).dropna()
        frame["condition"] = display_name
        return frame.reset_index(drop=True)

    external = pd.DataFrame(
        {
            "ring_number": pd.to_numeric(s3["ring_number"], errors="coerce"),
            "pressure_state": pd.to_numeric(s3["工作舱压力"], errors="coerce"),
            "total_thrust": pd.to_numeric(s3["总推力"], errors="coerce"),
            "cutterhead_torque": pd.to_numeric(s3["刀盘扭矩1"], errors="coerce") / 1000.0,
        }
    ).dropna()
    external = external.sort_values("ring_number").reset_index(drop=True)
    external["condition"] = "S3"
    return {"S1": condition("S1"), "S2": condition("S2"), "S3": external}


def continuous_segments(domain: pd.DataFrame, min_segment_length: int = 20):
    valid = domain[COMMON_FEATURES].notna().all(axis=1)
    segments = []
    current = []
    previous = None
    for _, row in domain.loc[valid].iterrows():
        ring = int(row["ring_number"])
        if previous is None or ring == previous + 1:
            current.append(row)
        else:
            if len(current) >= min_segment_length:
                segments.append(pd.DataFrame(current))
            current = [row]
        previous = ring
    if len(current) >= min_segment_length:
        segments.append(pd.DataFrame(current))
    combined = pd.concat([segment[COMMON_FEATURES] for segment in segments], ignore_index=True)
    mean = combined.mean()
    std = combined.std(ddof=0).replace(0, 1)
    normalized = [(segment[COMMON_FEATURES] - mean) / std for segment in segments]
    metadata = [
        {
            "segment_id": index + 1,
            "ring_start": int(segment["ring_number"].iloc[0]),
            "ring_end": int(segment["ring_number"].iloc[-1]),
            "rows": len(segment),
        }
        for index, segment in enumerate(segments)
    ]
    return normalized, metadata


def make_design(data: pd.DataFrame, target: str, state_variables):
    rows = []
    feature_names = [f"{variable}_lag{lag}" for lag in [1, 2] for variable in state_variables]
    for index in range(2, len(data)):
        rings = data.loc[index - 2 : index, "ring_number"].to_numpy()
        if not (rings[1] - rings[0] == 1 and rings[2] - rings[1] == 1):
            continue
        values = []
        for lag in [1, 2]:
            values.extend(data.loc[index - lag, state_variables].to_numpy(dtype=float).tolist())
        row = dict(zip(feature_names, values))
        row["target"] = float(data.loc[index, target])
        rows.append(row)
    return pd.DataFrame(rows), feature_names


def anchor_mask(target: str, feature_names, include_pressure: bool):
    stable = {
        "total_thrust": {"total_thrust_lag1", "cutterhead_torque_lag1"},
        "cutterhead_torque": {"total_thrust_lag1", "cutterhead_torque_lag1"},
    }[target]
    if include_pressure:
        stable = set(stable)
        stable.add("pressure_state_lag1")
    return np.array([1.0 if name in stable else 0.05 for name in feature_names], dtype=float)


class StandardizedLinearModel:
    def __init__(self, x_scaler, y_mean, y_scale, weights):
        self.x_scaler = x_scaler
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.weights = weights

    def matrix(self, frame, columns):
        x = self.x_scaler.transform(frame[columns])
        return np.c_[np.ones(len(x)), x]

    def predict(self, frame, columns):
        pred = self.matrix(frame, columns) @ self.weights
        return pred * self.y_scale + self.y_mean


class AdapterModel:
    def __init__(self, anchor: StandardizedLinearModel, delta):
        self.anchor = anchor
        self.delta = delta

    def predict(self, frame, columns):
        standardized = self.anchor.matrix(frame, columns) @ (self.anchor.weights + self.delta)
        return standardized * self.anchor.y_scale + self.anchor.y_mean


def fit_initial_model(train: pd.DataFrame, columns):
    x_scaler = StandardScaler().fit(train[columns])
    y_mean = float(train["target"].mean())
    y_scale = float(train["target"].std()) or 1.0
    x = np.c_[np.ones(len(train)), x_scaler.transform(train[columns])]
    y = (train["target"].to_numpy() - y_mean) / y_scale
    penalty = np.eye(x.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return StandardizedLinearModel(x_scaler, y_mean, y_scale, weights)


def fit_from_data(reference: StandardizedLinearModel, train: pd.DataFrame, columns):
    x = reference.matrix(train, columns)
    y = (train["target"].to_numpy() - reference.y_mean) / reference.y_scale
    penalty = np.eye(x.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return StandardizedLinearModel(reference.x_scaler, reference.y_mean, reference.y_scale, weights)


def fit_anchored(reference: StandardizedLinearModel, observed: pd.DataFrame, columns, penalties, lam: float):
    x = reference.matrix(observed, columns)
    y = (observed["target"].to_numpy() - reference.y_mean) / reference.y_scale
    penalties = np.r_[0.0, penalties] * lam + np.r_[0.0, np.repeat(RIDGE, len(columns))]
    diagonal = np.diag(penalties)
    weights = np.linalg.solve(x.T @ x + diagonal, x.T @ y + diagonal @ reference.weights)
    return StandardizedLinearModel(reference.x_scaler, reference.y_mean, reference.y_scale, weights)


def metrics(model, frame: pd.DataFrame, columns):
    pred = model.predict(frame, columns)
    target = frame["target"].to_numpy()
    return float(r2_score(target, pred)), float(np.sqrt(mean_squared_error(target, pred)))


def adapter_metrics(model, frame: pd.DataFrame, columns):
    prediction = model.predict(frame, columns)
    target = frame["target"].to_numpy()
    residual = target - prediction
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(target - target.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.square(residual).mean()))
    return r2, rmse


def mean_r2(model, frames, columns):
    return float(np.mean([metrics(model, frame, columns)[0] for frame in frames]))


def choose_lambda(reference, observed, prior_validation, columns, penalties, lambda_grid):
    cut = max(10, int(round(len(observed) * 0.70)))
    adaptation_train = observed.iloc[:cut]
    adaptation_validation = observed.iloc[cut:]
    reference_retention = mean_r2(reference, prior_validation, columns)
    best = None
    for lam in lambda_grid:
        candidate = fit_anchored(reference, adaptation_train, columns, penalties, lam)
        target_score = metrics(candidate, adaptation_validation, columns)[0]
        retention = mean_r2(candidate, prior_validation, columns)
        forgetting = reference_retention - retention
        score = target_score - RETENTION_WEIGHT * max(0.0, forgetting)
        if best is None or score > best["score"]:
            best = {"lambda": lam, "score": score}
    return float(best["lambda"])


def adapt_global_model(strategy, reference, observed, memory, prior_validation, columns, penalties, seed, lambda_grid):
    if strategy == "source_only":
        return reference, np.nan, 0
    if strategy == "target_only":
        return fit_from_data(reference, observed, columns), np.nan, 0
    if strategy == "full_rehearsal":
        training = pd.concat([memory, observed], ignore_index=True)
        return fit_from_data(reference, training, columns), np.nan, len(memory)
    if strategy == "random_replay":
        n = min(len(memory), max(10, int(round(len(observed) * 0.35))))
        replay = memory.sample(n=n, random_state=seed)
        training = pd.concat([replay, observed], ignore_index=True)
        return fit_from_data(reference, training, columns), np.nan, n
    if strategy in {"uniform_anchor", "mechanism_anchor"}:
        mask = np.ones(len(columns)) if strategy == "uniform_anchor" else penalties
        lam = choose_lambda(reference, observed, prior_validation, columns, mask, lambda_grid)
        return fit_anchored(reference, observed, columns, mask, lam), lam, 0
    raise ValueError(f"Unknown strategy: {strategy}")


def fit_adapter(anchor: StandardizedLinearModel, train: pd.DataFrame, columns, penalties, lam: float):
    x = anchor.matrix(train, columns)
    y = (train["target"].to_numpy() - anchor.y_mean) / anchor.y_scale
    residual = y - x @ anchor.weights
    diagonal = np.diag(np.r_[0.0, penalties] * lam + np.r_[0.0, np.repeat(RIDGE, len(columns))])
    delta = np.linalg.solve(x.T @ x + diagonal, x.T @ residual)
    return AdapterModel(anchor, delta)


def choose_adapter(anchor: StandardizedLinearModel, observed: pd.DataFrame, columns, penalties, lambda_grid):
    cut = max(10, int(round(len(observed) * 0.70)))
    train = observed.iloc[:cut]
    validation = observed.iloc[cut:]
    best = None
    for lam in lambda_grid:
        candidate = fit_adapter(anchor, train, columns, penalties, lam)
        score = adapter_metrics(candidate, validation, columns)[0]
        if best is None or score > best["score"]:
            best = {"lambda": float(lam), "score": float(score)}
    return fit_adapter(anchor, observed, columns, penalties, best["lambda"]), best["lambda"]


def fit_subset_adapter(anchor: StandardizedLinearModel, train: pd.DataFrame, columns, selected_columns, lam: float):
    full_x = anchor.matrix(train, columns)
    indices = [0] + [columns.index(column) + 1 for column in selected_columns]
    x = full_x[:, indices]
    target = (train["target"].to_numpy() - anchor.y_mean) / anchor.y_scale
    residual = target - full_x @ anchor.weights
    penalty = np.eye(len(indices)) * lam
    penalty[0, 0] = 0.0
    small_delta = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    delta = np.zeros(len(columns) + 1)
    delta[indices] = small_delta
    return AdapterModel(anchor, delta)


def choose_subset_adapter(anchor: StandardizedLinearModel, observed: pd.DataFrame, columns, selected_columns, lambda_grid):
    cut = min(len(observed) - 3, max(5, int(round(len(observed) * 0.70))))
    train = observed.iloc[:cut]
    validation = observed.iloc[cut:]
    best = None
    for lam in lambda_grid:
        candidate = fit_subset_adapter(anchor, train, columns, selected_columns, lam)
        score = adapter_metrics(candidate, validation, columns)[0]
        if best is None or score > best["score"]:
            best = {"lambda": float(lam), "score": float(score)}
    return fit_subset_adapter(anchor, observed, columns, selected_columns, best["lambda"]), best["lambda"]


def split_initial(frame: pd.DataFrame):
    first = int(round(len(frame) * 0.60))
    second = int(round(len(frame) * 0.80))
    return frame.iloc[:first].copy(), frame.iloc[first:second].copy(), frame.iloc[second:].copy()


def split_adaptation(frame: pd.DataFrame):
    end = max(15, int(round(len(frame) * 0.33)))
    return frame.iloc[:end].copy(), frame.iloc[end:].copy()
