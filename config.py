from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

MAIN_DATA = DATA_DIR / "sheet1_cycle_duration_modeling_shrunk.csv"
S3_DATA = DATA_DIR / "new_condition_ring_level_mean.csv"

COMMON_FEATURES = ["pressure_state", "total_thrust", "cutterhead_torque"]
CONDITION_IDS = ["S1", "S2", "S3"]
RAW_TO_DISPLAY = {"S908": "S1", "S909": "S2", "S3": "S3"}
DISPLAY_TO_RAW = {"S1": "S908", "S2": "S909", "S3": "S3"}
COMMON_DISPLAY = {
    "pressure_state": "chamber_pressure",
    "total_thrust": "total_thrust",
    "cutterhead_torque": "cutterhead_torque",
}
LAYER_ORDER = {
    "pressure_state": 1,
    "total_thrust": 2,
    "cutterhead_torque": 2,
}

TARGETS = ["total_thrust", "cutterhead_torque"]
INPUT_SPACES = {
    "strict_load_backbone": ["total_thrust", "cutterhead_torque"],
    "pressure_load_backbone": ["pressure_state", "total_thrust", "cutterhead_torque"],
}

SEEDS = [11, 23, 37, 51, 73]
RIDGE = 1e-5
LAMBDA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RETENTION_WEIGHT = 0.2
