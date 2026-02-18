"""Central configuration for CRF estimation pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
SEGMENTS_SOURCE_DIR = DATA_DIR / "segments_source"
SEGMENTS_LOWRES_DIR = DATA_DIR / "segments_lowres"
ENCODED_DIR = DATA_DIR / "encoded"
VMAF_DIR = DATA_DIR / "vmaf"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ── Video segmentation ────────────────────────────────────────────────────
SEGMENT_DURATION = 2  # seconds

# ── Resolutions ───────────────────────────────────────────────────────────
DOWNSCALE_WIDTH = 426
DOWNSCALE_HEIGHT = 240

# ── Codec: SVT-AV1 ───────────────────────────────────────────────────────
CRF_VALUES = [15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]
NUM_CRF_CLASSES = len(CRF_VALUES)
CRF_TO_INDEX = {crf: i for i, crf in enumerate(CRF_VALUES)}
INDEX_TO_CRF = {i: crf for i, crf in enumerate(CRF_VALUES)}
SVT_PRESET = 8  # encoding speed/quality tradeoff (0=slow/best, 13=fast/worst)

# ── Feature extraction ────────────────────────────────────────────────────
# Per-frame features from ffprobe signalstats/entropy + siti-tools
# SI is excluded (does not correlate across resolutions per paper)
FRAME_FEATURES = ["Y", "U", "V", "SAT", "HUE", "SY", "SU", "SV", "TI"]
FEATURE_STATS = ["mean", "std", "kurtosis", "skewness"]
FRAME_RATE_CATEGORIES = [23.98, 24, 25, 29.97, 30, 50, 59.94, 60, 120]
NUM_FPS_CATEGORIES = len(FRAME_RATE_CATEGORIES)
FPS_TOLERANCE = 0.02

NUM_INPUT_FEATURES = (
    len(FRAME_FEATURES) * len(FEATURE_STATS)
    + 1  # scene_detection
    + NUM_FPS_CATEGORIES
    + 1  # target VMAF
)

# ── Training ──────────────────────────────────────────────────────────────
VMAF_CAP = 99  # cap VMAF values in training data
TRAIN_SPLIT = 0.8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 10

# ── Parallelism ───────────────────────────────────────────────────────────
MAX_WORKERS = 8  # for subprocess-based parallelism
