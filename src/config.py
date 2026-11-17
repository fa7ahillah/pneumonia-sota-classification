import os

# ==============================================================================
# 1. KONFIGURASI DIREKTORI (COLAB RUNTIME + GOOGLE DRIVE)
# ==============================================================================
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/DeepLearning/Pneumonia SoTA Classification"

# Direktori lokal cepat di VM Colab untuk ekstraksi ribuan file citra mentah
LOCAL_WORK_DIR = "/content/pneumonia_workspace"
RAW_DATA_DIR = os.path.join(LOCAL_WORK_DIR, "raw")

# Sub-direktori permanen di Google Drive
SPLITS_DIR = os.path.join(DRIVE_PROJECT_DIR, "data", "splits")
SAVED_MODELS_DIR = os.path.join(DRIVE_PROJECT_DIR, "saved_models")
LOGS_DIR = os.path.join(DRIVE_PROJECT_DIR, "results", "logs")
PLOTS_DIR = os.path.join(DRIVE_PROJECT_DIR, "results", "plots")

# ==============================================================================
# 2. PARAMETER DATASET & REPRODUCIBILITY
# ==============================================================================
KAGGLE_DATASET_SLUG = "pcbreviglieri/pneumonia-xray-images"
SEED = 42
IMG_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 32

# Penyeimbangan data: 1.000 NORMAL + 1.000 PNEUMONIA = 2.000 citra
SAMPLES_PER_CLASS = 1000
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # NORMAL = 0, PNEUMONIA = 1 (kelas positif)

# Rasio pembagian data (70% Train, 15% Validation, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# 3. PARAMETER AUGMENTASI TERSTRUKTUR (KHUSUS DATA PELATIHAN)
# ==============================================================================
AUGMENTATION_CONFIG = {
    "rotation_range": 10,
    "width_shift_range": 0.05,
    "height_shift_range": 0.05,
    "zoom_range": 0.05,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# ==============================================================================
# 4. PARAMETER MODEL & PELATIHAN DUA FASE (10x REPETISI)
# ==============================================================================
ARCHITECTURES = ["ResNet50", "MobileNetV2"]
DROPOUT_RATE = 0.5
N_REPETITIONS = 10

# Fase 1: Feature Extraction (Backbone Freeze)
EPOCHS_PHASE_1 = 10
LR_PHASE_1 = 1e-3

# Fase 2: Fine-Tuning (Unfreeze 30 lapisan terakhir)
UNFREEZE_LAYERS = 30
EPOCHS_PHASE_2 = 20
LR_PHASE_2 = 1e-5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-7


def ensure_directories():
    """Memastikan seluruh direktori kerja lokal dan Google Drive tersedia."""
    dirs = [
        RAW_DATA_DIR,
        SPLITS_DIR,
        os.path.join(SAVED_MODELS_DIR, "ResNet50"),
        os.path.join(SAVED_MODELS_DIR, "MobileNetV2"),
        LOGS_DIR,
        PLOTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)