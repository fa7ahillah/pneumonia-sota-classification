import glob
import os
import shutil
import subprocess
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50, mobilenet_v2

from src import config


def _collect_all_images():
    """Mengumpulkan seluruh path citra NORMAL dan PNEUMONIA (folder opacity) dari folder raw."""
    valid_ext = (".png", ".jpg", ".jpeg")
    normal_files, pneumonia_files = [], []

    for root, _, files in os.walk(config.RAW_DATA_DIR):
        folder_name = os.path.basename(root).upper()
        if folder_name == "NORMAL":
            normal_files.extend(
                [os.path.join(root, f) for f in files if f.lower().endswith(valid_ext)]
            )
        elif folder_name in ("PNEUMONIA", "OPACITY"):
            pneumonia_files.extend(
                [os.path.join(root, f) for f in files if f.lower().endswith(valid_ext)]
            )

    normal_files.sort()
    pneumonia_files.sort()
    return normal_files, pneumonia_files


def _setup_kaggle_credentials(kaggle_username=None, kaggle_key=None):
    """Menyiapkan autentikasi Kaggle dari parameter, environment variable, atau file kaggle.json."""
    if kaggle_username and kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_username.strip()
        os.environ["KAGGLE_KEY"] = kaggle_key.strip()
        return

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    default_kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(default_kaggle_json):
        os.chmod(default_kaggle_json, 0o600)
        return

    # Cek apakah kaggle.json disimpan di folder proyek Google Drive
    drive_kaggle_json = os.path.join(config.DRIVE_PROJECT_DIR, "kaggle.json")
    if os.path.exists(drive_kaggle_json):
        os.makedirs(os.path.dirname(default_kaggle_json), exist_ok=True)
        shutil.copy(drive_kaggle_json, default_kaggle_json)
        os.chmod(default_kaggle_json, 0o600)
        print("[INFO] Kredensial dimuat dari kaggle.json di Google Drive.")
        return

    raise RuntimeError(
        "Kredensial Kaggle tidak ditemukan. Masukkan kaggle_username & kaggle_key, "
        "simpan di Colab Secrets, atau letakkan file kaggle.json di folder proyek Google Drive."
    )


def download_kaggle_dataset(kaggle_username=None, kaggle_key=None):
    """Mengunduh dan mengekstrak dataset Kaggle ke direktori lokal Colab."""
    config.ensure_directories()

    # Cek apakah citra untuk kedua kelas sudah tersedia di disk lokal VM Colab
    normal_files, pneumonia_files = _collect_all_images()
    if (
        len(normal_files) >= config.SAMPLES_PER_CLASS
        and len(pneumonia_files) >= config.SAMPLES_PER_CLASS
    ):
        total_found = len(normal_files) + len(pneumonia_files)
        print(
            f"[INFO] Dataset sudah tersedia ({total_found} citra: "
            f"{len(normal_files)} NORMAL, {len(pneumonia_files)} PNEUMONIA) di {config.RAW_DATA_DIR}"
        )
        return

    _setup_kaggle_credentials(kaggle_username, kaggle_key)

    dataset_slug = getattr(config, "KAGGLE_DATASET_SLUG", "pcbreviglieri/pneumonia-xray-images")
    print(f"[INFO] Mengunduh dataset '{dataset_slug}' dari Kaggle...")

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", config.LOCAL_WORK_DIR,
            "--force",
        ],
        check=True,
    )

    # Deteksi otomatis file .zip yang baru diunduh di LOCAL_WORK_DIR
    zip_candidates = sorted(
        glob.glob(os.path.join(config.LOCAL_WORK_DIR, "*.zip")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not zip_candidates:
        raise FileNotFoundError(f"Arsip ZIP hasil unduhan Kaggle tidak ditemukan di {config.LOCAL_WORK_DIR}")

    zip_path = zip_candidates[0]
    print(f"[INFO] Mengekstrak '{os.path.basename(zip_path)}' ke {config.RAW_DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(config.RAW_DATA_DIR)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"[INFO] Ekstraksi selesai di: {config.RAW_DATA_DIR}")


def prepare_balanced_splits(force_regenerate=False):
    """
    Melakukan Random Under Sampling sesuai config.SAMPLES_PER_CLASS
    dan Stratified Split (70% Train, 15% Val, 15% Test) dengan SEED = 42,
    lalu menyimpannya ke folder data/splits/.
    """
    config.ensure_directories()

    total_samples = config.SAMPLES_PER_CLASS * 2
    balanced_csv = os.path.join(config.SPLITS_DIR, f"balanced_{total_samples}.csv")
    train_csv = os.path.join(config.SPLITS_DIR, "train_split.csv")
    val_csv = os.path.join(config.SPLITS_DIR, "val_split.csv")
    test_csv = os.path.join(config.SPLITS_DIR, "test_split.csv")

    all_exist = all(os.path.exists(p) for p in [balanced_csv, train_csv, val_csv, test_csv])

    if all_exist and not force_regenerate:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        if os.path.exists(train_df["filepath"].iloc[0]):
            print("[INFO] Memuat manifest pembagian data yang sudah ada dari data/splits/...")
            return train_df, val_df, test_df

    normal_files, pneumonia_files = _collect_all_images()
    print(f"[INFO] Total Citra Asli -> NORMAL: {len(normal_files)}, PNEUMONIA: {len(pneumonia_files)}")

    if len(normal_files) < config.SAMPLES_PER_CLASS or len(pneumonia_files) < config.SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Jumlah citra mentah tidak mencukupi (butuh minimal {config.SAMPLES_PER_CLASS} per kelas). "
            f"Pastikan download_kaggle_dataset() sudah berhasil."
        )

    # Random Under Sampling terkontrol dengan SEED = 42
    rng = np.random.RandomState(config.SEED)
    sampled_normal = rng.choice(normal_files, size=config.SAMPLES_PER_CLASS, replace=False)
    sampled_pneumonia = rng.choice(pneumonia_files, size=config.SAMPLES_PER_CLASS, replace=False)

    balanced_df = pd.DataFrame({
        "filepath": np.concatenate([sampled_normal, sampled_pneumonia]),
        "label": ["NORMAL"] * config.SAMPLES_PER_CLASS + ["PNEUMONIA"] * config.SAMPLES_PER_CLASS,
    })

    # Stratified Split 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=balanced_df["label"],
        random_state=config.SEED,
    )

    relative_test_size = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=config.SEED,
    )

    balanced_df.to_csv(balanced_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("[INFO] File manifest CSV berhasil disimpan ke:", config.SPLITS_DIR)
    return train_df, val_df, test_df


def create_data_generators(train_df, val_df, test_df, arch_name="ResNet50", seed=config.SEED):
    """Membuat ImageDataGenerator untuk Train (dengan augmentasi), Val, dan Test (tanpa augmentasi)."""
    if arch_name == "ResNet50":
        preprocess_fn = resnet50.preprocess_input
    elif arch_name == "MobileNetV2":
        preprocess_fn = mobilenet_v2.preprocess_input
    else:
        raise ValueError(f"Arsitektur '{arch_name}' tidak dikenali. Pilih 'ResNet50' atau 'MobileNetV2'.")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        **config.AUGMENTATION_CONFIG,
    )
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        classes=config.CLASS_NAMES,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        classes=config.CLASS_NAMES,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        classes=config.CLASS_NAMES,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen