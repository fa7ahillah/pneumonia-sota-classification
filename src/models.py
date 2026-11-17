import time
import numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, MobileNetV2

from src import config


def build_transfer_model(arch_name):
    """
    Membangun model Transfer Learning (ResNet50 atau MobileNetV2)
    dengan bobot awal ImageNet dan lapisan head kustom.
    """
    if arch_name == "ResNet50":
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )
    elif arch_name == "MobileNetV2":
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )
    else:
        raise ValueError(f"Arsitektur '{arch_name}' tidak valid. Pilih 'ResNet50' atau 'MobileNetV2'.")

    # Bekukan seluruh lapisan internal backbone untuk Fase 1
    base_model.trainable = False

    inputs = layers.Input(shape=config.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{arch_name}_Transfer")
    return model, base_model


def train_two_phase(arch_name, train_gen, val_gen, epochs_p1=None, epochs_p2=None):
    """
    Menjalankan protokol pelatihan dua fase:
    - Fase 1 (Feature Extraction): Backbone freeze, lr = 1e-3
    - Fase 2 (Fine-Tuning): Unfreeze 30 lapisan terakhir, lr = 1e-5 + ReduceLROnPlateau
    """
    ep1 = epochs_p1 if epochs_p1 is not None else config.EPOCHS_PHASE_1
    ep2 = epochs_p2 if epochs_p2 is not None else config.EPOCHS_PHASE_2

    model, base_model = build_transfer_model(arch_name)

    # --- FASE 1: FEATURE EXTRACTION ---
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LR_PHASE_1),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    params_phase1 = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

    start_time = time.time()

    hist_1 = model.fit(
        train_gen,
        epochs=ep1,
        validation_data=val_gen,
        verbose=1
    )

    # --- FASE 2: FINE-TUNING (Unfreeze 30 lapisan terakhir) ---
    base_model.trainable = True
    for layer in base_model.layers[:-config.UNFREEZE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LR_PHASE_2),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    params_phase2 = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )

    hist_2 = model.fit(
        train_gen,
        epochs=ep2,
        validation_data=val_gen,
        callbacks=[lr_scheduler],
        verbose=1
    )

    training_duration = time.time() - start_time

    full_history = {
        "loss": hist_1.history["loss"] + hist_2.history["loss"],
        "val_loss": hist_1.history["val_loss"] + hist_2.history["val_loss"],
        "accuracy": hist_1.history["accuracy"] + hist_2.history["accuracy"],
        "val_accuracy": hist_1.history["val_accuracy"] + hist_2.history["val_accuracy"],
    }

    return model, full_history, training_duration, params_phase1, params_phase2