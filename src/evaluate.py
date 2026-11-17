import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src import config


def evaluate_on_test_set(model, test_gen):
    """Menghitung metrik klasifikasi dan confusion matrix pada data pengujian."""
    test_gen.reset()
    y_true = np.array(test_gen.classes)
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "cm": cm
    }
    return metrics


def summarize_results(all_results_df):
    """Menghitung rata-rata (mean) dan simpangan baku (std) lintas repetisi."""
    cols = [
        "Trainable_Params_Phase2",
        "Training_Time_Sec",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "FP",
        "FN"
    ]
    summary = all_results_df.groupby("Model")[cols].agg(["mean", "std"])
    summary_path = os.path.join(config.LOGS_DIR, "ringkasan_mean_std_sota.csv")
    summary.to_csv(summary_path)
    return summary


def _format_cm_annotations(cm_array, is_float=False):
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    total = np.sum(cm_array)
    annot = np.empty_like(cm_array, dtype=object)
    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            pct = (val / total) * 100
            val_str = f"{val:.1f}" if is_float else f"{int(val)}"
            annot[i, j] = f"{labels[i, j]}\n{val_str}\n({pct:.1f}%)"
    return annot


def plot_comparison_confusion_matrices(best_resnet, best_mobilenet, all_results_df):
    """Membuat dan menyimpan grafik perbandingan Confusion Matrix (Best Run & Mean 10x)."""
    class_names = ["NORMAL (0)", "PNEUMONIA (1)"]

    # 1. Confusion Matrix Repetisi Terbaik
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (info, cmap) in zip(axes, [(best_resnet, "Blues"), (best_mobilenet, "Greens")]):
        annot = _format_cm_annotations(info["cm"], is_float=False)
        sns.heatmap(
            info["cm"], annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(
            f"{info['arch_name']} (Best Run #{info['best_run']})\n"
            f"Acc: {info['accuracy']*100:.2f}% | Prec: {info['precision']*100:.2f}% | Rec: {info['recall']*100:.2f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Prediksi Model", fontsize=10, fontweight="bold")
        ax.set_ylabel("Label Aktual", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_best_comparison.png"), dpi=300)
    plt.show()

    # 2. Rata-rata Confusion Matrix Lintas Repetisi
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (arch, cmap) in zip(axes, [("ResNet50", "Blues"), ("MobileNetV2", "Greens")]):
        sub = all_results_df[all_results_df["Model"] == arch]
        mean_cm = np.array([
            [sub["TN"].mean(), sub["FP"].mean()],
            [sub["FN"].mean(), sub["TP"].mean()]
        ])
        annot = _format_cm_annotations(mean_cm, is_float=True)
        sns.heatmap(
            mean_cm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(
            f"{arch} - Rata-rata Repetisi\n"
            f"Mean Acc: {sub['Accuracy'].mean()*100:.2f}% | Mean FP: {sub['FP'].mean():.1f} | Mean FN: {sub['FN'].mean():.1f}",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Prediksi Model", fontsize=10, fontweight="bold")
        ax.set_ylabel("Label Aktual", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_mean_comparison.png"), dpi=300)
    plt.show()


def plot_learning_curves(histories, arch_name, split_epoch=None):
    """Membuat dan menyimpan grafik rata-rata kurva Loss & Accuracy."""
    ep_split = split_epoch if split_epoch is not None else config.EPOCHS_PHASE_1
    total_epochs = len(histories[0]["accuracy"])
    epochs = range(1, total_epochs + 1)

    mean_train_acc = np.mean([h["accuracy"] for h in histories], axis=0)
    mean_val_acc = np.mean([h["val_accuracy"] for h in histories], axis=0)
    mean_train_loss = np.mean([h["loss"] for h in histories], axis=0)
    mean_val_loss = np.mean([h["val_loss"] for h in histories], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, mean_train_acc, label="Train Accuracy (Mean)", linewidth=2)
    axes[0].plot(epochs, mean_val_acc, label="Val Accuracy (Mean)", linewidth=2)
    axes[0].axvline(x=ep_split, color="gray", linestyle="--", label=f"Start Fine-Tuning (Epoch {ep_split})")
    axes[0].set_title(f"{arch_name} - Rata-rata Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, mean_train_loss, label="Train Loss (Mean)", linewidth=2)
    axes[1].plot(epochs, mean_val_loss, label="Val Loss (Mean)", linewidth=2)
    axes[1].axvline(x=ep_split, color="gray", linestyle="--", label=f"Start Fine-Tuning (Epoch {ep_split})")
    axes[1].set_title(f"{arch_name} - Rata-rata Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Crossentropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"learning_curve_{arch_name}.png"), dpi=300)
    plt.show()