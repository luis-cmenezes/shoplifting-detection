#!/usr/bin/env python3
"""
Gera tabelas (LaTeX + CSV) e gráficos (PNG) para a seção
"Resultados dos Treinamentos" do TCC.

Uso:
    uv run scripts/generate_results.py
    uv run scripts/generate_results.py --config scripts/config.yaml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

# ──────────────────────────────────────────────────────────────────────
# Configurações globais de estilo
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

GPU_COST_USD_PER_HOUR = 0.412  # RTX 5090 cloud rental rate


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
I3D_DISPLAY_NAMES = {
    "aug_full_unfreeze_rgb_only": "I3D Completo / RGB",
    "aug_full_unfreeze_rgb_optical": "I3D Completo / RGB+Fluxo",
    "aug_head_unfreeze_rgb_only": "I3D Cabeça / RGB",
    "aug_head_unfreeze_rgb_optical": "I3D Cabeça / RGB+Fluxo",
}

I3D_SHORT_NAMES = {
    "aug_full_unfreeze_rgb_only": "Completo / RGB",
    "aug_full_unfreeze_rgb_optical": "Completo / RGB+Fluxo",
    "aug_head_unfreeze_rgb_only": "Cabeça / RGB",
    "aug_head_unfreeze_rgb_optical": "Cabeça / RGB+Fluxo",
}


def ts_display_name(dirname: str) -> str:
    """Converte nome do diretório do TimeSformer em label legível."""
    # ex: timesformer-base-finetuned-k400_frames32_unfreeze_all
    parts = dirname.split("_")
    # Extract pretrained dataset
    if "k400" in dirname:
        pretrained = "K400"
    else:
        pretrained = "SSv2"
    # Extract frames
    for p in parts:
        if p.startswith("frames"):
            frames = p.replace("frames", "")
            break
    # Extract freeze strategy
    if "unfreeze_all" in dirname:
        freeze = "Completo"
    else:
        freeze = "Cabeça"
    return f"TS {pretrained} / {frames}f / {freeze}"


def ts_short_name(dirname: str) -> str:
    if "k400" in dirname:
        pretrained = "K400"
    else:
        pretrained = "SSv2"
    for p in dirname.split("_"):
        if p.startswith("frames"):
            frames = p.replace("frames", "")
            break
    if "unfreeze_all" in dirname:
        freeze = "Completo"
    else:
        freeze = "Cabeça"
    return f"{pretrained}/{frames}f/{freeze}"


def _escape_latex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    """Salva DataFrame como tabela LaTeX standalone."""
    latex = df.to_latex(index=False, float_format="%.4f", escape=False)
    content = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{" + caption + "}\n"
        r"\label{" + label + "}\n"
        r"\resizebox{\textwidth}{!}{%" + "\n"
        + latex
        + "}\n"
        r"\end{table}" + "\n"
    )
    path.write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
@dataclass
class I3DExperiment:
    name: str
    display_name: str
    short_name: str
    df: pd.DataFrame
    roc_dir: Path
    best_epoch: int = 0
    best_auc: float = 0.0


@dataclass
class TSExperiment:
    name: str
    display_name: str
    short_name: str
    df: pd.DataFrame  # raw tb_metrics
    eval_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_epoch: float = 0.0
    best_auc: float = 0.0
    total_runtime_s: float = 0.0
    total_flos: float = 0.0


def load_i3d_experiments(results_dir: Path) -> list[I3DExperiment]:
    exps = []
    i3d_dir = results_dir / "i3d"
    for d in sorted(i3d_dir.iterdir()):
        csv_path = d / "training_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        roc_dir = d / "roc_curves"
        best_idx = df["val_auc_roc"].idxmax()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_auc = df.loc[best_idx, "val_auc_roc"]
        exps.append(
            I3DExperiment(
                name=d.name,
                display_name=I3D_DISPLAY_NAMES.get(d.name, d.name),
                short_name=I3D_SHORT_NAMES.get(d.name, d.name),
                df=df,
                roc_dir=roc_dir,
                best_epoch=best_epoch,
                best_auc=best_auc,
            )
        )
    return exps


def load_ts_experiments(results_dir: Path) -> list[TSExperiment]:
    exps = []
    ts_dir = results_dir / "timesformer"
    for d in sorted(ts_dir.iterdir()):
        csv_path = d / "final_model" / "tb_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        eval_df = df.dropna(subset=["eval/auc_roc"]).copy()
        train_df = df.dropna(subset=["train/loss"]).copy()

        best_auc = 0.0
        best_epoch = 0.0
        if len(eval_df) > 0:
            best_idx = eval_df["eval/auc_roc"].idxmax()
            best_auc = eval_df.loc[best_idx, "eval/auc_roc"]
            best_epoch = eval_df.loc[best_idx, "train/epoch"]

        runtime = df["train/train_runtime"].dropna()
        total_runtime = runtime.iloc[-1] if len(runtime) > 0 else 0.0
        flos = df["train/total_flos"].dropna()
        total_flos = flos.iloc[-1] if len(flos) > 0 else 0.0

        exps.append(
            TSExperiment(
                name=d.name,
                display_name=ts_display_name(d.name),
                short_name=ts_short_name(d.name),
                df=df,
                eval_df=eval_df,
                train_df=train_df,
                best_epoch=best_epoch,
                best_auc=best_auc,
                total_runtime_s=total_runtime,
                total_flos=total_flos,
            )
        )
    return exps


# ──────────────────────────────────────────────────────────────────────
# 1. Tabela de hyperparameters comparativos
# ──────────────────────────────────────────────────────────────────────
def generate_hyperparams_table(cfg: dict, out_dir: Path):
    i3d_cfg = cfg["training"]["i3d"]
    ts_cfg = cfg["training"]["timesformer"]

    rows = [
        ("Semente aleatória", str(i3d_cfg["seed"]), str(ts_cfg["seed"])),
        ("Épocas máximas", str(i3d_cfg["epochs"]), str(ts_cfg["epochs"])),
        ("Tamanho do lote", str(i3d_cfg["batch_size"]), str(ts_cfg["batch_size"])),
        ("Taxa de aprendizado", str(i3d_cfg["learning_rate"]), str(ts_cfg["learning_rate"])),
        ("Otimizador", "Adam", "AdamW (HF Trainer)"),
        ("Função de perda", "BCEWithLogitsLoss", "CrossEntropyLoss"),
        ("Early stopping", "Não", f'Sim (paciência={ts_cfg["early_stopping_patience"]})'),
        ("Acúmulo de gradiente", "1", str(ts_cfg["gradient_accumulation_steps"])),
        (
            "Divisão treino/val/teste",
            f'{int((1-i3d_cfg["split"]["test_size"])*100)}/{int(i3d_cfg["split"]["test_size"]*i3d_cfg["split"]["val_test_ratio"]*100)}/{int(i3d_cfg["split"]["test_size"]*(1-i3d_cfg["split"]["val_test_ratio"])*100)}',
            f'{int((1-ts_cfg["split"]["test_size"])*100)}/{int(ts_cfg["split"]["test_size"]*ts_cfg["split"]["val_test_ratio"]*100)}/{int(ts_cfg["split"]["test_size"]*(1-ts_cfg["split"]["val_test_ratio"])*100)}',
        ),
        (
            "Aumento de dados",
            f'Flip={i3d_cfg["augmentation"]["p_flip"]}, ColorJitter',
            f'Flip={ts_cfg["augmentation"]["p_flip"]}, ColorJitter',
        ),
        ("Frames de entrada", "64", "8 / 32 / 64"),
        ("Resolução", "224×224", "224×224"),
        ("Modalidades", "RGB / RGB+Fluxo Ótico", "RGB"),
        ("Fine-tuning", "Completo / Cabeça", "Completo / Cabeça"),
        ("Pré-treino", "Kinetics (ImageNet init)", "K400 / SSv2"),
    ]

    df = pd.DataFrame(rows, columns=["Parâmetro", "I3D", "TimeSformer"])
    df.to_csv(out_dir / "hyperparameters_table.csv", index=False)
    save_latex_table(
        df,
        out_dir / "hyperparameters_table.tex",
        caption="Comparação de hiperparâmetros entre os modelos I3D e TimeSformer.",
        label="tab:hyperparams",
    )
    print(f"  ✓ Tabela de hiperparâmetros salva")


# ──────────────────────────────────────────────────────────────────────
# 2. Tabela de variações dos experimentos
# ──────────────────────────────────────────────────────────────────────
def generate_experiment_variants_table(
    i3d_exps: list[I3DExperiment], ts_exps: list[TSExperiment], out_dir: Path
):
    rows = []
    for exp in i3d_exps:
        if "full_unfreeze" in exp.name:
            freeze = "Completo"
        else:
            freeze = "Somente cabeça"
        if "rgb_optical" in exp.name:
            modality = "RGB + Fluxo Ótico"
        else:
            modality = "Somente RGB"
        rows.append(("I3D", freeze, modality, "64", "Kinetics"))

    for exp in ts_exps:
        if "unfreeze_all" in exp.name:
            freeze = "Completo"
        else:
            freeze = "Somente cabeça"
        if "k400" in exp.name:
            pretrained = "K400"
        else:
            pretrained = "SSv2"
        for p in exp.name.split("_"):
            if p.startswith("frames"):
                frames = p.replace("frames", "")
                break
        rows.append(("TimeSformer", freeze, "RGB", frames, pretrained))

    df = pd.DataFrame(
        rows, columns=["Modelo", "Fine-tuning", "Modalidade", "Frames", "Pré-treino"]
    )
    df.to_csv(out_dir / "experiment_variants.csv", index=False)
    save_latex_table(
        df,
        out_dir / "experiment_variants.tex",
        caption="Variações de experimentos realizados.",
        label="tab:variants",
    )
    print(f"  ✓ Tabela de variações dos experimentos salva")


# ──────────────────────────────────────────────────────────────────────
# 3. Curvas de Loss (treino)
# ──────────────────────────────────────────────────────────────────────
def plot_i3d_training_loss(i3d_exps: list[I3DExperiment], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, exp in enumerate(i3d_exps):
        ax.plot(
            exp.df["epoch"],
            exp.df["train_loss"],
            label=exp.short_name,
            color=colors[i % len(colors)],
            linewidth=1.5,
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda")
    ax.set_title("I3D — Curva de Perda no Treinamento")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 70)
    fig.savefig(out_dir / "i3d_training_loss.png")
    plt.close(fig)
    print(f"  ✓ Gráfico I3D training loss salvo")


def plot_ts_training_loss(ts_exps: list[TSExperiment], out_dir: Path):
    # Agrupar: 3 linhas (8f, 32f, 64f) × 2 colunas (K400, SSv2)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
    fig.suptitle("TimeSformer — Curva de Perda no Treinamento", fontsize=14, y=1.02)

    pretrained_order = ["k400", "ssv2"]
    pretrained_labels = {"k400": "K400", "ssv2": "SSv2"}
    frame_order = ["8", "32", "64"]
    colors = {"Completo": "#1f77b4", "Cabeça": "#d62728"}

    for row, nf in enumerate(frame_order):
        for col, pt in enumerate(pretrained_order):
            ax = axes[row][col]
            for exp in ts_exps:
                if pt not in exp.name:
                    continue
                if f"frames{nf}_" not in exp.name:
                    continue
                freeze_label = "Completo" if "unfreeze_all" in exp.name else "Cabeça"
                train_data = exp.train_df.dropna(subset=["train/loss"])
                if len(train_data) == 0:
                    continue
                epoch_col = train_data["train/epoch"]
                ax.plot(
                    epoch_col,
                    train_data["train/loss"],
                    label=freeze_label,
                    color=colors[freeze_label],
                    linewidth=1.0,
                    alpha=0.8,
                )
            ax.set_title(f"{pretrained_labels[pt]} — {nf} Quadros")
            ax.set_xlabel("Época")
            ax.set_ylabel("Perda")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "timesformer_training_loss.png")
    plt.close(fig)
    print(f"  ✓ Gráfico TimeSformer training loss salvo")


# ──────────────────────────────────────────────────────────────────────
# 4. Métricas de validação ao longo das épocas
# ──────────────────────────────────────────────────────────────────────
def plot_i3d_validation_metrics(i3d_exps: list[I3DExperiment], out_dir: Path):
    metrics = [
        ("val_accuracy", "Acurácia"),
        ("val_f1", "F1-Score"),
        ("val_auc_roc", "AUC-ROC"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("I3D — Métricas de Validação por Época", fontsize=14, y=1.02)

    # Linha 1: 2 gráficos; Linha 2: 1 gráfico centralizado
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, (3, 4))
    axes = [ax0, ax1, ax2]

    for ax, (col, label) in zip(axes, metrics):
        for i, exp in enumerate(i3d_exps):
            ax.plot(
                exp.df["epoch"],
                exp.df[col],
                label=exp.short_name,
                color=colors[i % len(colors)],
                linewidth=1.3,
            )
        ax.set_xlabel("Época")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 70)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / "i3d_validation_metrics.png")
    plt.close(fig)
    print(f"  ✓ Gráfico I3D validation metrics salvo")


def plot_ts_validation_metrics(ts_exps: list[TSExperiment], out_dir: Path):
    eval_metrics = [
        ("eval/accuracy", "Acurácia"),
        ("eval/f1", "F1-Score"),
        ("eval/auc_roc", "AUC-ROC"),
    ]
    pretrained_order = ["k400", "ssv2"]
    pretrained_labels = {"k400": "K400", "ssv2": "SSv2"}
    frame_order = ["8", "32", "64"]
    colors = {"Completo": "#1f77b4", "Cabeça": "#d62728"}

    for metric_col, metric_label in eval_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=True)
        fig.suptitle(
            f"TimeSformer — {metric_label} de Validação por Época",
            fontsize=14,
            y=1.02,
        )
        for row, pt in enumerate(pretrained_order):
            for col_idx, nf in enumerate(frame_order):
                ax = axes[row][col_idx]
                for exp in ts_exps:
                    if pt not in exp.name:
                        continue
                    if f"frames{nf}_" not in exp.name:
                        continue
                    freeze_label = (
                        "Completo" if "unfreeze_all" in exp.name else "Cabeça"
                    )
                    edf = exp.eval_df
                    if len(edf) == 0:
                        continue
                    ax.plot(
                        edf["train/epoch"],
                        edf[metric_col],
                        label=freeze_label,
                        color=colors[freeze_label],
                        linewidth=1.3,
                        marker=".",
                        markersize=3,
                    )
                ax.set_title(f"{pretrained_labels[pt]} — {nf} Quadros")
                ax.set_xlabel("Época")
                ax.set_ylabel(metric_label)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.05)

        fig.tight_layout()
        safe_name = metric_col.replace("/", "_")
        fig.savefig(out_dir / f"timesformer_val_{safe_name}.png")
        plt.close(fig)

    print(f"  ✓ Gráficos TimeSformer validation metrics salvos")


# ──────────────────────────────────────────────────────────────────────
# 5. Tabela: Melhores métricas de validação
# ──────────────────────────────────────────────────────────────────────
def generate_best_validation_table(
    i3d_exps: list[I3DExperiment], ts_exps: list[TSExperiment], out_dir: Path
):
    rows = []
    for exp in i3d_exps:
        best_idx = exp.df["val_auc_roc"].idxmax()
        r = exp.df.iloc[best_idx]
        rows.append(
            {
                "Modelo": "I3D",
                "Experimento": exp.short_name,
                "Melhor Época": int(r["epoch"]),
                "Acurácia": r["val_accuracy"],
                "Precisão": r["val_precision"],
                "Revocação": r["val_recall"],
                "F1-Score": r["val_f1"],
                "AUC-ROC": r["val_auc_roc"],
            }
        )

    for exp in ts_exps:
        if len(exp.eval_df) == 0:
            continue
        best_idx = exp.eval_df["eval/auc_roc"].idxmax()
        r = exp.eval_df.loc[best_idx]
        rows.append(
            {
                "Modelo": "TimeSformer",
                "Experimento": exp.short_name,
                "Melhor Época": int(r["train/epoch"]),
                "Acurácia": r["eval/accuracy"],
                "Precisão": r["eval/precision"],
                "Revocação": r["eval/recall"],
                "F1-Score": r["eval/f1"],
                "AUC-ROC": r["eval/auc_roc"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "best_validation_metrics.csv", index=False)

    # For LaTeX, bold the best AUC-ROC per model
    df_latex = df.copy()
    for col in ["Acurácia", "Precisão", "Revocação", "F1-Score", "AUC-ROC"]:
        df_latex[col] = df_latex[col].apply(lambda x: f"{x:.4f}")

    # Bold best overall
    best_i3d_idx = df[df["Modelo"] == "I3D"]["AUC-ROC"].idxmax()
    best_ts_idx = df[df["Modelo"] == "TimeSformer"]["AUC-ROC"].idxmax()
    for idx in [best_i3d_idx, best_ts_idx]:
        df_latex.loc[idx, "AUC-ROC"] = r"\textbf{" + df_latex.loc[idx, "AUC-ROC"] + "}"

    save_latex_table(
        df_latex,
        out_dir / "best_validation_metrics.tex",
        caption="Melhores métricas de validação por experimento (selecionadas pela maior AUC-ROC).",
        label="tab:best_val",
    )
    print(f"  ✓ Tabela de melhores métricas de validação salva")


# ──────────────────────────────────────────────────────────────────────
# 6. Curvas ROC comparativas (I3D — dados NPZ)
# ──────────────────────────────────────────────────────────────────────
def plot_i3d_roc_curves(i3d_exps: list[I3DExperiment], out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, exp in enumerate(i3d_exps):
        npz_path = exp.roc_dir / f"roc_data_epoch_{exp.best_epoch:03d}.npz"
        if not npz_path.exists():
            print(f"    ⚠ ROC NPZ não encontrado: {npz_path}")
            continue
        data = np.load(npz_path)
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_val = float(data["roc_auc"])
        ax.plot(
            fpr,
            tpr,
            label=f"{exp.short_name} (AUC={auc_val:.3f})",
            color=colors[i % len(colors)],
            linewidth=1.8,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("I3D — Curvas ROC (Melhor Época por Experimento)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.savefig(out_dir / "i3d_roc_curves.png")
    plt.close(fig)
    print(f"  ✓ Gráfico I3D ROC curves salvo")


# ──────────────────────────────────────────────────────────────────────
# 7. Gráfico comparativo AUC-ROC (barras — todos os experimentos)
# ──────────────────────────────────────────────────────────────────────
def plot_auc_comparison_bar(
    i3d_exps: list[I3DExperiment], ts_exps: list[TSExperiment], out_dir: Path
):
    names = []
    aucs = []
    model_colors = []

    for exp in i3d_exps:
        names.append(exp.short_name)
        aucs.append(exp.best_auc)
        model_colors.append("#1f77b4")

    for exp in ts_exps:
        names.append(exp.short_name)
        aucs.append(exp.best_auc)
        model_colors.append("#ff7f0e")

    # Identificar o melhor de cada modelo
    best_i3d_idx = max(range(len(i3d_exps)), key=lambda i: i3d_exps[i].best_auc)
    best_ts_idx = max(range(len(ts_exps)), key=lambda i: ts_exps[i].best_auc)
    # índices globais na lista concatenada
    best_indices = {best_i3d_idx, len(i3d_exps) + best_ts_idx}

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), aucs, color=model_colors, edgecolor="white", height=0.7)

    # Adicionar valores nas barras
    for idx, (bar, auc) in enumerate(zip(bars, aucs)):
        label = f"{auc:.3f} *" if idx in best_indices else f"{auc:.3f}"
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
            fontweight="bold" if idx in best_indices else "normal",
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUC-ROC (Validação)")
    ax.set_title("Comparação de AUC-ROC — Todos os Experimentos")
    ax.set_xlim(0, 1.08)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.3)

    # Legenda manual
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", label="I3D"),
        Patch(facecolor="#ff7f0e", label="TimeSformer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "auc_comparison_all.png")
    plt.close(fig)
    print(f"  ✓ Gráfico comparativo AUC-ROC salvo")


# ──────────────────────────────────────────────────────────────────────
# 8. Tabela de custo computacional
# ──────────────────────────────────────────────────────────────────────
I3D_ESTIMATED_MINUTES_PER_EPOCH = 4.0  # Estimativa dos logs de console


def generate_cost_table(
    i3d_exps: list[I3DExperiment], ts_exps: list[TSExperiment], out_dir: Path
):
    rows = []

    for exp in i3d_exps:
        n_epochs = len(exp.df)
        runtime_s = n_epochs * I3D_ESTIMATED_MINUTES_PER_EPOCH * 60
        gpu_hours = runtime_s / 3600
        cost = gpu_hours * GPU_COST_USD_PER_HOUR
        rows.append(
            {
                "Modelo": "I3D",
                "Experimento": exp.short_name,
                "Épocas": n_epochs,
                "Tempo (s)": f"{runtime_s:.0f}",
                "GPU-Horas": f"{gpu_hours:.2f}",
                "TFLOPS Totais": "—",
                "Custo (USD)": f"{cost:.2f}",
            }
        )

    for exp in ts_exps:
        gpu_hours = exp.total_runtime_s / 3600
        cost = gpu_hours * GPU_COST_USD_PER_HOUR
        # Get actual epochs trained
        epoch_col = exp.df["train/epoch"].dropna()
        n_epochs = int(epoch_col.iloc[-1]) if len(epoch_col) > 0 else 0
        tflops = exp.total_flos / 1e12 if exp.total_flos > 0 else 0
        rows.append(
            {
                "Modelo": "TimeSformer",
                "Experimento": exp.short_name,
                "Épocas": n_epochs,
                "Tempo (s)": f"{exp.total_runtime_s:.0f}",
                "GPU-Horas": f"{gpu_hours:.2f}",
                "TFLOPS Totais": f"{tflops:.1f}",
                "Custo (USD)": f"{cost:.2f}",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "computational_cost.csv", index=False)

    # Totais
    total_gpu_h = sum(float(r["GPU-Horas"]) for r in rows)
    total_cost = sum(float(r["Custo (USD)"]) for r in rows)
    total_row = {
        "Modelo": r"\textbf{Total}",
        "Experimento": "",
        "Épocas": "",
        "Tempo (s)": "",
        "GPU-Horas": f"\\textbf{{{total_gpu_h:.2f}}}",
        "TFLOPS Totais": "",
        "Custo (USD)": f"\\textbf{{{total_cost:.2f}}}",
    }
    df_latex = df.copy()
    df_latex.loc[len(df_latex)] = total_row

    footnote = (
        f"Custo estimado com base em RTX 5090 a US\\$ {GPU_COST_USD_PER_HOUR}/hora. "
        "Tempos do I3D estimados a partir dos logs (~4 min/época)."
    )

    save_latex_table(
        df_latex,
        out_dir / "computational_cost.tex",
        caption=f"Custo computacional estimado dos treinamentos. {footnote}",
        label="tab:cost",
    )
    print(f"  ✓ Tabela de custo computacional salva (Total: {total_gpu_h:.2f} GPU-h, US${total_cost:.2f})")


# ──────────────────────────────────────────────────────────────────────
# 9. Gráfico: GPU-Horas vs AUC-ROC (eficiência)
# ──────────────────────────────────────────────────────────────────────

# Markers atribuídos por experimento (short_name)
_I3D_MARKERS = {
    "Full / RGB": "o",
    "Full / RGB+Fluxo": "s",
    "Cabeça / RGB": "^",
    "Cabeça / RGB+Fluxo": "D",
}

_TS_MARKERS = {
    "K400/8f/Completo": "o",
    "K400/8f/Cabeça": "v",
    "K400/32f/Completo": "s",
    "K400/32f/Cabeça": "<",
    "K400/64f/Completo": "^",
    "K400/64f/Cabeça": ">",
    "SSv2/8f/Completo": "D",
    "SSv2/8f/Cabeça": "d",
    "SSv2/32f/Completo": "P",
    "SSv2/32f/Cabeça": "X",
    "SSv2/64f/Completo": "p",
    "SSv2/64f/Cabeça": "h",
}

_COLOR_I3D = "#1f77b4"
_COLOR_TS = "#ff7f0e"


def plot_efficiency_scatter(
    i3d_exps: list[I3DExperiment], ts_exps: list[TSExperiment], out_dir: Path
):
    fig, ax = plt.subplots(figsize=(10, 7))

    from matplotlib.lines import Line2D

    legend_handles: list[Line2D] = []

    # I3D
    for exp in i3d_exps:
        n_epochs = len(exp.df)
        gpu_h = (n_epochs * I3D_ESTIMATED_MINUTES_PER_EPOCH * 60) / 3600
        marker = _I3D_MARKERS.get(exp.short_name, "o")
        ax.scatter(
            gpu_h, exp.best_auc, color=_COLOR_I3D, s=100, zorder=5,
            edgecolors="white", marker=marker, linewidths=0.8,
        )
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=_COLOR_I3D,
                   markersize=8, label=f"I3D — {exp.short_name}")
        )

    # TimeSformer
    for exp in ts_exps:
        gpu_h = exp.total_runtime_s / 3600
        marker = _TS_MARKERS.get(exp.short_name, "o")
        ax.scatter(
            gpu_h, exp.best_auc, color=_COLOR_TS, s=100, zorder=5,
            edgecolors="white", marker=marker, linewidths=0.8,
        )
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=_COLOR_TS,
                   markersize=8, label=f"TS — {exp.short_name}")
        )

    ax.set_xlabel("GPU-Horas")
    ax.set_ylabel("Melhor AUC-ROC (Validação)")
    ax.set_title("Eficiência: Custo Computacional vs. Desempenho")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)

    ax.legend(
        handles=legend_handles, fontsize=7, loc="center left",
        bbox_to_anchor=(1.02, 0.5), borderaxespad=0, frameon=True,
        ncol=1,
    )

    fig.savefig(out_dir / "efficiency_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gráfico de eficiência custo×desempenho salvo")


import json

# ──────────────────────────────────────────────────────────────────────
# 10. Tabela e gráfico: Métricas no conjunto de teste
# ──────────────────────────────────────────────────────────────────────
def generate_test_metrics_table(
    i3d_exps: list[I3DExperiment],
    ts_exps: list[TSExperiment],
    results_dir: Path,
    out_dir: Path,
):
    """Gera tabela com métricas de teste a partir dos test_metrics.json."""
    rows = []

    def _f2(p, r):
        """Calcula F2-Score: prioriza revocação (beta=2)."""
        return (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0

    for exp in i3d_exps:
        json_path = results_dir / "i3d" / exp.name / "test_evaluation" / "test_metrics.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            m = json.load(f)
        p, r = m.get("precision", 0), m.get("recall", 0)
        rows.append(
            {
                "Modelo": "I3D",
                "Experimento": exp.short_name,
                "Precisão": p,
                "Revocação": r,
                "F2-Score": _f2(p, r),
            }
        )

    for exp in ts_exps:
        json_path = results_dir / "timesformer" / exp.name / "test_evaluation" / "test_metrics.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            m = json.load(f)
        p, r = m.get("precision", 0), m.get("recall", 0)
        rows.append(
            {
                "Modelo": "TimeSformer",
                "Experimento": exp.short_name,
                "Precisão": p,
                "Revocação": r,
                "F2-Score": _f2(p, r),
            }
        )

    if not rows:
        print("  ⚠ Nenhum test_metrics.json encontrado — pule a avaliação ou execute os scripts de avaliação.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_metrics.csv", index=False)

    df_latex = df.copy()
    for col in ["Precisão", "Revocação", "F2-Score"]:
        df_latex[col] = df_latex[col].apply(lambda x: f"{x:.4f}")

    save_latex_table(
        df_latex,
        out_dir / "test_metrics.tex",
        caption="Métricas no conjunto de teste dos melhores modelos (Precisão, Revocação e F2-Score).",
        label="tab:test_metrics",
    )
    print(f"  ✓ Tabela de métricas de teste salva ({len(rows)} experimentos)")

    # Gráfico de barras agrupadas
    if len(rows) >= 2:
        _plot_test_comparison_bar(df, out_dir)


def _plot_test_comparison_bar(df: pd.DataFrame, out_dir: Path):
    metrics = ["Precisão", "Revocação", "F2-Score"]
    labels = df["Modelo"] + " — " + df["Experimento"]
    x = np.arange(len(metrics))
    width = 0.8 / len(df)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=labels.iloc[i], color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Valor")
    ax.set_title("Métricas no Conjunto de Teste — Melhores Modelos")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "test_metrics_comparison.png")
    plt.close(fig)
    print(f"  ✓ Gráfico comparativo de métricas de teste salvo")


def plot_test_roc_curves(
    i3d_exps: list[I3DExperiment],
    ts_exps: list[TSExperiment],
    results_dir: Path,
    out_dir: Path,
):
    """Plota curvas ROC no conjunto de teste para os modelos avaliados."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    idx = 0
    found = False

    for exp in i3d_exps:
        npz_path = results_dir / "i3d" / exp.name / "test_evaluation" / "roc_curve_test.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        fpr, tpr = data["fpr"], data["tpr"]
        auc_val = float(data["roc_auc"]) if "roc_auc" in data else float(data.get("auc", 0))
        ax.plot(fpr, tpr, label=f"I3D {exp.short_name} (AUC={auc_val:.3f})",
                color=colors[idx % len(colors)], linewidth=1.8)
        idx += 1
        found = True

    for exp in ts_exps:
        npz_path = results_dir / "timesformer" / exp.name / "test_evaluation" / "roc_curve_test.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        fpr, tpr = data["fpr"], data["tpr"]
        auc_val = float(data["roc_auc"]) if "roc_auc" in data else float(data.get("auc", 0))
        ax.plot(fpr, tpr, label=f"TS {exp.short_name} (AUC={auc_val:.3f})",
                color=colors[idx % len(colors)], linewidth=1.8)
        idx += 1
        found = True

    if not found:
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC no Conjunto de Teste")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.savefig(out_dir / "test_roc_curves.png")
    plt.close(fig)
    print(f"  ✓ Gráfico ROC curves de teste salvo")


# ──────────────────────────────────────────────────────────────────────
# 11. Matrizes de confusão (heatmap) — conjunto de teste
# ──────────────────────────────────────────────────────────────────────
def plot_test_confusion_matrices(
    i3d_exps: list[I3DExperiment],
    ts_exps: list[TSExperiment],
    results_dir: Path,
    out_dir: Path,
):
    """Plota heatmaps de matrizes de confusão no conjunto de teste."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as cm_func

    panels: list[tuple[str, np.ndarray]] = []  # (label, cm)

    for exp in i3d_exps:
        npz_path = results_dir / "i3d" / exp.name / "test_evaluation" / "test_predictions.npz"
        json_path = results_dir / "i3d" / exp.name / "test_evaluation" / "test_metrics.json"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        labels = data["labels"]
        probs = data["probs"]
        threshold = float(data["threshold"])
        preds = (probs >= threshold).astype(int)
        matrix = cm_func(labels, preds)
        panels.append((f"I3D — {exp.short_name}", matrix))

    for exp in ts_exps:
        npz_path = results_dir / "timesformer" / exp.name / "test_evaluation" / "test_predictions.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        labels = data["labels"]
        preds = data["preds"]
        matrix = cm_func(labels, preds)
        panels.append((f"TS — {exp.short_name}", matrix))

    if not panels:
        print("  ⚠ Nenhum test_predictions.npz encontrado para matrizes de confusão.")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    class_names = ["Normal", "Shoplifting"]
    for ax, (title, matrix) in zip(axes, panels):
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, annot_kws={"size": 16},
            linewidths=0.5, linecolor="gray",
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(title, fontsize=11)

    fig.suptitle("Matrizes de Confusão — Conjunto de Teste", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "test_confusion_matrices.png")
    plt.close(fig)
    print(f"  ✓ Matrizes de confusão de teste salvas ({n} modelos)")


# ──────────────────────────────────────────────────────────────────────
# 12. Tabela + gráfico: Throughput de inferência (I3D + TimeSformer)
# ──────────────────────────────────────────────────────────────────────

# Throughput medido via benchmark unificado (RTX 5090, batch_size=1,
# 20 warmup + 100 iterações cronometradas, entrada sintética).
# Gerado por scripts/benchmark_i3d_throughput.py → results/figures/throughput_benchmark.json
_THROUGHPUT_JSON = Path(__file__).resolve().parent.parent / "results" / "figures" / "throughput_benchmark.json"


def generate_inference_throughput(
    ts_exps: list[TSExperiment], out_dir: Path
):
    """Gera tabela e gráfico de throughput (amostras/s) para os modelos vencedores."""

    if not _THROUGHPUT_JSON.exists():
        print("  ⚠ Arquivo de benchmark não encontrado. Execute: uv run python scripts/benchmark_i3d_throughput.py")
        return

    with open(_THROUGHPUT_JSON) as f:
        bench = json.load(f)

    rows = []
    for entry in bench:
        label = entry["model"]
        if "I3D" in label:
            modelo = "I3D"
        else:
            modelo = "TimeSformer"
        rows.append({
            "Modelo": modelo,
            "Experimento": label,
            "Amostras/s": entry["samples_per_second"],
            "ms/amostra": entry["ms_per_sample"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "inference_throughput.csv", index=False)
    save_latex_table(
        df,
        out_dir / "inference_throughput.tex",
        caption=(
            "Throughput de inferência dos melhores modelos I3D e TimeSformer "
            "(RTX 5090, batch\\_size=1, entrada sintética)."
        ),
        label="tab:throughput",
    )

    # Gráfico de barras horizontais
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#2ca02c", "#1f77b4"]  # I3D verde, TS azul
    sps_vals = [r["Amostras/s"] for r in rows]
    ms_vals = [r["ms/amostra"] for r in rows]

    bars = ax.barh(
        range(len(rows)), sps_vals,
        color=colors[:len(rows)], edgecolor="white", height=0.5,
    )
    for bar, sps, ms in zip(bars, sps_vals, ms_vals):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{sps:.2f} amostras/s  ({ms:.1f} ms)',
            va="center", fontsize=9,
        )

    ylabels = [r["Experimento"] for r in rows]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Amostras por Segundo (batch_size=1)")
    ax.set_title("Throughput de Inferência — Melhores Modelos (RTX 5090)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, max(sps_vals) * 1.35)

    fig.tight_layout()
    fig.savefig(out_dir / "inference_throughput.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Tabela e gráfico de throughput de inferência salvos")


# ──────────────────────────────────────────────────────────────────────
# 13. Tabela + gráfico: Parâmetros treináveis (por variante de frames)
# ──────────────────────────────────────────────────────────────────────

# Valores pré-computados via scripts/count_params.py
# TS "somente cabeça" varia por frame count: 8f = só head, 32f/64f = head + time_embeddings
_PARAM_COUNTS = [
    # (label, total, trainable_full, trainable_head)
    ("I3D",           12_288_289,  12_288_289,   1_025),
    ("TS 8f",        121_260_290, 121_260_290,   1_538),
    ("TS 32f",       121_278_722, 121_278_722,  26_114),
    ("TS 64f",       121_303_298, 121_303_298,  50_690),
]


def generate_trainable_params_table(out_dir: Path):
    """Gera tabela e gráfico comparando parâmetros totais vs treináveis."""
    rows = []
    for label, total, trainable_full, trainable_head in _PARAM_COUNTS:
        for strategy, trainable in [("Completo", trainable_full), ("Somente cabeça", trainable_head)]:
            frozen = total - trainable
            pct = trainable / total * 100
            rows.append({
                "Modelo": label,
                "Fine-tuning": strategy,
                "Parâmetros Totais": f"{total:,}",
                "Treináveis": f"{trainable:,}",
                "Congelados": f"{frozen:,}",
                "% Treináveis": f"{pct:.2f}\\%",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "trainable_params.csv", index=False)
    save_latex_table(
        df,
        out_dir / "trainable_params.tex",
        caption=(
            "Comparação de parâmetros totais e treináveis por estratégia de fine-tuning. "
            "Para TimeSformer com mais de 8 frames, os \\textit{time\\_embeddings} interpolados "
            "também são descongelados no modo ``somente cabeça''."
        ),
        label="tab:params",
    )

    # ── Gráfico: duas visualizações lado a lado ──
    fig, (ax_full, ax_head) = plt.subplots(1, 2, figsize=(12, 5))

    labels_short = [row[0] for row in _PARAM_COUNTS]
    totals = np.array([row[1] for row in _PARAM_COUNTS])
    full_trainable = np.array([row[2] for row in _PARAM_COUNTS])
    head_trainable = np.array([row[3] for row in _PARAM_COUNTS])

    x = np.arange(len(labels_short))
    bar_width = 0.55

    # --- Painel esquerdo: Fine-tuning completo (todos treináveis) ---
    bars = ax_full.bar(x, totals / 1e6, width=bar_width, color="#2ca02c", label="Treináveis")
    for bar, val in zip(bars, totals):
        ax_full.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val/1e6:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_full.set_xticks(x)
    ax_full.set_xticklabels(labels_short, fontsize=9)
    ax_full.set_ylabel("Milhões de Parâmetros")
    ax_full.set_title("Fine-tuning Completo\n(todos treináveis)", fontsize=10)
    ax_full.set_ylim(0, totals.max() / 1e6 * 1.18)
    ax_full.grid(True, axis="y", alpha=0.3)

    # --- Painel direito: Somente cabeça (poucos treináveis, maioria congelados) ---
    frozen_head = totals - head_trainable
    bars_t = ax_head.bar(x, head_trainable / 1e6, width=bar_width, color="#2ca02c", label="Treináveis")
    bars_f = ax_head.bar(x, frozen_head / 1e6, width=bar_width, bottom=head_trainable / 1e6,
                         color="#d62728", alpha=0.5, label="Congelados")

    for bar_t, ht in zip(bars_t, head_trainable):
        lbl = f"{ht/1e6:.1f}M" if ht >= 1_000_000 else f"{ht/1e3:.1f}K" if ht >= 1000 else str(ht)
        total_h = bar_t.get_height() + (frozen_head[list(head_trainable).index(ht)] / 1e6)
        ax_head.text(
            bar_t.get_x() + bar_t.get_width() / 2, total_h + 1,
            lbl, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_head.set_xticks(x)
    ax_head.set_xticklabels(labels_short, fontsize=9)
    ax_head.set_ylabel("Milhões de Parâmetros")
    ax_head.set_title("Somente Cabeça\n(treináveis em destaque)", fontsize=10)
    ax_head.set_ylim(0, totals.max() / 1e6 * 1.18)
    ax_head.legend(fontsize=8)
    ax_head.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Parâmetros Treináveis vs. Congelados", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "trainable_params.png")
    plt.close(fig)
    print(f"  ✓ Tabela e gráfico de parâmetros treináveis salvos")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Gera tabelas e gráficos para a seção de resultados do TCC."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Diretório raiz dos resultados.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Diretório de saída para figuras e tabelas.",
    )
    args = parser.parse_args()

    # Resolve paths relative to CWD (project root)
    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gerando artefatos para seção de resultados do TCC")
    print(f"  Config:     {config_path}")
    print(f"  Resultados: {results_dir}")
    print(f"  Saída:      {out_dir}")
    print("=" * 60)

    # Carregar config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Carregar dados
    print("\n📂 Carregando dados dos experimentos...")
    i3d_exps = load_i3d_experiments(results_dir)
    ts_exps = load_ts_experiments(results_dir)
    print(f"  I3D: {len(i3d_exps)} experimentos")
    print(f"  TimeSformer: {len(ts_exps)} experimentos")

    # Gerar artefatos
    print("\n📊 Gerando tabelas...")
    generate_hyperparams_table(cfg, out_dir)
    generate_experiment_variants_table(i3d_exps, ts_exps, out_dir)
    generate_best_validation_table(i3d_exps, ts_exps, out_dir)
    generate_cost_table(i3d_exps, ts_exps, out_dir)

    print("\n📈 Gerando gráficos...")
    plot_i3d_training_loss(i3d_exps, out_dir)
    plot_ts_training_loss(ts_exps, out_dir)
    plot_i3d_validation_metrics(i3d_exps, out_dir)
    plot_ts_validation_metrics(ts_exps, out_dir)
    plot_i3d_roc_curves(i3d_exps, out_dir)
    plot_auc_comparison_bar(i3d_exps, ts_exps, out_dir)
    plot_efficiency_scatter(i3d_exps, ts_exps, out_dir)

    # Métricas no conjunto de teste (se disponíveis)
    print("\n🧪 Métricas no conjunto de teste...")
    generate_test_metrics_table(i3d_exps, ts_exps, results_dir, out_dir)
    plot_test_roc_curves(i3d_exps, ts_exps, results_dir, out_dir)
    plot_test_confusion_matrices(i3d_exps, ts_exps, results_dir, out_dir)

    # Análises adicionais
    print("\n🔬 Análises complementares...")
    generate_inference_throughput(ts_exps, out_dir)
    generate_trainable_params_table(out_dir)

    print("\n" + "=" * 60)
    print(f"✅ Todos os artefatos salvos em: {out_dir}/")
    print("=" * 60)

    # Resumir arquivos gerados
    generated = sorted(out_dir.iterdir())
    print(f"\nArquivos gerados ({len(generated)}):")
    for f in generated:
        size = f.stat().st_size
        print(f"  {f.name:45s} {size:>8,} bytes")


if __name__ == "__main__":
    main()
