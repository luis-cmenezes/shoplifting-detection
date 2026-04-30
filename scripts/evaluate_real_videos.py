#!/usr/bin/env python3
"""
Avaliação de vídeos reais com janela deslizante (Sliding Window Inference).

Simula um buffer de câmera de segurança em tempo real:

  - Os frames do vídeo são reamostrados para TARGET_FPS (25 fps) e
    redimensionados para 224x224, replicando exatamente o pré-processamento
    padrão usado no TimeSformer e I3D.
  - Uma janela desliza pelo vídeo com passo STRIDE (8 quadros).
    A sobreposição garante que ações de furto não sejam cortadas entre janelas.
  - Cada posição da janela gera uma probabilidade P(Furto) bruta.
  - Uma Média Móvel Simples (SMA) sobre as últimas SMOOTHING_N janelas suaviza
    picos isolados (falsos alarmes).
  - O sistema sinaliza "SUSPEITO" apenas quando a probabilidade suavizada
    permanece acima do limiar por pelo menos MIN_SUSTAINED_WINDOWS janelas
    consecutivas.

Modelos utilizados (melhores pesos de cada arquitetura):
  I3D       → aug_full_unfreeze_rgb_only  (AUC-ROC = 0.9639, threshold = 0.7449)
  TimeSformer → timesformer-base-finetuned-ssv2_frames32_unfreeze_all
                                          (AUC-ROC = 0.9840, threshold = 0.50)

Saídas (por vídeo, em results/real_videos/<nome>/):
  predictions.json          — predições brutas, suavizadas e timestamps
  i3d_analysis.png          — curva de probabilidade I3D
  timesformer_analysis.png  — curva de probabilidade TimeSformer
  combined_analysis.png     — comparação lado a lado (para o relatório PDF)
  i3d_overlay.mp4           — vídeo com overlay da predição I3D
  timesformer_overlay.mp4   — vídeo com overlay da predição TimeSformer

Uso:
  uv run scripts/evaluate_real_videos.py
  uv run scripts/evaluate_real_videos.py --videos datasets/real/furto-celular.mp4
  uv run scripts/evaluate_real_videos.py --no-video   # apenas plots estáticos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")

from i3d_shoplifting.inference.evaluate import load_i3d_model
from timesformer_shoplifting.inference.evaluate import load_model_and_processor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constantes globais
# ---------------------------------------------------------------------------
TARGET_FPS: int = 25          # FPS alvo (mesmo do TimeSformer no treino)
I3D_WINDOW: int = 64          # janela I3D (mesmo do treino)
TS_WINDOW: int = 32           # janela TimeSformer (mesmo do treino)
STRIDE: int = 8               # passo da janela deslizante

# Experimentos campeões
I3D_CHECKPOINT = PROJECT_ROOT / "results/i3d/aug_full_unfreeze_rgb_only/model_weights/epoch_010_auc_0.9660.pt"
I3D_THRESHOLD = 0.7449        # threshold calibrado no conjunto de validação

TS_MODEL_DIR = PROJECT_ROOT / "results/timesformer/timesformer-base-finetuned-ssv2_frames32_unfreeze_all/final_model"
TS_THRESHOLD = 0.50           # P(Shoplifting) > 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paleta de cores para os plots
COLOR_I3D = "#E64646"          # vermelho
COLOR_TS = "#2E86AB"           # azul
COLOR_RAW_ALPHA = 0.35
COLOR_SMOOTH = 0.9
COLOR_ALERT_BG = "#FFEBEB"

# Configuração global de matplotlib (estilo acadêmico)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


# ===========================================================================
# 1. Leitura e reamostragem de vídeo
# ===========================================================================

def read_video_at_fps(
    video_path: Path,
    target_fps: int = TARGET_FPS,
    resize_hw: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, float, int]:
    """Lê vídeo reamostrado para *target_fps*, redimensionando cada frame.

    Estratégia de reamostragem:
      Para cada frame alvo *i_t*, o índice de frame fonte é calculado como
      ``i_s = round(i_t * src_fps / target_fps)``.  Isso dá uma amostragem
      uniforme do vídeo sem duplicação de frames quando src_fps ≈ target_fps
      e sem dropping excessivo quando src_fps < target_fps.

    Returns:
        frames   : uint8 numpy array (N, H, W, 3) em RGB
        src_fps  : FPS original do vídeo
        src_total: número de frames no vídeo original
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

    src_fps: float = cap.get(cv2.CAP_PROP_FPS)
    src_total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0 or src_total <= 0:
        cap.release()
        raise ValueError(f"Metadados inválidos para: {video_path}")

    duration = src_total / src_fps
    n_target = int(duration * target_fps)

    # Índices dos frames fonte para cada frame alvo
    src_indices = np.round(
        np.arange(n_target) * (src_fps / target_fps)
    ).astype(int)
    src_indices = np.clip(src_indices, 0, src_total - 1)

    target_h, target_w = resize_hw
    frames: list[np.ndarray] = []
    last_src_idx = -1
    last_frame: np.ndarray | None = None

    def _resize_and_crop(img: np.ndarray) -> np.ndarray:
        """Redimensiona mantendo aspect ratio (lado maior → target) e corta centralizado."""
        ih, iw = img.shape[:2]
        scale = target_h / max(ih, iw)
        new_h, new_w = round(ih * scale), round(iw * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Crop centralizado
        y0 = (new_h - target_h) // 2
        x0 = (new_w - target_w) // 2
        return resized[y0 : y0 + target_h, x0 : x0 + target_w]

    for src_idx in src_indices:
        if src_idx != last_src_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last_frame = _resize_and_crop(rgb)
            last_src_idx = src_idx
        frames.append(last_frame.copy())

    cap.release()

    if not frames:
        raise ValueError(f"Nenhum frame extraído de: {video_path}")

    return np.stack(frames, axis=0), src_fps, src_total


# ===========================================================================
# 2. Carregamento dos modelos - delegado a inference/evaluate.py de cada pacote
# ===========================================================================


# ===========================================================================
# 3. Inferência por janela
# ===========================================================================

@torch.no_grad()
def i3d_infer(model: torch.nn.Module, window: np.ndarray) -> float:
    """Inferência I3D em uma janela RGB.

    Args:
        window: (T, H, W, C) uint8 RGB — T == I3D_WINDOW
    Returns:
        prob: P(Shoplifting) ∈ [0, 1]
    """
    x = torch.from_numpy(window).float() / 255.0  # (T, H, W, C)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)
    logit = model(x)  # (1, 1)
    return torch.sigmoid(logit).item()


@torch.no_grad()
def ts_infer(model, processor, window: np.ndarray) -> float:
    """Inferência TimeSformer em uma janela RGB.

    O AutoImageProcessor aplica resize, center-crop e normalização ImageNet
    idênticos ao treino.

    Args:
        window: (T, H, W, C) uint8 RGB — T == TS_WINDOW
    Returns:
        prob: P(Shoplifting) ∈ [0, 1]
    """
    # Converte para lista de tensores (C, H, W) uint8 — formato aceito pelo processor
    frames_list = [
        torch.from_numpy(window[i]).permute(2, 0, 1)
        for i in range(len(window))
    ]
    inputs = processor(frames_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)  # (1, T, C, H, W)
    outputs = model(pixel_values=pixel_values)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0, 1].item()  # P(Shoplifting)


# ===========================================================================
# 4. Janela deslizante + suavização temporal
# ===========================================================================
# 4. Janela deslizante + agregação por quadro
# ===========================================================================

def frame_level_average(
    n_frames: int,
    window_starts: list[int],
    window_size: int,
    window_probs: np.ndarray,
) -> np.ndarray:
    """Calcula a probabilidade média de cada quadro.

    Cada quadro recebe a média das predições de todas as janelas que o
    contêm.  Um quadro *f* pertence à janela com início *s* se
    ``s <= f < s + window_size``.

    Com stride uniforme, quadros interiores aparecem em exatamente
    ``window_size // stride`` janelas; quadros próximos às bordas aparecem
    em menos janelas, mas a média ainda é bem definida.
    """
    acc = np.zeros(n_frames, dtype=float)
    cnt = np.zeros(n_frames, dtype=int)
    for s, p in zip(window_starts, window_probs):
        acc[s : s + window_size] += p
        cnt[s : s + window_size] += 1
    # Quadros sem nenhuma janela (não deve ocorrer) ficam em 0
    valid = cnt > 0
    return np.where(valid, acc / np.where(valid, cnt, 1), 0.0)


def sliding_window_inference(
    frames: np.ndarray,
    infer_fn,
    window_size: int,
    stride: int,
    threshold: float,
    desc: str = "Inferência",
) -> dict:
    """Executa a inferência com janela deslizante e agrega por quadro.

    Para cada quadro, a probabilidade final é a média das predições de
    todas as janelas em que ele aparece (frame-level aggregation).

    Returns:
        dict com:
          window_starts    : índice inicial de cada janela
          window_probs     : probabilidade bruta por janela
          frame_probs      : probabilidade agregada por quadro (N_frames valores)
          frame_timestamps : tempo (s) de cada quadro
          alert_mask       : True nos quadros acima do limiar
    """
    n_frames = len(frames)
    starts = list(range(0, n_frames - window_size + 1, stride))

    if not starts:
        raise ValueError(
            f"Vídeo muito curto ({n_frames} frames) para janela de {window_size}."
        )

    window_probs: list[float] = []
    for start in tqdm(starts, desc=desc, leave=False):
        window = frames[start : start + window_size]
        window_probs.append(infer_fn(window))

    win_probs_arr = np.array(window_probs, dtype=float)
    frame_probs = frame_level_average(n_frames, starts, window_size, win_probs_arr)
    timestamps = np.arange(n_frames) / TARGET_FPS
    alert = frame_probs > threshold

    return {
        "window_starts": starts,
        "window_probs": win_probs_arr.tolist(),
        "frame_probs": frame_probs.tolist(),
        "frame_timestamps": timestamps.tolist(),
        "alert_mask": alert.tolist(),
    }


# ===========================================================================
# 5. Visualizações estáticas (PNG para PDF)
# ===========================================================================

def _plot_single_model(
    ax: plt.Axes,
    results: dict,
    threshold: float,
    model_name: str,
    color: str,
    video_name: str,
) -> None:
    """Plota a curva de probabilidade por quadro de um único modelo."""
    t = np.array(results["frame_timestamps"])
    fp = np.array(results["frame_probs"])
    alert = np.array(results["alert_mask"])

    # Regiões de alerta (fundo vermelho suave)
    in_alert = False
    alert_start = None
    for i, a in enumerate(alert):
        if a and not in_alert:
            alert_start = t[i]
            in_alert = True
        elif not a and in_alert:
            ax.axvspan(alert_start, t[i], alpha=0.15, color="red", zorder=0)
            in_alert = False
    if in_alert:
        ax.axvspan(alert_start, t[-1], alpha=0.15, color="red", zorder=0)

    # Curva de probabilidade por quadro
    ax.plot(
        t, fp,
        color=color, linewidth=1.8,
        label="P(Furto) por quadro (média das janelas)",
    )

    # Linha do limiar
    ax.axhline(
        threshold, color="black", linewidth=1.2,
        linestyle="--", label=f"Limiar ({threshold:.2f})",
        zorder=3,
    )

    # Formatação
    ax.set_xlim(0, t[-1] + 0.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("P(Furto)")
    ax.set_title(f"{model_name} — {video_name}")
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.1))


def save_single_model_plot(
    results: dict,
    threshold: float,
    model_name: str,
    color: str,
    video_name: str,
    output_path: Path,
) -> None:
    """Gera e salva o gráfico de probabilidade para um único modelo."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_single_model(ax, results, threshold, model_name, color, video_name)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → Plot salvo: {output_path.name}")


def save_combined_plot(
    i3d_results: dict,
    ts_results: dict,
    video_name: str,
    output_path: Path,
) -> None:
    """Gera comparação lado a lado dos dois modelos (para o PDF)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    _plot_single_model(
        axes[0], i3d_results, I3D_THRESHOLD,
        f"I3D (janela={I3D_WINDOW}q, stride={STRIDE}q)", COLOR_I3D, video_name,
    )
    _plot_single_model(
        axes[1], ts_results, TS_THRESHOLD,
        f"TimeSformer (janela={TS_WINDOW}q, stride={STRIDE}q)", COLOR_TS, video_name,
    )

    fig.suptitle(
        f"Análise de Furto em Tempo Real — {video_name}\n"
        f"Janela deslizante | Stride={STRIDE} quadros | "
        f"Agregação por quadro (média das janelas sobrepostas)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot combinado: {output_path.name}")


# ===========================================================================
# 6. Vídeo overlay
# ===========================================================================

def _draw_overlay(
    frame: np.ndarray,       # BGR, qualquer resolução
    prob: float,
    is_alert: bool,
    threshold: float,
) -> np.ndarray:
    """Desenha informações de inferência sobre um frame BGR.

    Layout:
      - Barra vertical de probabilidade na esquerda (verde→vermelho)
      - Rótulo "SUSPEITO" / "NORMAL" no canto superior direito
    """
    out = frame.copy()
    H, W = out.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Barra vertical de probabilidade (esquerda) ──────────────────
    bar_w = max(18, W // 32)
    bar_x0, bar_x1 = 8, 8 + bar_w
    bar_y0, bar_y1 = 8, H - 8
    bar_inner_h = bar_y1 - bar_y0

    # Fundo da barra
    cv2.rectangle(out, (bar_x0, bar_y0), (bar_x1, bar_y1), (40, 40, 40), -1)

    # Preenchimento de baixo para cima, proporcional a prob
    fill_h = int(prob * bar_inner_h)
    fill_y0 = bar_y1 - fill_h
    r = int(prob * 220)
    g = int((1.0 - prob) * 200)
    bar_color = (0, g, r)  # BGR: verde → vermelho
    if fill_h > 0:
        cv2.rectangle(out, (bar_x0, fill_y0), (bar_x1, bar_y1), bar_color, -1)

    # Linha do limiar na barra
    thr_y = bar_y1 - int(threshold * bar_inner_h)
    cv2.line(out, (bar_x0, thr_y), (bar_x1, thr_y), (200, 200, 0), 2)

    # Número de probabilidade logo acima do preenchimento
    fs = max(0.35, W / 1800)
    prob_text = f"{prob:.2f}"
    (tw, th), _ = cv2.getTextSize(prob_text, font, fs, 1)
    text_y = max(fill_y0 - 4, bar_y0 + th + 2)
    text_y = min(text_y, bar_y1 - 2)
    cv2.putText(out, prob_text, (bar_x0, text_y),
                font, fs, (230, 230, 230), 1, cv2.LINE_AA)

    # ── Rótulo SUSPEITO/NORMAL (canto superior direito) ─────────────
    status_txt = "SUSPEITO" if is_alert else "NORMAL"
    status_color = (0, 0, 220) if is_alert else (0, 180, 0)  # BGR
    status_fs = max(0.6, W / 900)
    (sw, sh), _ = cv2.getTextSize(status_txt, font, status_fs, 2)
    cv2.rectangle(out, (W - sw - 20, 8), (W - 8, 8 + sh + 10), (20, 20, 20), -1)
    cv2.putText(out, status_txt, (W - sw - 14, 8 + sh + 2),
                font, status_fs, status_color, 2, cv2.LINE_AA)

    return out


def generate_overlay_video(
    video_path: Path,
    frames: np.ndarray,          # (N, 224, 224, 3) RGB a 25 fps
    results: dict,
    threshold: float,
    model_label: str,
    output_path: Path,
) -> None:
    """Gera vídeo com overlay das predições.

    Usa os frames já reamostrados a 25 fps para manter sincronismo com
    as predições.  Escala de volta à resolução original para qualidade visual.
    """
    # Lê dimensões originais do vídeo
    cap_orig = cv2.VideoCapture(str(video_path))
    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_orig.release()

    out_w, out_h = max(orig_w, 480), max(orig_h, 270)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, (out_w, out_h))

    frame_probs = np.array(results["frame_probs"])
    alert_mask = results["alert_mask"]
    n_windows = len(results["window_starts"])

    for frame_idx, rgb_frame in enumerate(
        tqdm(frames, desc=f"  Overlay {model_label}", leave=False)
    ):
        prob = frame_probs[frame_idx]
        is_alert = bool(alert_mask[frame_idx])

        # Converte RGB → BGR para OpenCV
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        # Escala para resolução de saída
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        bgr = _draw_overlay(
            bgr,
            prob=prob,
            is_alert=is_alert,
            threshold=threshold,
        )
        writer.write(bgr)

    writer.release()
    print(f"  → Vídeo overlay: {output_path.name}")


# ===========================================================================
# 7. Pipeline principal por vídeo
# ===========================================================================

def process_video(
    video_path: Path,
    i3d_model: torch.nn.Module,
    ts_model,
    ts_processor,
    output_root: Path,
    generate_video: bool = True,
) -> None:
    """Executa o pipeline completo de avaliação para um único vídeo."""
    video_name = video_path.stem
    out_dir = output_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processando: {video_path.name}")
    print(f"{'='*60}")

    # ── 1. Leitura do vídeo ──────────────────────────────────────────
    print(f"  Lendo vídeo e reamostando para {TARGET_FPS} fps (224x224)...")
    frames, src_fps, src_total = read_video_at_fps(video_path, TARGET_FPS)
    n_frames = len(frames)
    duration = n_frames / TARGET_FPS
    print(f"  FPS original: {src_fps:.2f} | Frames originais: {src_total}")
    print(f"  Frames a {TARGET_FPS} fps: {n_frames} ({duration:.1f}s)")

    # ── 2. Sliding window I3D ────────────────────────────────────────
    if n_frames < I3D_WINDOW:
        print(f"  AVISO: vídeo ({n_frames} frames) menor que janela I3D ({I3D_WINDOW}). "
              "Pulando I3D.")
        i3d_results = None
    else:
        print(f"\n  [I3D] Janela={I3D_WINDOW} | Stride={STRIDE} | agregação por quadro")
        i3d_results = sliding_window_inference(
            frames,
            infer_fn=lambda w: i3d_infer(i3d_model, w),
            window_size=I3D_WINDOW,
            stride=STRIDE,
            threshold=I3D_THRESHOLD,
            desc="  I3D",
        )
        n_alerts_i3d = int(np.sum(i3d_results["alert_mask"]))
        print(f"  I3D: {len(i3d_results['window_starts'])} janelas | "
              f"Pico por quadro: {max(i3d_results['frame_probs']):.3f} | "
              f"Quadros em alerta: {n_alerts_i3d}")

    # ── 3. Sliding window TimeSformer ────────────────────────────────
    if n_frames < TS_WINDOW:
        print(f"  AVISO: vídeo ({n_frames} frames) menor que janela TS ({TS_WINDOW}). "
              "Pulando TimeSformer.")
        ts_results = None
    else:
        print(f"\n  [TS] Janela={TS_WINDOW} | Stride={STRIDE} | agregação por quadro")
        ts_results = sliding_window_inference(
            frames,
            infer_fn=lambda w: ts_infer(ts_model, ts_processor, w),
            window_size=TS_WINDOW,
            stride=STRIDE,
            threshold=TS_THRESHOLD,
            desc="  TimeSformer",
        )
        n_alerts_ts = int(np.sum(ts_results["alert_mask"]))
        print(f"  TS:  {len(ts_results['window_starts'])} janelas | "
              f"Pico por quadro: {max(ts_results['frame_probs']):.3f} | "
              f"Quadros em alerta: {n_alerts_ts}")

    # ── 4. Salva predições JSON ──────────────────────────────────────
    json_path = out_dir / "predictions.json"
    payload = {
        "video": video_path.name,
        "src_fps": src_fps,
        "src_total_frames": src_total,
        "target_fps": TARGET_FPS,
        "target_frames": n_frames,
        "duration_s": duration,
        "config": {
            "stride": STRIDE,
            "aggregation": "frame_level_average",
        },
        "i3d": {
            "experiment": I3D_CHECKPOINT.parent.parent.name,
            "window_size": I3D_WINDOW,
            "threshold": I3D_THRESHOLD,
            **(i3d_results or {}),
        },
        "timesformer": {
            "experiment": TS_MODEL_DIR.parent.name,
            "window_size": TS_WINDOW,
            "threshold": TS_THRESHOLD,
            **(ts_results or {}),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  → Predições: {json_path.name}")

    # ── 5. Plots estáticos ───────────────────────────────────────────
    if i3d_results:
        save_single_model_plot(
            i3d_results, I3D_THRESHOLD,
            f"I3D (janela={I3D_WINDOW}q, stride={STRIDE}q)",
            COLOR_I3D, video_name,
            out_dir / "i3d_analysis.png",
        )

    if ts_results:
        save_single_model_plot(
            ts_results, TS_THRESHOLD,
            f"TimeSformer (janela={TS_WINDOW}q, stride={STRIDE}q)",
            COLOR_TS, video_name,
            out_dir / "timesformer_analysis.png",
        )

    if i3d_results and ts_results:
        save_combined_plot(
            i3d_results, ts_results,
            video_name,
            out_dir / "combined_analysis.png",
        )

    # ── 6. Vídeos overlay ────────────────────────────────────────────
    if generate_video:
        if i3d_results:
            generate_overlay_video(
                video_path, frames, i3d_results,
                I3D_THRESHOLD,
                f"I3D | thr={I3D_THRESHOLD:.2f}",
                out_dir / "i3d_overlay.mp4",
            )
        if ts_results:
            generate_overlay_video(
                video_path, frames, ts_results,
                TS_THRESHOLD,
                f"TimeSformer | thr={TS_THRESHOLD:.2f}",
                out_dir / "timesformer_overlay.mp4",
            )


# ===========================================================================
# 8. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avaliação de vídeos reais com sliding window (I3D + TimeSformer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--videos", nargs="*", default=None,
        metavar="VIDEO",
        help="Caminhos dos vídeos a avaliar. Se omitido, avalia todos em datasets/real/.",
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Pula a geração dos vídeos overlay (apenas plots estáticos).",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/real_videos",
        help="Diretório raiz de saída.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    print(f"Saída: {output_root}")

    # ── Carrega modelos ──────────────────────────────────────────────
    print("\nCarregando modelos...")
    print(f"[I3D] Carregando: {I3D_CHECKPOINT.name}")
    i3d_model, _ = load_i3d_model(I3D_CHECKPOINT, model_mode="rgb_only")
    print(f"[TS] Carregando modelo de: {TS_MODEL_DIR}")
    ts_model, ts_processor = load_model_and_processor(TS_MODEL_DIR)
    print("Modelos carregados.\n")

    # ── Seleciona vídeos ─────────────────────────────────────────────
    if args.videos:
        video_paths = [Path(v).resolve() for v in args.videos]
    else:
        real_dir = PROJECT_ROOT / "datasets" / "real"
        video_paths = sorted(real_dir.glob("*.mp4"))

    if not video_paths:
        print("Nenhum vídeo encontrado. Use --videos ou coloque arquivos em datasets/real/")
        sys.exit(1)

    print(f"Vídeos a processar: {len(video_paths)}")
    for vp in video_paths:
        print(f"  - {vp.name}")

    # ── Processa cada vídeo ──────────────────────────────────────────
    for vp in video_paths:
        try:
            process_video(
                vp,
                i3d_model,
                ts_model,
                ts_processor,
                output_root,
                generate_video=not args.no_video,
            )
        except Exception as exc:
            print(f"\n[ERRO] {vp.name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Concluído! Resultados em: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
