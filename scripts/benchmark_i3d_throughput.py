"""
Benchmark de throughput de inferência — I3D e TimeSformer (melhores modelos).

Metodologia uniforme:
  - batch_size=1 (cenário realista: inferência em tempo real)
  - Dados sintéticos com shape idêntico aos dados reais
  - 20 iterações de warmup + 100 iterações cronometradas
  - CUDA synchronize antes/depois para timing preciso
  - Resultados salvos em results/figures/throughput_benchmark.json
"""

import json
import time
from pathlib import Path

import re
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_ITERS = 20
TIMED_ITERS = 100
BATCH_SIZE = 1
SPATIAL = 224

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "throughput_benchmark.json"


def _benchmark(model, dummy_input: torch.Tensor, label: str) -> dict:
    """Executa warmup + benchmark cronometrado."""
    model.eval()

    # Warmup
    print(f"  Warmup ({WARMUP_ITERS} iters)...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Timed
    print(f"  Benchmarking ({TIMED_ITERS} iters, batch_size={BATCH_SIZE})...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(TIMED_ITERS):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = TIMED_ITERS * BATCH_SIZE
    sps = total_samples / elapsed
    ms = (elapsed / total_samples) * 1000

    result = {
        "model": label,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "input_shape": str(list(dummy_input.shape)),
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(sps, 2),
        "ms_per_sample": round(ms, 2),
    }

    print(f"  → {sps:.2f} amostras/s  |  {ms:.1f} ms/amostra  |  {elapsed:.1f}s total")
    return result


def benchmark_i3d() -> dict:
    """I3D — melhor modelo: aug_full_unfreeze_rgb_only."""
    from i3d_shoplifting.models.i3d_pytorch import InceptionI3d

    print("\n" + "=" * 60)
    print("I3D — Full / RGB (melhor modelo)")
    print("=" * 60)

    weights_dir = PROJECT_ROOT / "results" / "i3d" / "aug_full_unfreeze_rgb_only" / "model_weights"
    pattern = re.compile(r"epoch_(\d+)_auc_(\d\.\d+)\.pt")
    best_path, best_auc = None, -1.0
    for f in weights_dir.iterdir():
        m = pattern.match(f.name)
        if m and float(m.group(2)) > best_auc:
            best_auc = float(m.group(2))
            best_path = f
    print(f"  Checkpoint: {best_path.name} (AUC={best_auc:.4f})")

    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(1)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_rgb_state_dict"])
    model.to(DEVICE)

    # Input: (1, 3, 64, 224, 224)
    dummy = torch.randn(BATCH_SIZE, 3, 64, SPATIAL, SPATIAL, device=DEVICE)
    return _benchmark(model, dummy, "I3D — Full / RGB (64 frames)")


def benchmark_timesformer() -> dict:
    """TimeSformer — melhor modelo: SSv2/32f/Completo."""
    from transformers import TimesformerForVideoClassification

    print("\n" + "=" * 60)
    print("TimeSformer — SSv2/32f/Completo (melhor modelo)")
    print("=" * 60)

    model_dir = PROJECT_ROOT / "results" / "timesformer" / "timesformer-base-finetuned-ssv2_frames32_unfreeze_all" / "final_model"
    print(f"  Carregando de: {model_dir}")

    model = TimesformerForVideoClassification.from_pretrained(str(model_dir))
    model.to(DEVICE)

    # Input: (1, 32, 3, 224, 224) — formato HuggingFace: (B, T, C, H, W)
    dummy = torch.randn(BATCH_SIZE, 32, 3, SPATIAL, SPATIAL, device=DEVICE)
    return _benchmark(model, dummy, "TimeSformer — SSv2/32f/Completo (32 frames)")


def main():
    results = []

    results.append(benchmark_i3d())

    # Liberar memória GPU antes de carregar o próximo modelo
    torch.cuda.empty_cache()

    results.append(benchmark_timesformer())

    # Salvar
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    for r in results:
        print(f"  {r['model']:<45s}  {r['samples_per_second']:>6.2f} amostras/s  |  {r['ms_per_sample']:>7.1f} ms/amostra")
    print(f"\nSalvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
