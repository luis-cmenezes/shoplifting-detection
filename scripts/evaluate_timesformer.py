"""
Pipeline de avaliação (inferência + métricas) do modelo TimeSformer no conjunto de teste.

Lê a configuração centralizada de ``config.yaml`` e executa a avaliação
para um ou mais experimentos.

Uso:
  # Avalia todos os experimentos definidos em config.yaml
  python scripts/evaluate_timesformer.py --config scripts/config.yaml

  # Avalia apenas experimento(s) específico(s)
  python scripts/evaluate_timesformer.py --config scripts/config.yaml --experiments head_k400_8f

  # Override de parâmetros
  python scripts/evaluate_timesformer.py --config scripts/config.yaml --batch-size 32
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

from timesformer_shoplifting.inference import EvalConfig, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(base: Path, relative: str) -> Path:
    return (base / relative).resolve()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"ERRO: Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _slugify(text: str) -> str:
    """Reproduz a mesma slugify do treino para construir o nome do run."""
    text = text.strip().lower()
    text = text.replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def build_run_name(model_name: str, num_frames: int, freeze_strategy: str) -> str:
    model_id = model_name.split("/")[-1]
    return _slugify(f"{model_id}_frames{num_frames}_{freeze_strategy}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação TimeSformer — orquestrador via config.yaml",
    )
    parser.add_argument(
        "--config", type=str, default="scripts/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Nomes dos experimentos a avaliar (default: todos).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = parse_cli()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(cli.config)

    train_cfg = config.get("training", {}).get("timesformer", {})
    preprocess_cfg = config.get("preprocessing", {}).get("timesformer", {})

    if not train_cfg:
        print("ERRO: Seção 'training.timesformer' não encontrada no config.yaml.")
        sys.exit(1)
    if not preprocess_cfg:
        print("ERRO: Seção 'preprocessing.timesformer' não encontrada no config.yaml.")
        sys.exit(1)

    experiments: list[dict] = train_cfg.get("experiments", [])
    if not experiments:
        print("ERRO: Nenhum experimento definido em 'training.timesformer.experiments'.")
        sys.exit(1)

    # Filtrar por nome se necessário
    if cli.experiments is not None:
        selected = {name.strip() for name in cli.experiments}
        experiments = [e for e in experiments if e.get("name") in selected]
        if not experiments:
            print(f"ERRO: Nenhum experimento com os nomes: {cli.experiments}")
            sys.exit(1)

    # Caminhos dos dados
    data_root = str(resolve_path(project_root, preprocess_cfg["output_dir"]))
    output_base = str(resolve_path(project_root, train_cfg.get("output_dir", "results/timesformer")))

    # Split
    split = train_cfg.get("split", {})
    seed = cli.seed or train_cfg.get("seed", 42)

    total = len(experiments)
    print("=" * 70)
    print(f"AVALIAÇÃO TIMESFORMER — {total} experimento(s)")
    print("=" * 70)

    for idx, experiment in enumerate(experiments, start=1):
        exp_name = experiment.get("name", f"exp_{idx}")
        run_name = build_run_name(
            model_name=experiment["model_name"],
            num_frames=experiment.get("num_frames", 8),
            freeze_strategy=experiment.get("freeze_strategy", "unfreeze_head"),
        )
        exp_dir = str(Path(output_base) / run_name)

        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Avaliando: {exp_name}")
        print(f"  diretório: {exp_dir}")
        print(f"  modelo:    {experiment['model_name']}")
        print(f"  frames:    {experiment.get('num_frames', 8)}")
        print(f"{'='*70}\n")

        # Verificar se o diretório existe
        if not Path(exp_dir).exists():
            print(f"  AVISO: Diretório não encontrado ({exp_dir}). Pulando...")
            continue

        cfg = EvalConfig(
            experiment_dir=exp_dir,
            data_root=data_root,
            num_frames=experiment.get("num_frames", 8),
            seed=seed,
            split_test_size=split.get("test_size", 0.3),
            split_val_test_ratio=split.get("val_test_ratio", 0.5),
            batch_size=cli.batch_size,
            dataloader_num_workers=cli.dataloader_num_workers,
        )
        evaluate(cfg)
        print(f"\n[{idx}/{total}] Avaliação '{exp_name}' concluída.")

    print("\n" + "=" * 70)
    print("Todas as avaliações TimeSformer finalizadas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
