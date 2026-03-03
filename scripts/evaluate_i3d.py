"""
Pipeline de avaliação (inferência + métricas) do modelo I3D no conjunto de teste.

Lê a configuração centralizada de ``config.yaml`` e executa a avaliação
para um ou mais experimentos.

Uso:
  # Avalia todos os experimentos definidos em config.yaml
  python scripts/evaluate_i3d.py --config scripts/config.yaml

  # Avalia apenas experimento(s) específico(s)
  python scripts/evaluate_i3d.py --config scripts/config.yaml --experiments head_rgb_only

  # Override de batch-size para inferência (maior que treino pois não há gradientes)
  python scripts/evaluate_i3d.py --config scripts/config.yaml --batch-size 8
"""

import argparse
import sys
from pathlib import Path

import yaml

from i3d_shoplifting.inference import EvalConfig, evaluate


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


def experiment_dir_name(experiment: dict) -> str:
    """Reconstrói o nome do diretório de saída que o treino criou.

    O treino monta: ``aug_{"full_unfreeze" if unfreeze else "head_unfreeze"}_{model_mode}``
    """
    unfreeze = experiment.get("unfreeze_full_model", False)
    mode = experiment["model_mode"]
    prefix = "full_unfreeze" if unfreeze else "head_unfreeze"
    return f"aug_{prefix}_{mode}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação I3D — orquestrador via config.yaml",
    )
    parser.add_argument(
        "--config", type=str, default="scripts/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Nomes dos experimentos a avaliar (default: todos).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = parse_cli()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(cli.config)

    train_cfg = config.get("training", {}).get("i3d", {})
    preprocess_cfg = config.get("preprocessing", {}).get("i3d", {})

    if not train_cfg:
        print("ERRO: Seção 'training.i3d' não encontrada no config.yaml.")
        sys.exit(1)
    if not preprocess_cfg:
        print("ERRO: Seção 'preprocessing.i3d' não encontrada no config.yaml.")
        sys.exit(1)

    experiments: list[dict] = train_cfg.get("experiments", [])
    if not experiments:
        print("ERRO: Nenhum experimento definido em 'training.i3d.experiments'.")
        sys.exit(1)

    # Filtrar por nome se necessário
    if cli.experiments is not None:
        selected = {name.strip() for name in cli.experiments}
        experiments = [e for e in experiments if e.get("name") in selected]
        if not experiments:
            print(f"ERRO: Nenhum experimento com os nomes: {cli.experiments}")
            sys.exit(1)

    # Caminhos dos dados
    rgb_dir = str(resolve_path(project_root, preprocess_cfg["rgb_dir"]))
    flow_dir = str(resolve_path(project_root, preprocess_cfg["optical_flow_dir"]))
    output_base = str(resolve_path(project_root, train_cfg.get("output_dir", "results/i3d/")))

    # Split
    split = train_cfg.get("split", {})
    seed = cli.seed or train_cfg.get("seed", 42)

    total = len(experiments)
    print("=" * 70)
    print(f"AVALIAÇÃO I3D — {total} experimento(s)")
    print("=" * 70)

    for idx, experiment in enumerate(experiments, start=1):
        exp_name = experiment.get("name", f"exp_{idx}")
        exp_dir_name = experiment_dir_name(experiment)
        exp_dir = str(Path(output_base) / exp_dir_name)

        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Avaliando: {exp_name}")
        print(f"  diretório: {exp_dir}")
        print(f"  modo:      {experiment['model_mode']}")
        print(f"{'='*70}\n")

        # Verificar se o diretório existe
        if not Path(exp_dir).exists():
            print(f"  AVISO: Diretório não encontrado ({exp_dir}). Pulando...")
            continue

        cfg = EvalConfig(
            experiment_dir=exp_dir,
            model_mode=experiment["model_mode"],
            rgb_dir=rgb_dir,
            flow_dir=flow_dir,
            seed=seed,
            split_test_size=split.get("test_size", 0.3),
            split_val_test_ratio=split.get("val_test_ratio", 0.5),
            batch_size=cli.batch_size,
        )
        evaluate(cfg)
        print(f"\n[{idx}/{total}] Avaliação '{exp_name}' concluída.")

    print("\n" + "=" * 70)
    print("Todas as avaliações I3D finalizadas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
