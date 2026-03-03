"""
Pipeline de treinamento do modelo I3D para classificação binária de Shoplifting.

Lê a configuração centralizada de ``config.yaml`` (seções ``preprocessing.i3d``
para localizar os dados e ``training.i3d`` para hiperparâmetros, splits,
augmentação e lista de experimentos) e executa um ou mais experimentos
sequencialmente.

Cada experimento combina:
  • modalidade de entrada  — ``rgb_optical`` (late fusion) ou ``rgb_only``
  • estratégia de fine-tuning — cabeça somente ou modelo completo

Uso:
  # Executa todos os experimentos definidos em config.yaml
  python scripts/train_i3d.py --config scripts/config.yaml

  # Executa apenas o(s) experimento(s) selecionado(s) pelo nome
  python scripts/train_i3d.py --config scripts/config.yaml --experiments head_rgb_only full_rgb_only

  # Sobrescreve hiperparâmetros via CLI (se necessário)
  python scripts/train_i3d.py --config scripts/config.yaml --epochs 30 --lr 5e-4
"""

import argparse
import sys
from pathlib import Path

import yaml

from i3d_shoplifting.training import TrainConfig, train
from i3d_shoplifting.training.find_max_batch_size import find_max_batch_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(base: Path, relative: str) -> Path:
    """Resolve um caminho relativo a partir da raiz do projeto."""
    return (base / relative).resolve()


def load_config(config_path: str) -> dict:
    """Carrega e retorna o dicionário de configuração YAML."""
    path = Path(config_path)
    if not path.exists():
        print(f"ERRO: Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Montagem de TrainConfig a partir do dicionário YAML
# ---------------------------------------------------------------------------

def build_train_config(
    experiment: dict,
    train_cfg: dict,
    preprocess_cfg: dict,
    project_root: Path,
    *,
    cli_overrides: argparse.Namespace,
    data_cfg: dict,
) -> TrainConfig:
    """Constrói um ``TrainConfig`` mesclando YAML + overrides de CLI.

    Prioridade: CLI > experiment-level > training-level > defaults da dataclass.
    """
    # Caminhos de dados resolvidos a partir do preprocessing.i3d
    rgb_dir = str(resolve_path(project_root, preprocess_cfg["rgb_dir"]))
    flow_dir = str(resolve_path(project_root, preprocess_cfg["optical_flow_dir"]))

    # Checkpoints pré-treinados (resolvidos a partir de data.pretrained_weights.i3d)
    ckpts = data_cfg.get("pretrained_weights", {}).get("i3d", {})
    rgb_ckpt = str(resolve_path(project_root, ckpts.get("rgb", "")))
    flow_ckpt = str(resolve_path(project_root, ckpts.get("flow", "")))

    # Diretório de saída
    output_dir = str(resolve_path(project_root, train_cfg.get("output_dir", "experiments/i3d")))

    # Split
    split = train_cfg.get("split", {})

    # Augmentação
    aug = train_cfg.get("augmentation", {})
    color_jitter = aug.get("color_jitter", None)
    # Converte de OrderedDict / dict YAML para dict puro (caso seja)
    if color_jitter is not None:
        color_jitter = dict(color_jitter)

    # --- Mescla valores: CLI > experiment > training-level ---
    epochs = cli_overrides.epochs or train_cfg.get("epochs", 70)
    learning_rate = cli_overrides.lr or train_cfg.get("learning_rate", 1e-3)
    seed = cli_overrides.seed or train_cfg.get("seed", 42)

    # batch_size: CLI > YAML.  Se "auto", determina via find_max_batch_size.
    raw_bs = cli_overrides.batch_size or train_cfg.get("batch_size", 1)
    if str(raw_bs).lower() == "auto":
        batch_size = find_max_batch_size(
            model_mode=experiment["model_mode"],
            unfreeze_full_model=experiment.get("unfreeze_full_model", False),
            rgb_checkpoint=rgb_ckpt,
            flow_checkpoint=flow_ckpt,
        )
    else:
        batch_size = int(raw_bs)

    return TrainConfig(
        model_mode=experiment["model_mode"],
        unfreeze_full_model=experiment.get("unfreeze_full_model", False),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        rgb_dir=rgb_dir,
        flow_dir=flow_dir,
        rgb_checkpoint=rgb_ckpt,
        flow_checkpoint=flow_ckpt,
        output_dir=output_dir,
        split_test_size=split.get("test_size", 0.3),
        split_val_test_ratio=split.get("val_test_ratio", 0.5),
        augmentation_p_flip=aug.get("p_flip", 0.5),
        augmentation_color_jitter=color_jitter,
        steps_to_visualize_per_epoch=train_cfg.get("steps_to_visualize_per_epoch", 2),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento I3D — orquestrador via config.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Nomes dos experimentos a executar (default: todos).",
    )

    # Overrides opcionais de hiperparâmetros (têm prioridade sobre o YAML)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", default=None,
                        help="Batch size (inteiro ou 'auto' para detecção automática).")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = parse_cli()

    # Raiz do projeto (um nível acima de scripts/)
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(cli.config)

    # Seções relevantes do YAML
    data_cfg = config.get("data", {})
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

    # Filtra experimentos se o usuário especificou nomes via --experiments
    if cli.experiments is not None:
        selected = {name.strip() for name in cli.experiments}
        experiments = [e for e in experiments if e.get("name") in selected]
        if not experiments:
            print(f"ERRO: Nenhum experimento encontrado com os nomes: {cli.experiments}")
            print(f"Disponíveis: {[e.get('name') for e in train_cfg['experiments']]}")
            sys.exit(1)

    total = len(experiments)
    print("=" * 70)
    print(f"TREINAMENTO I3D — {total} experimento(s) programado(s)")
    print("=" * 70)

    for idx, experiment in enumerate(experiments, start=1):
        exp_name = experiment.get("name", f"exp_{idx}")
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Experimento: {exp_name}")
        print(f"  modalidade      : {experiment['model_mode']}")
        print(f"  unfreeze completo: {experiment.get('unfreeze_full_model', False)}")
        print(f"{'='*70}\n")

        cfg = build_train_config(
            experiment=experiment,
            train_cfg=train_cfg,
            preprocess_cfg=preprocess_cfg,
            project_root=project_root,
            cli_overrides=cli,
            data_cfg=data_cfg,
        )
        train(cfg)

        print(f"\n[{idx}/{total}] Experimento '{exp_name}' concluído.")

    print("\n" + "=" * 70)
    print("Todos os experimentos finalizados.")
    print("=" * 70)


if __name__ == "__main__":
    main()
