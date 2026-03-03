"""
Pipeline de preprocessamento completo para a rede I3D.

Fluxo:
  1. Extrai frames brutos dos 3 datasets de vídeo (DCSASS, MNNIT, Shoplifting 2.0)
     → datasets/preprocessed/i3d/event_blocks_frames/
  2. Amostra 64 frames por bloco, redimensiona para 224x224 (input RGB do I3D)
     → datasets/preprocessed/i3d/i3d_inputs/rgb/
  3. Gera fluxo ótico denso TV-L1 a partir dos frames RGB
     → datasets/preprocessed/i3d/i3d_inputs/optical_flow/

Uso:
  python scripts/preprocess_i3d.py --config scripts/config.yaml
  python scripts/preprocess_i3d.py --config scripts/config.yaml --steps extract sample flow
  python scripts/preprocess_i3d.py --config scripts/config.yaml --steps sample flow  # pula extração
"""

import argparse
import sys
from pathlib import Path

import yaml

# Imports do pacote i3d_shoplifting (workspace member)
from i3d_shoplifting.preprocessing.extract_DSCASS import main_extract_dcsass
from i3d_shoplifting.preprocessing.extract_others import main_extract_others
from i3d_shoplifting.preprocessing.sample_event_blocks_i3d import (
    main as sample_blocks,
)
from i3d_shoplifting.preprocessing.gen_optical_flow import main as gen_optical_flow


#  Helpers 

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


#  Etapa 1: Extração de frames brutos 

def step_extract(config: dict, project_root: Path) -> None:
    """Extrai frames brutos de todos os datasets para event_blocks_frames/."""
    data_cfg = config["data"]
    datasets_cfg = data_cfg["datasets"]
    output_dir = str(resolve_path(
        project_root,
        config["preprocessing"]["i3d"]["event_blocks_dir"],
    ))

    print("=" * 70)
    print("ETAPA 1 — Extração de frames brutos dos datasets")
    print("=" * 70)

    #  1a. DCSASS 
    dcsass_root = str(resolve_path(project_root, datasets_cfg["dcsass"]["root"]))
    annotations = str(resolve_path(project_root, datasets_cfg["dcsass"]["annotations"]))

    if Path(dcsass_root).exists() and Path(annotations).exists():
        print(f"\n[DCSASS] root={dcsass_root}")
        print(f"[DCSASS] annotations={annotations}")
        print(f"[DCSASS] output={output_dir}\n")
        main_extract_dcsass(dcsass_root, annotations, output_dir)
    else:
        print("[DCSASS] AVISO: dataset ou anotações não encontrados — pulando.")

    #  1b. MNNIT 
    mnnit_root = resolve_path(project_root, datasets_cfg["mnnit"]["root"])

    mnnit_normal = mnnit_root / "Normal"
    mnnit_shoplifting = mnnit_root / "Shoplifting"

    if mnnit_normal.exists():
        print(f"\n[MNNIT-Normal] input={mnnit_normal}")
        main_extract_others(str(mnnit_normal), output_dir, dataset_type="Normal")
    else:
        print(f"[MNNIT-Normal] AVISO: '{mnnit_normal}' não encontrado — pulando.")

    if mnnit_shoplifting.exists():
        print(f"\n[MNNIT-Shoplifting] input={mnnit_shoplifting}")
        main_extract_others(str(mnnit_shoplifting), output_dir, dataset_type="Shoplifting")
    else:
        print(f"[MNNIT-Shoplifting] AVISO: '{mnnit_shoplifting}' não encontrado — pulando.")

    #  1c. Shoplifting Dataset 2.0 
    s2_root = resolve_path(project_root, datasets_cfg["shoplifting_2"]["root"])

    s2_normal = s2_root / "normal"
    s2_see_and_let = s2_root / "see and let"
    s2_shoplifting = s2_root / "shoplifting"

    if s2_normal.exists():
        print(f"\n[S2-Normal] input={s2_normal}")
        main_extract_others(str(s2_normal), output_dir, dataset_type="Normal")
    else:
        print(f"[S2-Normal] AVISO: '{s2_normal}' não encontrado — pulando.")

    # "see and let" → comportamento normal
    if s2_see_and_let.exists():
        print(f"\n[S2-SeeAndLet] input={s2_see_and_let}  (tratado como Normal)")
        main_extract_others(str(s2_see_and_let), output_dir, dataset_type="Normal")
    else:
        print(f"[S2-SeeAndLet] AVISO: '{s2_see_and_let}' não encontrado — pulando.")

    if s2_shoplifting.exists():
        print(f"\n[S2-Shoplifting] input={s2_shoplifting}")
        main_extract_others(str(s2_shoplifting), output_dir, dataset_type="Shoplifting")
    else:
        print(f"[S2-Shoplifting] AVISO: '{s2_shoplifting}' não encontrado — pulando.")

    print("\n✔ Extração de frames brutos concluída.\n")


#  Etapa 2: Amostragem + resize para I3D (RGB) 

def step_sample(config: dict, project_root: Path) -> None:
    """Amostra 64 frames por bloco e redimensiona para 224×224."""
    i3d_cfg = config["preprocessing"]["i3d"]
    source_dir = str(resolve_path(project_root, i3d_cfg["event_blocks_dir"]))
    output_dir = str(resolve_path(project_root, i3d_cfg["rgb_dir"]))

    print("=" * 70)
    print("ETAPA 2 — Amostragem de frames e geração de inputs RGB (I3D)")
    print("=" * 70)
    print(f"  source : {source_dir}")
    print(f"  output : {output_dir}\n")

    if not Path(source_dir).exists():
        print(f"ERRO: Diretório de blocos '{source_dir}' não existe. Execute a etapa 'extract' primeiro.")
        sys.exit(1)

    sample_blocks(source_dir, output_dir)
    print("\n✔ Inputs RGB gerados.\n")


#  Etapa 3: Fluxo ótico 

def step_flow(config: dict, project_root: Path) -> None:
    """Gera fluxo ótico denso (TV-L1) a partir dos inputs RGB."""
    i3d_cfg = config["preprocessing"]["i3d"]
    source_dir = str(resolve_path(project_root, i3d_cfg["rgb_dir"]))
    output_dir = str(resolve_path(project_root, i3d_cfg["optical_flow_dir"]))

    print("=" * 70)
    print("ETAPA 3 — Geração de fluxo ótico (TV-L1)")
    print("=" * 70)
    print(f"  source : {source_dir}")
    print(f"  output : {output_dir}\n")

    if not Path(source_dir).exists():
        print(f"ERRO: Diretório RGB '{source_dir}' não existe. Execute a etapa 'sample' primeiro.")
        sys.exit(1)

    gen_optical_flow(source_dir, output_dir)
    print("\n✔ Fluxo ótico gerado.\n")


#  Orquestrador principal 

STEPS = {
    "extract": step_extract,
    "sample": step_sample,
    "flow": step_flow,
}

ALL_STEPS = ["extract", "sample", "flow"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocessamento para a rede I3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config.yaml",
        help="Caminho para o arquivo de configuração YAML (default: scripts/config.yaml)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Etapas a executar (default: todas). Opções: extract, sample, flow",
    )
    args = parser.parse_args()

    # Raiz do projeto = diretório pai de scripts/
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(args.config)

    print(f"Raiz do projeto : {project_root}")
    print(f"Config          : {args.config}")
    print(f"Etapas          : {', '.join(args.steps)}\n")

    for step_name in args.steps:
        STEPS[step_name](config, project_root)

    print("=" * 70)
    print("PIPELINE I3D CONCLUÍDO COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
