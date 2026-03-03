"""
Pipeline de preprocessamento completo para o TimeSformer.

Fluxo:
  1. Processa o DCSASS (blocos de eventos com contexto) → vídeos .mp4 padronizados
  2. Processa o MNNIT (vídeos individuais Normal/Shoplifting) → vídeos .mp4 padronizados
  3. Processa o Shoplifting Dataset 2.0 (normal/shoplifting/see-and-let) → vídeos .mp4 padronizados
  4. Gera manifest.csv com caminhos absolutos e labels

Saída:
  datasets/preprocessed/timesformer/standardized/
      Normal/
          dcsass_normal_0000.mp4
          mnnit_0000.mp4
          s2_0000.mp4
          ...
      Shoplifting/
          dcsass_shoplifting_0000.mp4
          mnnit_0000.mp4
          s2_0000.mp4
          ...
      manifest.csv

Uso:
  python scripts/preprocess_timesformer.py --config scripts/config.yaml
  python scripts/preprocess_timesformer.py --config scripts/config.yaml --steps dcsass mnnit s2
  python scripts/preprocess_timesformer.py --config scripts/config.yaml --steps manifest
"""

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

#  Imports do pacote timesformer_shoplifting (workspace member) 
from timesformer_shoplifting.preprocessing.process_and_standardize_data import (
    ensure_ffmpeg_exists,
    safe_makedirs,
    load_annotations,
    identify_event_blocks_with_context,
    write_ffmpeg_file_list,
    run_ffmpeg_concat_and_standardize,
    process_simple_dataset,
    generate_manifest,
    TMP_LIST_FILENAME,
)


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


def _get_timesformer_cfg(config: dict) -> dict:
    """Extrai a seção de configuração do TimeSformer."""
    return config["preprocessing"]["timesformer"]


#  Etapa 1: DCSASS (blocos com contexto) 

def step_dcsass(config: dict, project_root: Path) -> None:
    """Processa o DCSASS: identifica blocos de eventos, concatena e padroniza."""
    datasets_cfg = config["data"]["datasets"]
    tf_cfg = _get_timesformer_cfg(config)
    output_root = str(resolve_path(project_root, tf_cfg["output_dir"]))

    print("=" * 70)
    print("ETAPA 1 — DCSASS (blocos de eventos com contexto)")
    print("=" * 70)

    dcsass_root = str(resolve_path(project_root, datasets_cfg["dcsass"]["root"]))
    annotations_path = str(resolve_path(project_root, datasets_cfg["dcsass"]["annotations"]))

    if not Path(dcsass_root).exists() or not Path(annotations_path).exists():
        print("[DCSASS] AVISO: dataset ou anotações não encontrados — pulando.")
        return

    print(f"  root        : {dcsass_root}")
    print(f"  annotations : {annotations_path}")
    print(f"  output      : {output_root}\n")

    ensure_ffmpeg_exists()
    safe_makedirs(output_root)
    safe_makedirs(str(Path(output_root) / "Normal"))
    safe_makedirs(str(Path(output_root) / "Shoplifting"))

    annotations = load_annotations(annotations_path)
    if annotations is None:
        print("[DCSASS] Falha ao carregar anotações — pulando.")
        return

    blocks = identify_event_blocks_with_context(dcsass_root, annotations)
    print(f"Total de blocos identificados: {len(blocks)}")

    prefix = tf_cfg["prefixes"]["dcsass"]
    fps = tf_cfg["target_fps"]
    w = tf_cfg["target_width"]
    h = tf_cfg["target_height"]

    counter_norm = 0
    counter_shop = 0

    for idx, blk in enumerate(tqdm(blocks, desc="DCSASS: processando blocos")):
        label = blk["label"]
        clip_paths = blk["clip_paths"]

        out_dir_name = "Shoplifting" if label == 1 else "Normal"
        out_dir = str(Path(output_root) / out_dir_name)
        safe_makedirs(out_dir)

        if label == 1:
            out_basename = f"{prefix}_shoplifting_{counter_shop:04d}.mp4"
            counter_shop += 1
        else:
            out_basename = f"{prefix}_normal_{counter_norm:04d}.mp4"
            counter_norm += 1

        out_path = str(Path(out_dir) / out_basename)

        tmp_list = str(Path(out_dir) / TMP_LIST_FILENAME)
        write_ffmpeg_file_list(clip_paths, tmp_list)

        ok, err = run_ffmpeg_concat_and_standardize(
            tmp_list, out_path, fps=fps, w=w, h=h
        )
        if not ok:
            print(f"Erro FFmpeg no bloco {idx} -> {out_path}: {err}")

        # Cleanup
        try:
            if Path(tmp_list).exists():
                Path(tmp_list).unlink()
        except Exception:
            pass

    print("\n✔ DCSASS concluído.\n")


#  Etapa 2: MNNIT 

def step_mnnit(config: dict, project_root: Path) -> None:
    """Processa o MNNIT: vídeos individuais Normal/ e Shoplifting/."""
    datasets_cfg = config["data"]["datasets"]
    tf_cfg = _get_timesformer_cfg(config)
    output_root = str(resolve_path(project_root, tf_cfg["output_dir"]))

    print("=" * 70)
    print("ETAPA 2 — MNNIT (vídeos individuais)")
    print("=" * 70)

    mnnit_root = resolve_path(project_root, datasets_cfg["mnnit"]["root"])
    if not mnnit_root.exists():
        print(f"[MNNIT] AVISO: '{mnnit_root}' não encontrado — pulando.")
        return

    print(f"  input  : {mnnit_root}")
    print(f"  output : {output_root}\n")

    ensure_ffmpeg_exists()
    safe_makedirs(output_root)

    mnnit_map = {"Normal": 0, "Shoplifting": 1}
    process_simple_dataset(
        str(mnnit_root),
        mnnit_map,
        output_root,
        tf_cfg["prefixes"]["mnnit"],
        fps=tf_cfg["target_fps"],
        w=tf_cfg["target_width"],
        h=tf_cfg["target_height"],
    )

    print("\n✔ MNNIT concluído.\n")


#  Etapa 3: Shoplifting Dataset 2.0 

def step_s2(config: dict, project_root: Path) -> None:
    """Processa o Shoplifting Dataset 2.0: normal, shoplifting, see-and-let."""
    datasets_cfg = config["data"]["datasets"]
    tf_cfg = _get_timesformer_cfg(config)
    output_root = str(resolve_path(project_root, tf_cfg["output_dir"]))

    print("=" * 70)
    print("ETAPA 3 — Shoplifting Dataset 2.0")
    print("=" * 70)

    s2_root = resolve_path(project_root, datasets_cfg["shoplifting_2"]["root"])
    if not s2_root.exists():
        print(f"[S2] AVISO: '{s2_root}' não encontrado — pulando.")
        return

    print(f"  input  : {s2_root}")
    print(f"  output : {output_root}\n")

    ensure_ffmpeg_exists()
    safe_makedirs(output_root)

    s2_map = {"normal": 0, "shoplifting": 1, "see and let": 0}
    process_simple_dataset(
        str(s2_root),
        s2_map,
        output_root,
        tf_cfg["prefixes"]["s2"],
        fps=tf_cfg["target_fps"],
        w=tf_cfg["target_width"],
        h=tf_cfg["target_height"],
    )

    print("\n✔ Shoplifting Dataset 2.0 concluído.\n")


#  Etapa 4: Manifest 

def step_manifest(config: dict, project_root: Path) -> None:
    """Gera manifest.csv varrendo as pastas Normal/ e Shoplifting/."""
    tf_cfg = _get_timesformer_cfg(config)
    output_root = str(resolve_path(project_root, tf_cfg["output_dir"]))

    print("=" * 70)
    print("ETAPA 4 — Geração de manifest.csv")
    print("=" * 70)

    if not Path(output_root).exists():
        print(f"ERRO: Diretório '{output_root}' não existe. Execute as etapas de extração primeiro.")
        sys.exit(1)

    generate_manifest(output_root)
    print("\n✔ Manifest gerado.\n")


#  Orquestrador principal 

STEPS = {
    "dcsass": step_dcsass,
    "mnnit": step_mnnit,
    "s2": step_s2,
    "manifest": step_manifest,
}

ALL_STEPS = ["dcsass", "mnnit", "s2", "manifest"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocessamento para o TimeSformer",
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
        help="Etapas a executar (default: todas). Opções: dcsass, mnnit, s2, manifest",
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
    print("PIPELINE TIMESFORMER CONCLUÍDO COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
