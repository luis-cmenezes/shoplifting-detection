# tcc-shoplifting

Monorepo para comparação de dois modelos de classificação binária de vídeo (**Shoplifting vs. Normal**) aplicados a vídeos de vigilância: **I3D** (Inflated 3D ConvNet) e **TimeSformer** (Time-Space Transformer).

---

## Subpacotes

| Subpacote | Descrição |
|---|---|
| [`i3d-shoplifting/`](i3d-shoplifting/) | Fine-tuning do I3D com late fusion RGB + Optical Flow. Entrada: 64 frames pré-extraídos (224×224). Treino com loop manual PyTorch e `BCEWithLogitsLoss`. |
| [`timesformer-shoplifting/`](timesformer-shoplifting/) | Fine-tuning do TimeSformer (HuggingFace). Entrada: vídeos `.mp4` padronizados, 8 frames amostrados em runtime via Decord. Treino com HuggingFace `Trainer` e `CrossEntropyLoss`. |

Cada subpacote possui seu próprio README com documentação detalhada dos módulos, argumentos CLI e exemplos de uso.

---

## Scripts centralizados (`scripts/`)

Os scripts na pasta `scripts/` orquestram o download de dados e o preprocessamento para ambos os modelos, usando caminhos centralizados definidos em `scripts/config.yaml`.

### `scripts/config.yaml`

Configuração central do workspace. Define:

- **Repositório HuggingFace** para download dos datasets (`menezes-luis/tcc-shoplifting`)
- **Caminhos dos datasets brutos:** DCSASS (com anotações CSV), MNNIT, Shoplifting Dataset 2.0
- **Caminhos de preprocessamento I3D:** `event_blocks_frames/`, `rgb/`, `optical_flow/`, `num_frames=64`
- **Caminhos de preprocessamento TimeSformer:** `standardized/`, `target_fps=25`, `target_width/height=224`

### `scripts/download_data.py`

Download automático dos datasets do HuggingFace via `snapshot_download`. Verifica integridade e organiza a estrutura de pastas conforme `config.yaml`.

```bash
uv run python scripts/download_data.py --config scripts/config.yaml
```

| Argumento | Default | Descrição |
|---|---|---|
| `--config` | `config.yaml` | Caminho para o arquivo de configuração |
| `--force-download` | `False` | Forçar re-download mesmo se já existir |
| `--cache-dir` | — | Diretório de cache do HuggingFace |

### `scripts/preprocess_i3d.py`

Pipeline completo de preprocessamento para o I3D em 3 etapas:

1. **`extract`** — Extração de frames brutos dos 3 datasets → `event_blocks_frames/`
2. **`sample`** — Amostragem de 64 frames + resize 224×224 → `i3d_inputs/rgb/`
3. **`flow`** — Geração de fluxo ótico TV-L1 → `i3d_inputs/optical_flow/`

```bash
# Todas as etapas
uv run python scripts/preprocess_i3d.py --config scripts/config.yaml

# Etapas específicas
uv run python scripts/preprocess_i3d.py --config scripts/config.yaml --steps extract sample
```

| Argumento | Default | Descrição |
|---|---|---|
| `--config` | `scripts/config.yaml` | Caminho para o arquivo de configuração |
| `--steps` | todas | Etapas a executar: `extract`, `sample`, `flow` |

### `scripts/preprocess_timesformer.py`

Pipeline completo de preprocessamento para o TimeSformer em 4 etapas:

1. **`dcsass`** — DCSASS → vídeos concatenados e padronizados (224×224, 25 FPS)
2. **`mnnit`** — MNNIT → vídeos padronizados
3. **`s2`** — Shoplifting Dataset 2.0 → vídeos padronizados
4. **`manifest`** — Geração de `manifest.csv` com caminhos absolutos e labels

```bash
# Todas as etapas
uv run python scripts/preprocess_timesformer.py --config scripts/config.yaml

# Etapas específicas
uv run python scripts/preprocess_timesformer.py --config scripts/config.yaml --steps dcsass mnnit
```

| Argumento | Default | Descrição |
|---|---|---|
| `--config` | `scripts/config.yaml` | Caminho para o arquivo de configuração |
| `--steps` | todas | Etapas a executar: `dcsass`, `mnnit`, `s2`, `manifest` |

---

## Estrutura do workspace

```
tcc-shoplifting/
├── main.py                  # Placeholder
├── pyproject.toml           # Workspace root (uv)
├── scripts/
│   ├── config.yaml          # Configuração central
│   ├── download_data.py     # Download dos datasets
│   ├── preprocess_i3d.py    # Pipeline de preprocessamento I3D
│   └── preprocess_timesformer.py  # Pipeline de preprocessamento TimeSformer
├── datasets/
│   ├── raw/                 # Datasets brutos (baixados do HuggingFace)
│   └── preprocessed/        # Dados preprocessados (I3D e TimeSformer)
├── i3d-shoplifting/         # Subpacote I3D
│   ├── pyproject.toml
│   ├── README.md
│   └── src/i3d_shoplifting/
└── timesformer-shoplifting/ # Subpacote TimeSformer
    ├── pyproject.toml
    ├── README.md
    └── src/timesformer_shoplifting/
```

---

## Setup

```bash
# Instalar todas as dependências (ambos os subpacotes)
uv sync --all-extras

# Download dos datasets
uv run python scripts/download_data.py --config scripts/config.yaml

# Preprocessar para I3D
uv run python scripts/preprocess_i3d.py --config scripts/config.yaml

# Preprocessar para TimeSformer
uv run python scripts/preprocess_timesformer.py --config scripts/config.yaml
```

### Requisitos

- **Python** ≥ 3.11, < 3.12
- **uv** como gerenciador de pacotes
- **FFmpeg** instalado no sistema (usado para extração/padronização de vídeos)
- **GPU** recomendada para treinamento
