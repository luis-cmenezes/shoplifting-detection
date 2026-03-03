#!/usr/bin/env python3
"""
Script para download automático dos dados do HuggingFace.
Baixa o dataset menezes-luis/tcc-shoplifting e organiza na estrutura esperada.
"""
import argparse
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

def load_config(config_path="config.yaml"):
    """Carrega configuração do arquivo YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_dataset(repo_id, local_dir, cache_dir=None):
    """
    Baixa dataset do HuggingFace usando snapshot_download.
    
    Args:
        repo_id: ID do repositório no HuggingFace (ex: 'menezes-luis/tcc-shoplifting')
        local_dir: Diretório local de destino
        cache_dir: Diretório de cache (opcional)
    """
    print(f"Baixando dataset {repo_id}...")
    
    try:
        # Download complete repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            cache_dir=cache_dir
        )
        
        print(f"Dataset baixado com sucesso em: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        raise

def organize_dataset_structure(downloaded_path, config):
    """
    Organiza a estrutura dos dados baixados conforme esperado pelos scripts.
    
    Args:
        downloaded_path: Caminho onde os dados foram baixados
        config: Configuração carregada do YAML
    """
    print("Organizando estrutura de dados...")
    
    # Ensure target directories exist
    for dataset_name, dataset_config in config['data']['datasets'].items():
        target_path = Path(dataset_config['root'])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset exists in downloaded data
        source_candidates = [
            Path(downloaded_path) / dataset_name,
            Path(downloaded_path) / dataset_config['root'].split('/')[-1],
            # Try common variations
            Path(downloaded_path) / "DCSASS_Dataset" if dataset_name == 'dcsass' else None,
            Path(downloaded_path) / "Shoplifting - MNNIT" if dataset_name == 'mnnit' else None,
            Path(downloaded_path) / "Shoplifting Dataset 2.0" if dataset_name == 'shoplifting_2' else None,
        ]
        
        source_path = None
        for candidate in source_candidates:
            if candidate and candidate.exists():
                source_path = candidate
                break
        
        if source_path:
            print(f"  {dataset_name}: {source_path} → {target_path}")
            
            # Create symlink or copy based on preference
            if target_path.exists():
                print(f"    {target_path} já existe, pulando...")
            else:
                # Create parent directory
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy or symlink
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
                    
                print(f"    Copiado para {target_path}")
        else:
            print(f"    Dataset {dataset_name} não encontrado nos dados baixados")

def verify_dataset_integrity(config):
    """
    Verifica se todos os datasets necessários estão disponíveis.
    
    Args:
        config: Configuração carregada do YAML
    """
    print("Verificando integridade dos datasets...")
    
    missing_datasets = []
    
    for dataset_name, dataset_config in config['data']['datasets'].items():
        dataset_path = Path(dataset_config['root'])
        
        if not dataset_path.exists():
            missing_datasets.append(dataset_name)
            print(f"  {dataset_name}: {dataset_path} não encontrado")
        else:
            # Check for expected files/structure
            if dataset_name == 'dcsass' and 'annotations' in dataset_config:
                annotations_path = Path(dataset_config['annotations'])
                if annotations_path.exists():
                    print(f"  {dataset_name}: Dataset e anotações OK")
                else:
                    print(f"   {dataset_name}: Dataset OK, mas anotações não encontradas em {annotations_path}")
            else:
                print(f"  {dataset_name}: Dataset OK")
    
    if missing_datasets:
        print(f"  Datasets em falta: {missing_datasets}")
        return False
    else:
        print(" Todos os datasets verificados com sucesso!")
        return True

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Download dados do HuggingFace")
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Caminho para arquivo de configuração (default: config.yaml)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Força novo download mesmo se dados já existirem"
    )
    parser.add_argument(
        "--cache-dir",
        help="Diretório de cache para HuggingFace (opcional)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        print(f"Arquivo de configuração não encontrado: {args.config}")
        return 1
    
    config = load_config(args.config)
    
    # Check if data already exists (unless force download)
    if not args.force_download:
        if verify_dataset_integrity(config):
            print("Todos os dados já estão disponíveis. Use --force-download para baixar novamente.")
            return 0
    
    # Download dataset
    repo_id = config['data']['huggingface_repo']
    local_dir = config['data']['raw_data_dir']
    
    try:
        downloaded_path = download_dataset(
            repo_id=repo_id,
            local_dir=local_dir,
            cache_dir=args.cache_dir
        )
        
        # Organize structure
        organize_dataset_structure(downloaded_path, config)
        
        # Final verification
        if verify_dataset_integrity(config):
            print("Download e organização concluídos com sucesso!")
            return 0
        else:
            print("Alguns problemas foram encontrados na verificação final.")
            return 1
            
    except Exception as e:
        print(f"Erro durante o download: {e}")
        return 1

if __name__ == "__main__":
    exit(main())