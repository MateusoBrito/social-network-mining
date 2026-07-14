import yaml
import json
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shutil

from src.utils.paths import ROOT 

from src.topic_modeling.data_prepare import load_stopwords, prepare_data
from src.topic_modeling.embeddings import generate_embeddings
from src.topic_modeling.optimization import run_single

def load_config(config_path: str = "config.yaml") -> dict:
    """Lê o arquivo YAML e retorna um dicionário."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    
    parser = argparse.ArgumentParser(description="Pipeline de Topic Modeling")
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        default="config.yaml",
        help="Nome do arquivo de configuração YAML (ex: config_teste.yaml)"
    )

    args = parser.parse_args()

    # 2. Carrega as configurações usando o argumento passado
    config_path = ROOT / args.config
    print(f"Carregando configurações de: {config_path}")
    config = load_config(config_path)
    
    stopwords_path = ROOT / config["paths"]["stopwords"]
    cache_embeddings = ROOT / config["paths"]["cache_embeddings"]
    dir_reports = ROOT / config["paths"]["dir_reports"]

    # 2. Prepara os dados
    stopwords = load_stopwords(stopwords_path)
    
    documents_unique, df_unique, df_full = prepare_data()
    
    # 3. Gera os Embeddings passando os parâmetros dinâmicos
    embeddings = generate_embeddings(
        documents_unique, 
        cache_path=cache_embeddings,
        model_name=config["model"]["embedding_name"]
    )
    
    # 4. Executa a busca de hiperparâmetros
    params = config["model_params"]  

    topic_model, topics = run_single(
        documents_unique,
        embeddings,
        stopwords,
        params=params,
        df_unique=df_unique,
        df_full=df_full,
        output_dir=dir_reports
    )

if __name__ == "__main__":
    main()