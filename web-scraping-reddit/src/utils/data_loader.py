from pathlib import Path
import pandas as pd
import json
import os
from src.utils.paths import ROOT

DATA_RAW = ROOT / "data" / "raw" 
DATA_PROCESSED = ROOT / "data" / "processed"

def load_preprocessed_data(
    only_valid_ids=True,
    columns=None
):
    """
    Carrega preprocess_text e opcionalmente filtra
    pelos ids válidos.
    """
    df = pd.read_parquet(
        DATA_PROCESSED / "titles_clean.parquet",
        columns=columns
    )
    if only_valid_ids:
        ids_validos = pd.read_parquet(
            DATA_PROCESSED / "valid_ids.parquet"
        )
        ids = set(ids_validos["id"])
        df = df[df["id"].isin(ids)]

    return df

def load_raw_data(
    columns=None,
    only_valid_ids=True
):
    """
    Carrega os dados da pasta raw/
    filtra os ids validos
    """
    rows = []

    for depth in range(3):
        dir_path = DATA_RAW / str(depth) / "subreddits"
        if not dir_path.exists():
            continue
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    posts = json.load(f)
                for post in posts:
                    post["depth"] = depth
                rows.extend(posts)
            except json.JSONDecodeError as e:
                print(f"Erro JSON em {file_path}: {e}")

    df = pd.DataFrame(rows)

    if only_valid_ids:
        ids_validos = pd.read_parquet(
            DATA_PROCESSED / "valid_ids.parquet"
        )
        ids = set(ids_validos["id"])
        df = df[df["id"].isin(ids)]

    if columns is not None:
        df = df[columns]
    
    print(f"Posts carregados: {len(df)}")

    return df

def load_all_data(columns=None):
    df_kdd = pd.read_csv(DATA_PROCESSED / "df_kdd.csv")
    df_lang = pd.read_csv(DATA_PROCESSED / "lang_detection.csv")
    df_lang = df_lang.drop(columns=["title"])
    df_pp = load_preprocessed_data(columns=["id", "title_clean"])
    df_subreddit = pd.read_parquet(DATA_PROCESSED / "subreddits_metrics.parquet")

    # 1) join base + linguagem
    df_kdd_lang = pd.merge(df_kdd, df_lang, on="id", how="left")

    # 2) adiciona métricas do subreddit
    df_kdd_lang_subreddit = pd.merge(df_kdd_lang,df_subreddit,on="subreddit",how="left")

    # 3) adiciona texto processado
    df_final = pd.merge(df_kdd_lang_subreddit,df_pp,on="id",how="left")
    
    if columns:
        df_final = df_final[columns]

    return df_final