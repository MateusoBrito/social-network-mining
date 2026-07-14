from src.utils.data_loader import load_all_data, ROOT
from pathlib import Path
import pandas as pd

def load_stopwords(path: Path) -> set:
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def prepare_data():
    print("1. Carregando posts...")
    
    df_full = load_all_data(columns=['id','title_clean','lang','engajamento'])
    print(f"   -> Dataset completo: {len(df_full)} posts")

    df_full = df_full[df_full["engajamento"] == "Alto"].copy()
    print(f"   -> Textos com engajamento alto: {len(df_full)}")

    df_full = df_full[df_full["lang"] == "en"].copy()
    print(f"   -> Textos em inglês para treinamento: {len(df_full)}")

    df_unique = df_full.drop_duplicates(subset=["title_clean"]).reset_index(drop=True)
    documents_unique = df_unique["title_clean"].tolist()
    print(f"   -> Textos ÚNICOS para treinamento: {len(df_unique)}")
    
    return documents_unique, df_unique, df_full
