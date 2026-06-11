import os
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from langdetect import detect, DetectorFactory

from src.utils.data_loader import load_raw_data, ROOT

# ===================================== FILTRANDO =====================================
def filter_df(df, checkpoint_path):
    mask_validos = (
        df['title']
        .astype(str)
        .str.strip()
        .str.len() > 0
    )

    df = df[mask_validos].copy()
    print(f"Após filtrar textos vazios: {len(df)}")

    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path)
        ids_processados = set(checkpoint['id'].tolist())
        df = df[~df['id'].isin(ids_processados)]
        print(
            f"Quantidade já classificados: "
            f"{len(ids_processados)} | "
            f"Faltantes: {len(df)}"
        )
    return df


# ===================================== DETECÇÃO =====================================
def detect_language(text):
    if not isinstance(text, str) or text.strip() == "":
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def parallel_detect(texts, num_workers=None):
    if num_workers is None:
        num_workers = int(cpu_count()/2)
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(detect_language, texts),
                total=len(texts)
            )
        )
    return results


# ===================================== MAIN =====================================
if __name__ == "__main__":
    OUTPUT = ROOT / "data" / "processed" / "lang_detection.csv"
    DetectorFactory.seed = 0

    df = load_raw_data(only_valid_ids=False)
    df = filter_df(df, OUTPUT)

    print("Iniciando classificação...")
    df["lang"] = parallel_detect(df["title"].tolist())

    df[["id", "title", "lang"]].to_csv(
        OUTPUT,
        index=False
    )

    print(f"\nProcessamento concluído.")
    print(f"Arquivo salvo em: {OUTPUT}")