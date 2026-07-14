import os
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from langdetect import detect, DetectorFactory

from src.utils.data_loader import load_raw_data, ROOT
from preprocessing import TextPreprocessor

OUTPUT = ROOT / "data" / "processed" / "titles_clean.parquet"

df = load_raw_data()
pp = TextPreprocessor()

def process_text(text):
    return pp.preprocess(text)

num_cpus = int(cpu_count()/2)

with Pool(num_cpus) as pool:
    df["title_clean"] = list(
        tqdm(
            pool.imap(process_text, df["title"]),
            total=len(df)
        )
    )

df.to_parquet(OUTPUT,index=False)