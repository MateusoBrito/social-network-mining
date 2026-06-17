from typing import List
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
#import joblib
import json
import pandas as pd

from src.topic_modeling.model import train_topic_model, evaluate_model
from src.topic_modeling.export import export_topic_dictionary, export_visualizations

def run_single(
    documents: List[str],
    embeddings: np.ndarray,
    stopwords: set,
    params: dict,
    df_unique: pd.DataFrame,
    df_full: pd.DataFrame,
    output_dir: Path
):
    print("Execução única BERTopic...")
    output_dir.mkdir(parents=True, exist_ok=True)

    umap_p = {
        'n_neighbors': params['n_neighbors'],
        'n_components': params['n_components']
    }
    hdbscan_p = {
        'min_cluster_size': params['min_cluster_size'],
        'min_samples': params['min_samples'],
        'cluster_selection_epsilon': params['cluster_selection_epsilon']
    }

    topic_model, topics = train_topic_model(
        documents=documents,
        embeddings=embeddings,
        stopwords=stopwords,
        umap_params=umap_p,
        hdbscan_params=hdbscan_p
    )

   #metrics = evaluate_model(topic_model, topics, documents, embeddings)

    n_outliers = sum(1 for t in topics if t == -1)
    #metrics["outlier_pct"] = round(n_outliers / len(topics) * 100, 2)
    #metrics["params"] = params

    model_dir = output_dir / "modelo_final"
    model_dir.mkdir(parents=True, exist_ok=True)

  #  with open(model_dir / "metricas.json", "w", encoding="utf-8") as f:
   #     json.dump(metrics, f, indent=4)

    export_topic_dictionary(topic_model, topics, df_unique, df_full, model_dir)
    export_visualizations(topic_model, model_dir)

    #print(f"   -> Métricas: {metrics}")
    return topic_model, topics