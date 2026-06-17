from typing import List
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import pandas as pd
import random
import json

def export_topic_dictionary(
    topic_model: BERTopic,
    topics: List[int],
    df_unique: pd.DataFrame, 
    df_full: pd.DataFrame,
    output_dir: Path
):
    print("3. Exportando Dicionário de Tópicos Enriquecido...")
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_mapping = dict(zip(df_unique["title_clean"], topics))
    
    df_full_temp = df_full.copy()
    df_full_temp["topic"] = df_full_temp["title_clean"].map(topic_mapping)

    df_topic_info = topic_model.get_topic_info()
    topics_data = []

    for _, row in df_topic_info.iterrows():
        t = row['Topic']
        
        topic_dict = {
            "Topic": t,
            "Count": row["Count"],
            "Name": row["Name"],
            "Representation": row["Representation"],
            "Representative_Docs": [],
            "Sample_Docs": []
        }

        if t != -1:
            df_topico = df_full_temp[df_full_temp['topic'] == t].dropna(subset=['title_clean'])
            tradutor = dict(zip(df_topico["title_clean"], df_topico["title_clean"]))

            rep_docs_limpos = topic_model.get_representative_docs(t)
            rep_docs_limpos = rep_docs_limpos[:3] if rep_docs_limpos else []
            rep_docs_originais = [tradutor.get(doc, doc) for doc in rep_docs_limpos]

            topic_dict["Representative_Docs"] = rep_docs_originais
            pool_amostra_original = df_topico["title_clean"].dropna().drop_duplicates().tolist()
            pool_amostra = [doc for doc in pool_amostra_original if doc not in rep_docs_originais]
            
            n_samples = min(7, len(pool_amostra))
            sample_docs = random.sample(pool_amostra, n_samples) if n_samples > 0 else []
            
            topic_dict["Sample_Docs"] = sample_docs

        topics_data.append(topic_dict)

    with open(output_dir / "topic_info.json", "w", encoding="utf-8") as f:
        json.dump(topics_data, f, indent=4, ensure_ascii=False)

    df_mapping = df_full[["id", "title_clean"]].copy()
    df_mapping["topic"] = df_mapping["title_clean"].map(topic_mapping)
    df_mapping = df_mapping.drop(columns=["title_clean"])
    df_mapping.to_parquet(output_dir / "post_topics.parquet", index=False)

    print(f"   -> Salvo em: {output_dir}")

def export_visualizations(topic_model: BERTopic, output_dir: Path):

    print("4. Gerando Visualizações HTML...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_topic_info = topic_model.get_topic_info()
    n_clusters_total = len(df_topic_info) - 1 
    
    fig_words = topic_model.visualize_barchart(
        top_n_topics=n_clusters_total, 
        n_words=10,
        title="Assinaturas Linguísticas do Ecossistema (c-TF-IDF)"
    )
    fig_words.write_html(output_dir / "visualizacao_topicos.html")

    fig_2d = topic_model.visualize_topics(
        title="Topologia do Ecossistema: Distância entre Clusters Ideológicos"
    )
    fig_2d.write_html(output_dir / "mapa_2d_topicos.html")

    print("   -> Gerando Árvore Hierárquica...")
    fig_hierarchy = topic_model.visualize_hierarchy(
        custom_labels=True, 
        title="Árvore Hierárquica dos Subreddits"
    )
    fig_hierarchy.write_html(output_dir / "hierarquia_topicos.html")
    
    print(f"   -> Visualizações salvas em: {output_dir}")