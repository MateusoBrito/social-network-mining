import pandas as pd
import networkx as nx
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
import pyLDAvis.gensim_models
import pyLDAvis
import multiprocessing
import json

DATA_PROCESSED = "../../data/processed"

def load_preprocessed_data(
    only_valid_ids=True,
    columns=None
):
    df = pd.read_parquet(
        f"{DATA_PROCESSED}/preprocess_text.parquet",
        columns=columns
    )
    if only_valid_ids:
        ids_validos = pd.read_parquet(
            f"{DATA_PROCESSED}/ids_validos.parquet"
        )
        ids = set(ids_validos["id"])
        df = df[df["id"].isin(ids)]

    return df

def filtrar_por_comunidade(df, graphml_path):
    print("Lendo comunidades do grafo...")
    G = nx.read_graphml(graphml_path)
    subs_validos = [n for n, d in G.nodes(data=True) if 'louvain_community' in d]
    print(f"Subreddits filtrados: {len(subs_validos)}")
    return df[df['subreddit'].isin(subs_validos)].copy()

def rodar_lda_final(df, num_topicos=20):
    df_clean = df.dropna(subset=['text_clean']).copy()
    textos = [str(t).split() for t in df_clean['text_clean']]
    
    dicionario = Dictionary(textos)
    dicionario.filter_extremes(no_below=50, no_above=0.5)
    corpus = [dicionario.doc2bow(t) for t in textos]
    
    print("Treinando LDA...")
    lda = LdaMulticore(corpus, id2word=dicionario, num_topics=num_topicos, 
                       workers=multiprocessing.cpu_count()-1, passes=5, random_state=42)
    
    lda.save("modelo_lda_definitivo.gensim")
    dicionario.save("dicionario_lda_definitivo.gensim")
    
    mapa_palavras = {}
    for i in range(num_topicos):
        termos = [palavra for palavra, peso in lda.show_topic(i, topn=10)]
        mapa_palavras[i] = termos
    
    with open("mapeamento_topicos.json", "w") as f:
        json.dump(mapa_palavras, f, indent=4)
    print("Mapeamento de palavras salvo em 'mapeamento_topicos.json'")

    # Gerar HTML
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dicionario)
    pyLDAvis.save_html(vis, "visualizacao_lda_final.html")
    
    # Atribuição
    lista_topicos = []
    for doc in corpus:
        topicos = lda.get_document_topics(doc)
        topico_max = max(topicos, key=lambda x: x[1])[0]
        lista_topicos.append(topico_max)
        
    df_clean['topico_dominante'] = lista_topicos
    df_clean.to_parquet("../../data/processed/df_final_com_topicos.parquet")
    
    return lda, df_clean

df = load_preprocessed_data()
df_filtrado = filtrar_por_comunidade(df, "network_disparity.graphml")
modelo, df_final = rodar_lda_final(df_filtrado)