from typing import List, Dict, Any
import numpy as np

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def train_topic_model(
    documents: List[str],
    embeddings: np.ndarray,
    stopwords: set,
    umap_params: Dict[str, Any] = None,  
    hdbscan_params: Dict[str, Any] = None 
):
    print("1. Treinando BERTopic na GPU (cuML UMAP + cuML HDBSCAN + c-TF-IDF)...")

    if umap_params is None:
        umap_params = {}
    if hdbscan_params is None:
        hdbscan_params = {}

    umap_model = UMAP(**umap_params)
    hdbscan_model = HDBSCAN(**hdbscan_params)

    vectorizer = CountVectorizer(stop_words=list(stopwords))
    
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        ctfidf_model=ClassTfidfTransformer(),
        vectorizer_model=vectorizer,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(documents, embeddings)

    return topic_model, topics

def evaluate_model(
    topic_model: BERTopic,
    topics: List[int],
    documents: List[str],
    embeddings: np.ndarray
) -> dict:
    print("2. Avaliando modelo...")

    # --- Silhoueta ---
    silhouette = silhouette_score(embeddings, topics, metric="cosine")
    print(f"   -> Silhoueta: {silhouette:.4f}")

    # --- Diversidade ---
    topic_words = []
    for t in set(topics):
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        if not words:
            continue
        topic_words.append([word for word, _ in words])
    all_words = [word for words in topic_words for word in words]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0.0
    print(f"   -> Diversidade: {diversity:.4f}")

    # --- Coerência (c_v) ---
    tokenized = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    print(f"   -> Coerência (c_v): {coherence:.4f}")

    return {
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
        "silhouette": round(silhouette, 4),
        "diversity": round(diversity, 4),
        "coherence_cv": round(coherence, 4),
    }