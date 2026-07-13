import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from src.utils.data_loader import load_raw_data, ROOT

def load_and_clean_data(columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Carregando e limpando dados...")
    df = load_raw_data(columns, False)
    
    print("\nRemovendo autores [deleted] para evitar super-nós de ruído...")
    df = df[df['author'] != '[deleted]']

    # deixa só quem postou em mais de um subreddit
    author_counts = df.groupby('author')['subreddit'].nunique()
    authors_conectors = author_counts[author_counts >= 2].index
    
    df_filtered = df[df['author'].isin(authors_conectors)].copy()
    df_unique = df_filtered[['subreddit', 'author']].drop_duplicates()
    
    print(f"Postagens originais: {len(df)}")
    print(f"Postagens após filtragem: {len(df_filtered)}")
    print(f"Conexões únicas (subreddit-autor): {len(df_unique)}")
    
    return df_filtered, df_unique

def create_bipartite_matrix(df: pd.DataFrame) -> tuple[sp.csr_matrix, dict, dict]:
    print("\nConvertendo autores e subreddits para índices numéricos...")
    author_cat = pd.Categorical(df['author'])
    subreddit_cat = pd.Categorical(df['subreddit'])

    row_indices = author_cat.codes
    col_indices = subreddit_cat.codes

    author_map = dict(enumerate(author_cat.categories))
    subreddit_map = dict(enumerate(subreddit_cat.categories))

    print("Construindo a Matriz de Incidência Esparsa B...")
    data = np.ones(len(row_indices), dtype=np.int8)
    B_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(len(author_map), len(subreddit_map)))
    
    return B_matrix, author_map, subreddit_map

def project_unipartite_network(B_matrix: sp.csr_matrix, subreddit_map: dict) -> pd.DataFrame:
    print("\nCalculando a projeção de co-autoria (B^T * B)...")
    A = B_matrix.T.dot(B_matrix)
    A.setdiag(0)
    A.eliminate_zeros()
    
    tamanhos_subs = np.array(B_matrix.sum(axis=0)).flatten()
    A_coo = A.tocoo()
    
    linhas, colunas, pesos_absolutos = A_coo.row, A_coo.col, A_coo.data
    uniao = tamanhos_subs[linhas] + tamanhos_subs[colunas] - pesos_absolutos
    jaccard = pesos_absolutos / uniao

    edges_df = pd.DataFrame({
        'source_id': linhas,
        'target_id': colunas,
        'peso_absoluto': pesos_absolutos,
        'peso_jaccard': jaccard
    })

    edges_df = edges_df[edges_df['source_id'] < edges_df['target_id']].copy()
    edges_df['source'] = edges_df['source_id'].map(subreddit_map)
    edges_df['target'] = edges_df['target_id'].map(subreddit_map)
    
    return edges_df[['source', 'target', 'peso_absoluto', 'peso_jaccard']]

def filter_by_jaccard(edges_df: pd.DataFrame, df_original: pd.DataFrame, quantile: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nAplicando Extração de Backbone (Top {1-quantile:.0%} Jaccard)...")
    limiar = edges_df['peso_jaccard'].quantile(quantile)
    edges_backbone = edges_df[edges_df['peso_jaccard'] >= limiar].copy()

    subs_sobreviventes = set(edges_backbone['source']).union(set(edges_backbone['target']))
    df_final = df_original[df_original['subreddit'].isin(subs_sobreviventes)]
    
    return edges_backbone, df_final

def filter_by_disparity(edges_df: pd.DataFrame, df_original: pd.DataFrame, alpha_level: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nAplicando Modelo Nulo Estatístico (Filtro de Disparidade)...")
    
    edges_sym = pd.concat([
        edges_df[['source', 'target', 'peso_absoluto']].rename(columns={'source': 'node', 'target': 'vizinho'}),
        edges_df[['target', 'source', 'peso_absoluto']].rename(columns={'target': 'node', 'source': 'vizinho'})
    ])

    node_stats = edges_sym.groupby('node').agg(k=('vizinho', 'count'), s=('peso_absoluto', 'sum')).reset_index()
    edges_sym = edges_sym.merge(node_stats, on='node')
    
    edges_sym['p_ij'] = edges_sym['peso_absoluto'] / edges_sym['s']
    edges_sym['alpha'] = np.where(edges_sym['k'] > 1, np.power(1 - edges_sym['p_ij'], edges_sym['k'] - 1), 1.0)

    edges_sym['edge_id'] = edges_sym.apply(lambda x: tuple(sorted([x['node'], x['vizinho']])), axis=1)
    edge_alphas = edges_sym.groupby('edge_id')['alpha'].min().reset_index()

    significant_edges = edge_alphas[edge_alphas['alpha'] < alpha_level]['edge_id']
    edges_backbone = edges_df[edges_df.apply(lambda x: tuple(sorted([x['source'], x['target']])), axis=1).isin(significant_edges)].copy()

    subs_sobreviventes = set(edges_backbone['source']).union(set(edges_backbone['target']))
    df_final = df_original[df_original['subreddit'].isin(subs_sobreviventes)]
    
    print(f"Confiança Estatística: {1 - alpha_level:.0%} (p-valor < {alpha_level})")
    print(f"Subreddits validados: {len(subs_sobreviventes)}")
    print(f"Arestas estruturais mantidas: {len(edges_backbone)}")
    
    return edges_backbone, df_final

def plot_distributions(edges_df: pd.DataFrame):
    print("\nGerando gráficos de distribuição...")
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.ecdfplot(data=edges_df, x='peso_jaccard', complementary=True, ax=axes[0], color='blue')
    p95 = edges_df['peso_jaccard'].quantile(0.95)
    axes[0].axvline(p95, color='red', linestyle='-', label=f'Top 5% (J > {p95:.4f})')
    axes[0].set_yscale('log')
    axes[0].set_title('CCDF do Índice de Jaccard')
    axes[0].legend()

    sns.ecdfplot(data=edges_df, x='peso_absoluto', ax=axes[1], color='purple')
    axes[1].set_xlim(0, 20) 
    axes[1].set_title('CDF da Co-autoria Absoluta')

    plt.tight_layout()
    output_path = ROOT / "reports/graph_analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "distribuicao_pesos.png", dpi=300, bbox_inches='tight')

def save_outputs(edges_backbone: pd.DataFrame, df_final: pd.DataFrame):
    output_dir = ROOT / "artifacts/graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGerando arquivo GraphML para visualização da rede...")
    G = nx.from_pandas_edgelist(
        edges_backbone, 
        source='source', 
        target='target', 
        edge_attr=['peso_absoluto', 'peso_jaccard']
    )
    nx.write_graphml(G, output_dir / "network_disparity.graphml")


def main():
    df_original, df_unique = load_and_clean_data(['author', 'subreddit'])
    B_matrix, _, subreddit_map = create_bipartite_matrix(df_unique)
    edges_df = project_unipartite_network(B_matrix, subreddit_map)

    plot_distributions(edges_df)

    edges_backbone, df_final = filter_by_disparity(edges_df, df_original, alpha_level=0.05)

    save_outputs(edges_backbone, df_final)

if __name__ == "__main__":
    main()