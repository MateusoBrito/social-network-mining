import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from src.utils.data_loader import ROOT

CAMINHO_GRAFO = ROOT / "artifacts/graph/network_disparity.graphml"
SAIDAS = ROOT / "reports/graph_analysis/louvain_communities_macro.png"
CAMINHO_CSV = ROOT / "reports/graph_analysis/louvain_communities.csv"

def plot_louvain_communities():
    print('='*50)
    print(f"Carregando o grafo de:\n{CAMINHO_GRAFO}")
    
    try:
        G = nx.read_graphml(CAMINHO_GRAFO)
    except FileNotFoundError:
        print("Erro: Arquivo .graphml não encontrado. Rode o pipeline de extração primeiro")
        return

    print(f"Grafo carregado com sucesso: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    
    print("\nFiltrando ilhas isoladas...")
    componentes = list(nx.connected_components(G))
    maior_componente = max(componentes, key=len)
    G = G.subgraph(maior_componente).copy()
    print(f"O Componente Gigante possui {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    print(f"Foram ignoradas {len(componentes) - 1} pequenas ilhas isoladas")
    
    print("\nExecutando clusterização de comunidades (Louvain)...")
    
    RESOLUCAO = 0.5
    
    communities = nx.community.louvain_communities(G, weight='peso_jaccard', resolution=RESOLUCAO)
    
    print(f"Com resolução {RESOLUCAO}, a rede foi particionada em {len(communities)} macro comunidades válidas")

    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    node_colors = [community_map[node] for node in G.nodes()]

    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 5 for node in G.nodes()] 

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    print("\nDesenhando e renderizando o gráfico de alta resolução")
    plt.figure(figsize=(18, 18), facecolor='white') 
    
    nx.draw_networkx_edges(G, pos, alpha=0.08, edge_color='#555555')
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        cmap=plt.cm.turbo,    
        node_color=node_colors, 
        alpha=0.9,
        linewidths=0          
    )

    plt.title("Estrutura Topológica do Ecossistema (Comunidades Louvain)", color='black', fontsize=24, pad=20)
    plt.axis('off') 

    SAIDAS.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(SAIDAS, dpi=300, facecolor='white', bbox_inches='tight')
    
    print(f"Gráfico gerado em:\n{SAIDAS}")

    community_rows = []

    for community_id, community in enumerate(communities):
        for subreddit in community:
            community_rows.append({
                "subreddit": subreddit,
                "community_id": community_id
            })

    df_communities = pd.DataFrame(community_rows)

    print(df_communities.head())
    
    df_communities.to_csv(CAMINHO_CSV,index=False)

    print(f"Comunidades salvas em {CAMINHO_CSV}")

if __name__ == "__main__":
    plot_louvain_communities()