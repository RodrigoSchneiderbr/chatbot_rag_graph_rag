import networkx as nx
import os
from src.config import CHROMA_PERSIST_DIR

GRAPH_FILE = os.path.join(CHROMA_PERSIST_DIR, "knowledge_graph.pkl")

def carregar_ou_criar_grafo() -> nx.Graph:
    """Carrega o grafo salvo no disco ou cria um novo se não existir."""
    if os.path.exists(GRAPH_FILE):
        try:
            with open(GRAPH_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            print("Erro ao carregar o grafo existente. Criando um novo grafo vazio.")
            return nx.Graph()
    return nx.Graph()

def salvar_grafo(graph: nx.Graph):
    """Salva o estado atual do grafo no disco."""
    os.makedirs(os.path.dirname(GRAPH_FILE), exist_ok=True)
    import pickle
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)