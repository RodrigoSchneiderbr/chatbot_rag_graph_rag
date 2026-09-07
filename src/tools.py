from langchain_core.tools import tool
from src.vector_store import get_vector_store
from src.graph_store import carregar_ou_criar_grafo

vector_store = get_vector_store()
graph = carregar_ou_criar_grafo()

@tool
def buscar_contexto_vetorial(pergunta: str) -> str:
    """Útil para buscar trechos de texto literais, definições específicas, 
    detalhes exatos e fatos contidos nos documentos da base vetorial do Chroma."""
    docs = vector_store.similarity_search(pergunta, k=3)
    if not docs:
        return "Nenhum trecho de texto relevante encontrado na base vetorial."
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def consultar_grafo_relacoes(entidade: str) -> str:
    """Útil para descobrir conexões estruturadas, como entidades se relacionam 
    entre si, hierarquias e o contexto global de um conceito específico no Grafo de Conhecimento."""
    entidade_limpa = entidade.strip()
    
    nos_proximos = [n for n in graph.nodes() if entidade_limpa.lower() in n.lower()]
    if not nos_proximos:
        return f"A entidade '{entidade}' não possui conexões mapeadas no grafo de conhecimento."
    
    no_alvo = nos_proximos[0]
    conexoes = []
    for vizinho in graph.neighbors(no_alvo):
        dados = graph.get_edge_data(no_alvo, vizinho)
        rel = dados.get("relation", "relaciona-se com")
        conexoes.append(f"- {no_alvo} [{rel}] {vizinho}")
        
    return f"Relações encontradas no Grafo para '{no_alvo}':\n" + "\n".join(conexoes)