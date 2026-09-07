from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import CHROMA_PERSIST_DIR, MODELO_EMBEDDING, OLLAMA_BASE_URL

def get_vector_store():
    embeddings = OllamaEmbeddings(
        model=MODELO_EMBEDDING,
        base_url=OLLAMA_BASE_URL
    )
    vector_store = Chroma(
        collection_name="graphrag_chroma_local",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vector_store