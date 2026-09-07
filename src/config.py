import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELO_LLM = os.getenv("MODELO_LLM", "llama3.1:latest")
MODELO_EMBEDDING = os.getenv("MODELO_EMBEDDING", "nomic-embed-text:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")