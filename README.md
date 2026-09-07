# 🧠 Chatbot GraphRAG Local (Ollama + Chroma + NetworkX + Streamlit)

Este repositório implementa um agente de **GraphRAG (Retrieval-Augmented Generation baseado em grafos)** rodando **100% local e gratuito** com modelos open-source. 

O sistema combina a precisão de buscas textuais tradicionais em bases vetoriais com a inteligência estruturada de grafos de conhecimento, permitindo que o agente navegue por relações complexas entre conceitos extraídos dos seus documentos.

---

##  Arquitetura do Projeto

```text
chatbot_graphrag_local/
├── data/                    # Coloque seus arquivos PDF ou TXT aqui
├── src/
│   ├── __init__.py
│   ├── config.py            # Configurações centralizadas via .env
│   ├── vector_store.py      # Configuração do Chroma DB (Ollama Embeddings)
│   ├── graph_store.py       # Gerenciamento e persistência do Grafo (NetworkX)
│   ├── ingestion.py         # Pipeline de leitura, chunking, Chroma e extração de triplos
│   ├── tools.py             # Ferramentas híbridas (Busca Vetorial + Navegação em Grafo)
│   └── agent.py             # Configuração do Agente LangChain + Ollama
├── chroma_db/               # Diretório de persistência local (Chroma + Grafo .pkl)
├── .env                     # Variáveis de ambiente
├── .gitignore
├── requirements.txt         # Dependências do projeto
└── app.py                   # Interface web interativa (Streamlit)

# 🛠️ Tecnologias Utilizadas

* **LLM & Embeddings:** Ollama (`llama3.1:latest` e `nomic-embed-text:latest`)
* **Banco Vetorial:** Chroma DB via LangChain
* **Banco de Grafos:** NetworkX (em memória com persistência local)
* **Orquestração:** LangChain & LangGraph
* **Interface Gráfica:** Streamlit

---

# ⚙️ Pré-requisitos e Instalação

### 1. Instalar e Configurar o Ollama
Certifique-se de ter o Ollama instalado e rodando em segundo plano no seu computador. Em seguida, baixe os modelos necessários executando no seu terminal:

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text:latest

2. Clonar/ Configurar o Repositório
No diretório raiz do projeto, crie um ambiente virtual e instale as dependências:

```text
# Cria e ativa o ambiente virtual
python -m venv venv

# No Windows:
venv\Scripts\activate

# No Linux/Mac:
source venv/bin/activate

# Instala as dependências
pip install -r requirements.txt
```
3. Configurar as Variáveis de Ambiente (.env)
Crie um arquivo .env na raiz do projeto com as seguintes configurações:

```text
CHROMA_PERSIST_DIR=./chroma_db
DATA_DIR=./data
MODELO_LLM=llama3.1:latest
MODELO_EMBEDDING=nomic-embed-text:latest
OLLAMA_BASE_URL=http://localhost:11434
```
4. Rodar
```text
streamlit run app.py
```