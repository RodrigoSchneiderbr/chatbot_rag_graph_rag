import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from src.vector_store import get_vector_store
from src.graph_store import carregar_ou_criar_grafo, salvar_grafo
from src.config import DATA_DIR, MODELO_LLM, OLLAMA_BASE_URL

def extrair_triplos(texto: str, llm) -> list:
    prompt = ChatPromptTemplate.from_template(
        "Você é um extrator de conhecimento rigoroso. "
        "Analise o texto abaixo e extraia conexões importantes entre entidades. "
        "Retorne APENAS linhas no formato exato: Entidade1 | Relação | Entidade2. "
        "Não inclua explicações, introduções ou markdown extra. Se não houver relações, retorne vazio.\n\n"
        "Texto: {texto}"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        resultado = chain.invoke({"texto": texto})
        triplos = []
        for linha in resultado.strip().split("\n"):
            linha_limpa = linha.replace("-", "").strip()
            partes = linha_limpa.split("|")
            if len(partes) == 3:
                origem = partes[0].strip()
                relacao = partes[1].strip()
                destino = partes[2].strip()
                if origem and relacao and destino:
                    triplos.append((origem, destino, {"relation": relacao})) 
        return triplos
    except Exception as e:
        print(f"Erro na extração de triplos: {e}")
        return []

def rodar_ingestao():
    vector_store = get_vector_store()
    graph = carregar_ou_criar_grafo()
    
    llm = ChatOllama(
        model=MODELO_LLM,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return

    # Carrega PDFs e Word (.docx) simultaneamente do diretório DATA_DIR
    loader_pdf = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_word = DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    
    docs = loader_pdf.load() + loader_word.load()
    
    if not docs:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content.strip() for c in chunks if c.page_content and c.page_content.strip()]

    if not texts:
        print("Nenhum texto válido encontrado nos documentos para ingestão.")
        return

    vector_store.add_texts(texts=texts)

    for chunk in chunks:
        conteudo = chunk.page_content.strip()
        if not conteudo:
            continue
        triplos = extrair_triplos(conteudo, llm)
        for origem, destino, dados_aresta in triplos:
            graph.add_edge(origem, destino, **dados_aresta)

    salvar_grafo(graph)