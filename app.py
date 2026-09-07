import os
import shutil
import streamlit as st
from src.agent import criar_agente_graphrag
from src.ingestion import rodar_ingestao
from src.config import DATA_DIR, CHROMA_PERSIST_DIR

# Configuração da página do Streamlit
st.set_page_config(page_title="GraphRAG Chatbot (Local)", page_icon="🧠", layout="centered")

st.title("🧠 Chatbot GraphRAG (Ollama + Chroma + Grafo)")
st.write("Faça perguntas sobre os seus documentos (PDFs e Word). O agente decidirá autonomamente entre buscar trechos no Chroma ou navegar pelas relações do Grafo.")

# --- BARRA LATERAL: UPLOAD AUTOMÁTICO E RESET ---
with st.sidebar:
    st.header("📁 Gerenciar Documentos")
    st.write("Envie novos arquivos PDF ou Word (.docx). A ingestão e atualização do grafo ocorrerão automaticamente!")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Atualizado para aceitar tanto PDF quanto Word
    uploaded_file = st.file_uploader("Escolha um arquivo PDF ou Word", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Arquivo '{uploaded_file.name}' salvo com sucesso!")
            
            with st.spinner("🔄 Processando documento, atualizando Chroma e construindo Grafo..."):
                try:
                    rodar_ingestao()
                    st.success("✅ Ingestão concluída com sucesso!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro na ingestão automática: {e}")
        else:
            st.info(f"O arquivo '{uploaded_file.name}' já está na base de dados.")

    st.markdown("---")
    st.subheader("⚙️ Zona de Perigo / Reset")
    st.write("Apague o histórico de conversas, os documentos enviados e reinicie o banco vetorial e o grafo do zero.")

    if st.button("🗑️ Limpar Memória e Documentos", type="primary"):
        with st.spinner("Limpando chat, banco vetorial e grafo..."):
            try:
                # 1. Limpa o histórico de mensagens da sessão
                st.session_state.messages = []
                
                # 2. Remove todos os arquivos da pasta data/
                if os.path.exists(DATA_DIR):
                    for filename in os.listdir(DATA_DIR):
                        file_path = os.path.join(DATA_DIR, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path, ignore_errors=True)
                
                # 3. Limpa o cache de recursos primeiro (fecha conexões ativas do Chroma)
                st.cache_resource.clear()
                
                # 4. Remove a pasta de persistência do Chroma e do Grafo com segurança
                if os.path.exists(CHROMA_PERSIST_DIR):
                    try:
                        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
                    except Exception:
                        pass
                
                st.success("✅ Tudo foi limpo com sucesso! Reiniciando...")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao realizar a limpeza: {e}")

    st.markdown("---")
    st.write("💡 **Dica:** Os modelos do Ollama (`llama3.1` e `nomic-embed-text`) precisam estar rodando em segundo plano.")

# --- INICIALIZAÇÃO DO AGENTE ---
@st.cache_resource
def carregar_agente():
    return criar_agente_graphrag()

try:
    agent_executor = carregar_agente()
except Exception as e:
    st.error(f"Erro ao inicializar o agente. Verifique se o Ollama está rodando. Detalhes: {e}")
    st.stop()

# Inicializa o histórico de chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens anteriores do chat na interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto do usuário
if prompt := st.chat_input("Digite sua pergunta sobre os documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Converte o histórico da sessão para o formato aceito pelo agente
    messages_payload = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages_payload.append(("user", msg["content"]))
        else:
            messages_payload.append(("assistant", msg["content"]))

    # Executa o Agente e exibe a resposta com painel de status visual
    with st.chat_message("assistant"):
        with st.status("🧠 Agente pensando e consultando ferramentas...", expanded=True) as status:
            try:
                resposta = agent_executor.invoke({
                    "messages": messages_payload
                })
                output_texto = resposta["output"]
                status.update(label="✅ Resposta gerada com sucesso!", state="complete", expanded=False)
            except Exception as e:
                output_texto = f"Ocorreu um erro ao processar sua solicitação: {e}"
                status.update(label="❌ Erro na execução", state="error", expanded=True)
            
        st.markdown(output_texto)
            
    st.session_state.messages.append({"role": "assistant", "content": output_texto})