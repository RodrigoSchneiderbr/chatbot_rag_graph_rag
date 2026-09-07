from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from src.tools import buscar_contexto_vetorial, consultar_grafo_relacoes
from src.config import MODELO_LLM, OLLAMA_BASE_URL

class SimpleGraphRAGAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=MODELO_LLM,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )
        self.tools_map = {
            "buscar_contexto_vetorial": buscar_contexto_vetorial,
            "consultar_grafo_relacoes": consultar_grafo_relacoes
        }
        self.llm_with_tools = self.llm.bind_tools(list(self.tools_map.values()))

    def invoke(self, inputs: dict) -> dict:
        messages = []
        if "messages" in inputs:
            for item in inputs["messages"]:
                if isinstance(item, tuple):
                    role, content = item
                else:
                    role, content = item.get("role"), item.get("content")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        elif "input" in inputs:
            if "chat_history" in inputs:
                for role, content in inputs["chat_history"]:
                    if role == "human":
                        messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=inputs["input"]))

        system_instruction = (
            "Você é um assistente inteligente especialista em GraphRAG e análise de documentos locais. "
            "Você tem acesso a duas ferramentas:\n"
            "1. buscar_contexto_vetorial: Use para buscar trechos textuais exatos, definições e detalhes pontuais.\n"
            "2. consultar_grafo_relacoes: Use para entender conexões complexas, hierarquias e relações globais entre conceitos.\n"
            "Sempre que necessário, acione as ferramentas para fundamentar sua resposta com precisão."
        )
        
        full_messages = [HumanMessage(content=system_instruction)] + messages

        # Primeira chamada: o LLM decide chamar a ferramenta
        response = self.llm_with_tools.invoke(full_messages)
        full_messages.append(response)

        # Se o modelo gerou chamadas de ferramentas
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in self.tools_map:
                    try:
                        # Executa a função Python correspondente à ferramenta
                        tool_output = self.tools_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_output = f"Erro ao executar a ferramenta: {e}"
                    
                    # Alimenta o resultado da ferramenta de volta para o agente
                    full_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Segunda chamada: o LLM lê o resultado da ferramenta e redige a resposta final
            final_response = self.llm.invoke(full_messages)
            output_content = final_response.content
        else:
            output_content = response.content

        return {"messages": full_messages, "output": output_content}

def criar_agente_graphrag():
    return SimpleGraphRAGAgent()