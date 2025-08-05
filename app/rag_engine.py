from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
import os

# --- Cargar variables de entorno ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("La variable GEMINI_API_KEY no está configurada.")

# --- Inicializar modelo ---
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=512
    )

# --- Inicializar embeddings y vectorstore ---
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# --- Definir el prompt ---
def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Eres un asistente especializado en **finanzas personales**. "
            "Responde SIEMPRE en español de manera clara y concisa. "
            "Si la pregunta no está relacionada con finanzas personales, responde brevemente "
            "que tu especialidad son las finanzas personales."
        ),
        ("human", "{context}"),
        ("human", "{question}")
    ])

# --- Crear la cadena RAG ---
def get_rag_chain():
    llm = get_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = get_prompt()

    # Memoria de conversación
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Cadena RAG
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# --- Función principal ---
def answer_question(question: str, chat_history: list):
    """
    question: Pregunta actual del usuario
    chat_history: Lista de mensajes previos con formato:
        [{"type": "human", "content": "texto"}, {"type": "ai", "content": "respuesta"}]
    """

    # Convertir historial a formato LangChain
    formatted_history = []
    for msg in chat_history:
        if msg["type"] == "human":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] in ["ai", "assistant"]:
            formatted_history.append(AIMessage(content=msg["content"]))

    # Crear la cadena RAG
    rag_chain = get_rag_chain()

    # Ejecutar la consulta
    result = rag_chain({
        "question": question,
        "context": "\n".join([f"{m.type}: {m.content}" for m in formatted_history])
    })

    return result["answer"] if "answer" in result else result
