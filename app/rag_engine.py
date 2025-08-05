from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import os

# --- Cargar variables de entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La variable GOOGLE_API_KEY no está configurada.")

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
            "Si la pregunta no está relacionada con finanzas personales, "
            "responde brevemente que tu especialidad son las finanzas personales."
        ),
        ("human", "Historial del chat (si existe): {chat_history}"),
        ("human", "Contexto relevante:\n{context}"),
        ("human", "Pregunta:\n{question}")
    ])

# --- Crear la cadena RAG ---
def get_rag_chain():
    llm = get_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = get_prompt()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# --- Función principal ---
def answer_question(question: str, chat_history: list = None):
    # Formatear historial para ConversationalRetrievalChain
    formatted_history = []
    if chat_history:
        for msg in chat_history:
            if msg["type"] == "human":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] in ["ai", "assistant"]:
                formatted_history.append(AIMessage(content=msg["content"]))

    rag_chain = get_rag_chain()

    result = rag_chain({
        "question": question,
        "chat_history": formatted_history
    })

    return result["answer"] if "answer" in result else result
