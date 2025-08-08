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
        model="gemini-2.5-flash-lite", #"gemini-1.5-flash",
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
            "Actúa como un asesor experto en finanzas personales, con un estilo conversacional, cálido y fácil de entender. "
            "Responde de forma útil, cercana y con empatía. Si el usuario habla de otros temas, puedes responder brevemente, "
            "pero conecta sutilmente con las finanzas sin sonar forzado.\n\n"
            "- Si el usuario saluda, da una bienvenida natural y ofrécele ayuda en temas financieros.\n"
            "- Si ya hay conversación previa, evita repetir saludos y continúa de forma fluida.\n"
            "- Si mencionan algo personal (edad, familia, trabajo), reconoce la información y relaciona con su impacto financiero.\n"
            "- Mantén las respuestas claras, variadas y sin sonar robótico o repetitivo.\n"
            "- Sé breve cuando el tema no sea de finanzas, pero enlaza de manera orgánica si es posible.\n\n"
            "Tu meta es ayudar al usuario a mejorar su bienestar financiero, como si hablaras con un amigo que confía en ti."
    ),
        ("human", "Historial reciente (si aplica): {chat_history}"),
        ("human", "Contexto útil:\n{context}"),
        ("human", "Pregunta del usuario:\n{question}")
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
def answer_question(question: str, chat_history: list = None, max_turns: int = 5):
    """
    :param question: Pregunta actual del usuario
    :param chat_history: Lista de mensajes anteriores, cada uno con {"type": "human"/"ai", "content": "..."}
    :param max_turns: Número máximo de interacciones (pares human + ai) a conservar
    """
    # Formatear historial para ConversationalRetrievalChain
    formatted_history = []
    if chat_history:
        # Cada turno es un par (usuario, modelo) => 2 mensajes
        trimmed_history = chat_history[-(2 * max_turns):]

        for msg in trimmed_history:
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

