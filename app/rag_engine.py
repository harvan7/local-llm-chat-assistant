from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

DATA_PATH = "data/Finanzas_Personales_Data.txt"
DB_PATH = "faiss_index"

def load_documents():
    loader = TextLoader(DATA_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_and_store():
    docs = load_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

def load_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

def get_rag_chain(chat_history=[]):
    db = load_vector_db()
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Convert chat_history from [{"type": "human"/"ai", "content": "..."}] to LangChain messages
    lc_history = []
    for msg in chat_history:
        if msg["type"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] in ["ai", "assistant"]:
            lc_history.append(AIMessage(content=msg["content"]))

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages = lc_history

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Eres un asistente de IA especializado en finanzas personales. Responde siempre en español."
                ),
                ("{context}", "user"),
                ("{question}", "user")
            ])
        }
    )
    return qa_chain

def answer_question(query: str, chat_history: list = []):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Check greetings
    lower_query = query.lower()
    if any(greeting in lower_query for greeting in ["hola", "como estas", "qué tal", "buenos días", "buenas tardes", "buenas noches"]):
        return "¡Hola! Estoy aquí para ayudarte con tus preguntas sobre finanzas personales. ¿En qué puedo asistirte hoy?"

    # Pre-filter query using LLM
    llm_response_prompt = f'''Eres un asistente de IA especializado en finanzas personales. Responde siempre en español.
Si la siguiente pregunta está directamente relacionada con finanzas personales (ahorro, presupuesto, inversión, deuda, crédito, etc.), responde a la pregunta utilizando tu conocimiento y los documentos proporcionados.
Si la pregunta incluye cálculos numéricos específicos o requiere un plan financiero personalizado (ej. "tengo una deuda X a interes del x% quiero disminuirla con ingresos de Y mensuales"), explica los conceptos financieros relevantes y sugiere al usuario que utilice una calculadora financiera o consulte a un profesional para obtener cifras exactas.
Si la pregunta es general o casual, responde de forma natural y amigable.
Si no es de finanzas personales y no puedes responderla de forma general, redirige al usuario a temas de finanzas personales.

Pregunta: '{query}'
'''
    llm_general_response = llm.invoke(llm_response_prompt).content.strip()

    if "finanzas personales" in llm_general_response.lower() or "puedo ayudarte con" in llm_general_response.lower():
        return llm_general_response

    # Run RAG with chat history
    rag_chain = get_rag_chain(chat_history)
    rag_answer = rag_chain.invoke({"question": query, "chat_history": chat_history})

    if rag_answer and "no tengo información" not in rag_answer['answer'].lower() and "no sé" not in rag_answer['answer'].lower():
        return rag_answer['answer']
    else:
        return llm_general_response
