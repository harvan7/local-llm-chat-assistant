from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

DATA_PATH = "data/Finanzas_Personales_Data.txt"
DB_PATH = "faiss_index"

# --- 1. Load and split documents ---
def load_documents():
    loader = TextLoader(DATA_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# --- 2. Embed and store vectors ---
def embed_and_store():
    docs = load_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

# --- 3. Load vector database ---
def load_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# --- 4. Build RAG chain ---
def get_rag_chain():
    db = load_vector_db()
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Updated memory handling
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Updated prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Eres un asistente de IA especializado en finanzas personales. Responde siempre en español.\n\n"
            "Usa el siguiente contexto cuando sea relevante:\n{context}\n\n"
            "Pregunta:\n{question}"
        )
    )

    # Updated ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return qa_chain

# --- 5. Handle user query ---
def answer_question(query: str, chat_history: list = []):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Detect greetings
    lower_query = query.lower()
    if any(greeting in lower_query for greeting in [
        "hola", "cómo estás", "que tal", "qué tal",
        "buenos días", "buenas tardes", "buenas noches"
    ]):
        return "¡Hola! Estoy aquí para ayudarte con tus preguntas sobre finanzas personales. ¿En qué puedo asistirte hoy?"

    # Classify query using LLM
    llm_response_prompt = f'''Eres un asistente de IA especializado en finanzas personales. Responde siempre en español.
Si la siguiente pregunta está directamente relacionada con finanzas personales (ahorro, presupuesto, inversión, deuda, crédito, etc.), responde a la pregunta utilizando tu conocimiento y los documentos proporcionados.
Si la pregunta incluye cálculos numéricos específicos o requiere un plan financiero personalizado (ej. "tengo una deuda X a interés del x% quiero disminuirla con ingresos de Y mensuales"), explica los conceptos financieros relevantes (ej. métodos de reducción de deuda como bola de nieve o avalancha) y sugiere al usuario que utilice una calculadora financiera o consulte a un profesional para obtener cifras exactas y un plan adaptado a su situación.
Si la pregunta es una consulta general, una pregunta sobre ti mismo, o una conversación casual, responde de forma natural y amigable.
Si la pregunta no es de finanzas personales y no puedes responderla de forma general, o si es una pregunta que requiere información específica que no tienes, redirige amablemente al usuario a temas de finanzas personales.

Pregunta: '{query}'
'''
    llm_general_response = llm.invoke(llm_response_prompt).content.strip()

    # If LLM already redirects or classifies out of scope, return it
    if "finanzas personales" in llm_general_response.lower() or "puedo ayudarte con" in llm_general_response.lower():
        return llm_general_response

    # Otherwise, try RAG
    rag_chain = get_rag_chain()
    rag_answer = rag_chain.invoke({"question": query, "chat_history": chat_history})

    # If RAG returns a meaningful answer, return it; else return LLM general response
    if rag_answer and "no tengo información" not in rag_answer['answer'].lower() and "no sé" not in rag_answer['answer'].lower():
        return rag_answer['answer']
    else:
        return llm_general_response
