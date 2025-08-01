from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

DATA_PATH = "data/Finanzas_Personales_Data.txt"
DB_PATH = "faiss_index"

def load_documents():
    loader = TextLoader(DATA_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_and_store():
    docs = load_documents()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

def get_rag_chain():
    db = load_vector_db()
    retriever = db.as_retriever()
    llm = Ollama(model="llama3:8b")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(query: str):
    llm = Ollama(model="llama3:8b") # Initialize LLM for scope check

    # Check for common greetings
    lower_query = query.lower()
    if any(greeting in lower_query for greeting in ["hola", "como estas", "qué tal", "buenos días", "buenas tardes", "buenas noches"]):
        return "¡Hola! Estoy aquí para ayudarte con tus preguntas sobre finanzas personales. ¿En qué puedo asistirte hoy?"

    # Use LLM to determine if the query is within the scope of personal finance
    # If not, the LLM will try to answer generally or redirect.
    llm_response_prompt = f"""Eres un asistente de IA especializado en finanzas personales.
Si la siguiente pregunta está directamente relacionada con finanzas personales (ahorro, presupuesto, inversión, deuda, crédito, etc.), responde a la pregunta utilizando tu conocimiento.
Si la pregunta es una consulta general, una pregunta sobre ti mismo, o una conversación casual, responde de forma natural y amigable.
Si la pregunta no es de finanzas personales y no puedes responderla de forma general, o si es una pregunta que requiere información específica que no tienes, redirige amablemente al usuario a temas de finanzas personales.

Pregunta: '{query}'
"""
    
    llm_general_response = llm.invoke(llm_response_prompt).strip()

    # A simple heuristic to decide if the LLM's response is a general answer or a redirection
    # This might need fine-tuning based on LLM behavior
    if "finanzas personales" in llm_general_response.lower() or "puedo ayudarte con" in llm_general_response.lower():
        # If the LLM's general response already mentions finance or redirection,
        # it means it's likely out of scope for RAG or already handled it.
        return llm_general_response
    
    # If the LLM didn't explicitly redirect or mention finance, try RAG
    # This assumes the LLM's general response was not a definitive answer
    # and the query might still be financial.
    rag_answer = get_rag_chain().run(query)
    
    # If RAG provides a meaningful answer, return it.
    # Otherwise, return the LLM's general response (which might be a polite "I can't help with that").
    if rag_answer and "no tengo información" not in rag_answer.lower() and "no sé" not in rag_answer.lower():
        return rag_answer
    else:
        return llm_general_response
