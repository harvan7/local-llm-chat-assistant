from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
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
    # Ollama Embeddings
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

def load_vector_db():
    # Ollama Embeddings
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

def get_rag_chain():
    db = load_vector_db()
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Memory for conversational context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ConversationalRetrievalChain handles chat history and question rephrasing
    # The system message for Spanish responses will be part of the overall prompt
    # within the ConversationalRetrievalChain's question generator.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # This part is crucial for ensuring the question generator also respects Spanish
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

    # Check for common greetings
    lower_query = query.lower()
    if any(greeting in lower_query for greeting in ["hola", "como estas", "qué tal", "buenos días", "buenas tardes", "buenas noches"]):
        return "¡Hola! Estoy aquí para ayudarte con tus preguntas sobre finanzas personales. ¿En qué puedo asistirte hoy?"

    # Use LLM to determine if the query is within the scope of personal finance
    # If not, the LLM will try to answer generally or redirect.
    llm_response_prompt = f'''Eres un asistente de IA especializado en finanzas personales. Responde siempre en español.
Si la siguiente pregunta está directamente relacionada con finanzas personales (ahorro, presupuesto, inversión, deuda, crédito, etc.), responde a la pregunta utilizando tu conocimiento y los documentos proporcionados.
Si la pregunta incluye cálculos numéricos específicos o requiere un plan financiero personalizado (ej. "tengo una deuda X a interes del x% quiero disminuirla con ingresos de Y mensuales"), explica los conceptos financieros relevantes (ej. métodos de reducción de deuda como bola de nieve o avalancha) y sugiere al usuario que utilice una calculadora financiera o consulte a un profesional para obtener cifras exactas y un plan adaptado a su situación.
Si la pregunta es una consulta general, una pregunta sobre ti mismo, o una conversación casual, responde de forma natural y amigable.
Si la pregunta no es de finanzas personales y no puedes responderla de forma general, o si es una pregunta que requiere información específica que no tienes, redirige amablemente al usuario a temas de finanzas personales.

Pregunta: '{query}'
'''
    
    llm_general_response = llm.invoke(llm_response_prompt).content.strip()

    # A simple heuristic to decide if the LLM's response is a general answer or a redirection
    # This might need fine-tuning based on LLM behavior
    if "finanzas personales" in llm_general_response.lower() or "puedo ayudarte con" in llm_general_response.lower():
        # If the LLM's general response already mentions finance or redirection,
        # it means it's likely out of scope for RAG or already handled it.
        return llm_general_response
    
    # If the LLM didn't explicitly redirect or mention finance, try RAG
    # This assumes the LLM's general response was not a definitive answer
    # and the query might still be financial.
    rag_chain = get_rag_chain()
    rag_answer = rag_chain.invoke({"question": query, "chat_history": chat_history})
    
    # If RAG provides a meaningful answer, return it.
    # Otherwise, return the LLM's general response (which might be a polite "I can't help with that").
    if rag_answer and "no tengo información" not in rag_answer['answer'].lower() and "no sé" not in rag_answer['answer'].lower():
        return rag_answer['answer']
    else:
        return llm_general_response
