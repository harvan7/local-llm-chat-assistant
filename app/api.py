from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.rag_engine import answer_question
import os
from dotenv import load_dotenv

# --- Cargar variables de entorno ---
load_dotenv()

app = FastAPI()

# --- Configuración CORS ---
origins = ["http://localhost:3000"]  # Ajusta según tu frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Seguridad ---
security = HTTPBearer()

def get_api_key():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("La variable API_KEY no está configurada.")
    return api_key

# --- Modelos de datos ---
class ChatMessage(BaseModel):
    type: str  # "human" o "ai"
    content: str

class Question(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []

# --- Autenticación ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), api_key: str = Depends(get_api_key)):
    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Token inválido")
    return True

# --- Endpoints ---
@app.post("/ask")
def ask_question(q: Question, creds: bool = Depends(verify_token)):
    """
    Endpoint principal para hacer preguntas al sistema RAG con Gemini.
    """
    # Convertir historial de Pydantic a lista de dicts
    chat_history_dicts = [msg.dict() for msg in q.chat_history]

    # Obtener respuesta del motor RAG
    response = answer_question(q.query, chat_history_dicts)

    return {"response": response}

@app.get("/")
def root():
    return {"message": "WealthAdvisor is running"}

@app.get("/robots933456.txt")
def robots():
    return ""
