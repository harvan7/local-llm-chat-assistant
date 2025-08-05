from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from app.rag_engine import answer_question
import os
from dotenv import load_dotenv

# --- Cargar variables de entorno ---
load_dotenv()

app = FastAPI()

# --- CORS ---
origins = ["http://localhost:3000"]
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
    return os.getenv("API_KEY")

# --- Modelos de datos ---
class ChatMessage(BaseModel):
    type: str  # "human" o "ai"
    content: str

class Question(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = None

# --- Autenticación ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), api_key: str = Depends(get_api_key)):
    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Token inválido")

# --- Endpoints ---
@app.post("/ask")
def ask_question(q: Question, creds: HTTPAuthorizationCredentials = Depends(verify_token)):
    chat_history_dicts = [msg.dict() for msg in q.chat_history] if q.chat_history else None
    response = answer_question(q.query, chat_history_dicts)
    return {"response": response}

@app.get("/")
def root():
    return {"message": "WealthAdvisor is running"}

@app.get("/robots933456.txt")
def robots():
    return ""
