from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.rag_engine import answer_question
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
security = HTTPBearer()

def get_api_key():
    return os.getenv("API_KEY")

class Question(BaseModel):
    query: str

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), api_key: str = Depends(get_api_key)):
    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Token inválido")

@app.post("/ask")
def ask_question(q: Question, creds: HTTPAuthorizationCredentials = Depends(verify_token)):
    response = answer_question(q.query)
    return {"response": response}

# Nueva ruta raíz para pruebas y monitoreo
@app.get("/")
def root():
    return {"message": "WealthAdvisor is running"}

# Ruta opcional para evitar el 404 en robots933456.txt
@app.get("/robots933456.txt")
def robots():
    return ""
