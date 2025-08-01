# preguntar.py
import requests
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

API_KEY = os.getenv("API_KEY") # Get API key from environment variables

if not API_KEY:
    print("Error: API_KEY not found in .env file.")
    exit()

question = input("Pregunta: ")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.post("http://127.0.0.1:8000/ask", json={"query": question}, headers=headers)

if response.status_code == 200:
    print("Respuesta:", response.json()["response"])
else:
    print(f"Error: {response.status_code} - {response.text}")