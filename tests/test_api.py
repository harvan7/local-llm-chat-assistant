from fastapi.testclient import TestClient
from app.api import app, get_api_key
import pytest

@pytest.fixture
def mock_answer_question(mocker):
    return mocker.patch("app.api.answer_question", return_value="Respuesta de prueba")

client = TestClient(app)

def get_test_api_key():
    return "testkey"

app.dependency_overrides[get_api_key] = get_test_api_key

def test_question_valid(mock_answer_question):
    headers = {"Authorization": "Bearer testkey"}
    data = {"query": "¿Cómo hacer un presupuesto?"}
    response = client.post("/ask", json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"response": "Respuesta de prueba"}

def test_invalid_token():
    headers = {"Authorization": "Bearer wrong"}
    data = {"query": "¿Qué es Python?"}
    response = client.post("/ask", json=data, headers=headers)
    assert response.status_code == 403

def test_out_of_scope_question(mock_answer_question):
    mock_answer_question.return_value = "Lo siento, solo puedo ayudarte con temas de finanzas personales."
    headers = {"Authorization": "Bearer testkey"}
    data = {"query": "¿Cuál es la capital de Francia?"}
    response = client.post("/ask", json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"response": "Lo siento, solo puedo ayudarte con temas de finanzas personales."}
