from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Stellaria Chatbot API is running"}

def test_ask_endpoint():
    # Note: This will hit the real Gemini API unless mocked
    response = client.post("/ask", json={"question": "What is Stellaria?"})
    assert response.status_code == 200
    assert "answer" in response.json()