from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_backend import ask_stellaria, sync_database
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs once when you start the server
    sync_database()
    yield

app = FastAPI(
    title="Stellaria Chatbot API",
    description="RAG-powered chatbot backend for Stellaria Club",
    version="1.0",
    lifespan=lifespan
)

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Stellaria Chatbot API is running"}

@app.post("/ask")
def ask(query: Query):
    answer = ask_stellaria(query.question)
    return {
        "question": query.question,
        "answer": answer
    }