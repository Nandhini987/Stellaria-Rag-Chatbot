# Stellaria RAG Chatbot Backend (v1)

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.102-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Project Overview

This repository contains the **FastAPI backend** for the Stellaria RAG Chatbot project (v1).  
It implements a **Retrieval-Augmented Generation (RAG)** system with a local vector database, enabling semantic search over documents and serving responses via REST API endpoints.

---

## Features

- **REST API with FastAPI** for serving RAG responses
- **Vector database integration** for semantic search
- **Swagger UI** for easy endpoint testing (`/docs`)
- **Unit tests** using `pytest`
- **Clean project structure** separating backend, tests, and development scripts

---

## How It Works

1. **User sends a query** to the API endpoint.
2. **RAG backend** searches the vector database for relevant documents.
3. **Generative model** produces a response based on retrieved documents.
4. **API returns** the response to the user.

---

## Project Structure

RAG-backend-project/
│
├─ backend/ # Main backend code
│ ├─ app.py # FastAPI entry point
│ └─ rag_backend.py
│
├─ tests/ # Unit tests
│ └─ test_main.py
│
├─ dev/ # Temporary/debug scripts (ignored)
│ └─ tempCodeRunnerFile.py
│
├─ stell_db/ # Local vector DB (ignored)
├─ myenv/ # Virtual environment (ignored)
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ .env # Environment variables (ignored)
├─ stellchatbotver1(in).csv # Dataset (ignored)

## Quick Start

# Clone repo

git clone https://github.com/Nandhini987/Stellaria-Rag-Chatbot.git
cd Stellaria-Rag-Chatbot

# Setup virtual environment

python -m venv myenv
myenv\Scripts\activate # Windows

source myenv/bin/activate # macOS/Linux

# Install dependencies

pip install -r requirements.txt

# Run FastAPI app

uvicorn backend.app:app --reload

Run unit tests:

pytest tests/

# Notes

## Security & Maintenance

- **Environment Safety:** `.env` and virtual environments are ignored to protect sensitive API keys.
- **Clean Repository:** Temporary scripts and local databases (`stell_db/`) are excluded to maintain a professional, production-ready codebase.
- **Version Control:** Adheres to Git best practices by excluding unnecessary build artifacts and datasets.

## Future Work

- Deploy backend to cloud (AWS, Heroku, etc.)

- Expand vector database with more documents

- Add user authentication for API endpoints

## Author

Nandhini Mahesu
