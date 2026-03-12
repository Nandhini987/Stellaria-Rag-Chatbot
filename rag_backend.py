import os
import warnings
import pandas as pd
from dotenv import load_dotenv

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from chromadb.config import Settings

from google import genai
from google.genai import types

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
CSV_FILE = "stellchatbotver1(in).csv"
DB_PATH = "./stell_db"
COLLECTION_NAME = "club_collection_v1"

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

gemini_client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(api_version="v1beta")
)

# =========================================================
# CLASSES & FUNCTIONS
# =========================================================

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[str(x) for x in input],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [e.values for e in response.embeddings]

embedding_function = GeminiEmbeddingFunction(gemini_client)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

def sync_database():
    """Reads CSV and performs an upsert to keep the vector store fresh."""
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found.")
        return
        
    df = pd.read_csv(CSV_FILE).fillna("")
    df["combined_text"] = "Question: " + df["question"] + " Answer: " + df["answer"]
    df = df[df["combined_text"].str.strip() != ""]

    documents = df["combined_text"].astype(str).tolist()
    metadatas = [{"category": str(t)} for t in df.get("type", ["general"] * len(df)).tolist()]
    ids = [str(i) for i in range(len(documents))]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Database synced: {len(documents)} records processed.")

def retrieve_context(question, top_k=3):
    query_resp = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = query_resp.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_vector], 
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else ""

def generate_response(question, context):
    prompt = f"""
You are the Stellaria Club Assistant.
Instructions:
- If the user greets you, greet them politely.
- Use the provided context to answer questions.
- Keep the answer concise (2-4 sentences).
- If the context does not contain the answer, say:
"I'm not sure about that. Please check with the Stellaria team."

Context:
{context}

User Question:
{question}
"""
    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    return response.text

def ask_stellaria(question):
    try:
        context = retrieve_context(question)
        if not context:
            return "I'm sorry, I don't have information on that."
        return generate_response(question, context)
    except Exception as e:
        print(f"RAG Error: {e}")
        return "An error occurred while fetching the answer."