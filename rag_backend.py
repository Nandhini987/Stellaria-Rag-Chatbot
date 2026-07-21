import os
import warnings
import pandas as pd
from dotenv import load_dotenv
import requests
from io import StringIO
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from google import genai
from google.genai import types

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
DB_PATH = "./stell_db"
COLLECTION_NAME = "club_collection_v1"

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

gemini_client = genai.Client(
    api_key=API_KEY,
    #http_options=types.HttpOptions(api_version="v1")
)

# =========================================================
# EMBEDDING FUNCTION
# =========================================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[str(x) for x in input],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        # Ensure clean float list output
        return [list(embedding.values) for embedding in response.embeddings]


embedding_function = GeminiEmbeddingFunction(gemini_client)

# =========================================================
# CHROMA DB SETUP
# =========================================================
chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}  # IMPORTANT FIX
)

# =========================================================
# SYNC DATABASE
# =========================================================
def sync_database():
    """Loads dataset from Google Drive and syncs ChromaDB."""

    dataset_url = os.getenv("DATASET_URL")

    if not dataset_url:
        print("DATASET_URL not configured.")
        return

    try:
        response = requests.get(dataset_url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text)).fillna("")
        print(f"Dataset loaded successfully: {len(df)} rows")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Combine text for embeddings
    df["combined_text"] = (
        "Question: " + df["question"].astype(str) +
        " Answer: " + df["answer"].astype(str)
    )

    df = df[df["combined_text"].str.strip() != ""]

    documents = df["combined_text"].tolist()

    metadatas = [
        {"category": str(t)}
        for t in df.get("type", ["general"] * len(df)).tolist()
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Database synced: {len(documents)} records processed.")

# =========================================================
# RETRIEVAL
# =========================================================
def retrieve_context(question, top_k=3):
    query_resp = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    query_vector = list(query_resp.embeddings[0].values)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else ""

# =========================================================
# GENERATION
# =========================================================
def generate_response(question, context):
    prompt = f"""
You are the Stellaria Club Assistant.

Instructions:
- Use ONLY the provided context to answer.
- If context does not clearly contain the answer, say you are not sure.
- Keep answers concise (2-4 sentences).
- If user greets you, respond politely.

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

# =========================================================
# MAIN RAG FUNCTION
# =========================================================
def ask_stellaria(question):
    try:
        context = retrieve_context(question)

        # safer check (but still allows weak context usage)
        if context.strip() == "":
            return "I'm not sure about that. Please check with the Stellaria team."

        return generate_response(question, context)

    except Exception as e:
        print(f"RAG Error: {e}")
        return "An error occurred while fetching the answer."