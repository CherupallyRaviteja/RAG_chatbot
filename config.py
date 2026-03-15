import os
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)

OLLAMA_URL = 'http://localhost:11434'
MODEL = 'phi3:mini'
EMBED_MODEL = 'all-MiniLM-L6-v2'
EMBED_DIM = 384
SIM_THRESHOLD = 0.3
FAISS_TOP_K = 4
SENT_SIM_THRESHOLD = 0.7

DB_CONFIG = {
    "dbname": "AgenticRAGDBPages",
    "user": "postgres",
    "password": "Gmail.com#1",#os.environ.get("Postgres_Password"),
    "host": "localhost",
    "port": "5432"
}

POPPLER_BIN = r"C:\poppler\Library\bin"
if POPPLER_BIN not in os.environ["PATH"]:
    os.environ["PATH"] = POPPLER_BIN + os.pathsep + os.environ["PATH"]

embed_model = SentenceTransformer(EMBED_MODEL)

if __name__ == "__main__":
    print("✅ Config loaded")
    print("Model:", MODEL)
    print("Embedding dim:", EMBED_DIM)
