import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import json
import requests
import re,os
import nltk
from docx import Document
nltk.download('punkt', quiet=True)

# ---------- CONFIG ----------
OLLAMA_URL = 'http://localhost:11434'
MODEL = 'phi3:mini'   # or 'mistral:7b'
EMBED_MODEL = 'all-MiniLM-L6-v2'
EMBED_DIM = 384
SIM_THRESHOLD = 0.3
FAISS_TOP_K = 4
SENT_SIM_THRESHOLD = 0.7

# PostgreSQL connection details
DB_CONFIG = {
    "dbname": "AgenticRAGDB",
    "user": "postgres",
    "password": os.environ.get("Postgres_Password"),
    "host": "localhost",
    "port": "5432"
}

def init_database():
    dbname = DB_CONFIG["dbname"]

    sys_conn = psycopg2.connect(
        dbname="postgres",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    sys_conn.autocommit = True
    cur = sys_conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{dbname}"')

    cur.close()
    sys_conn.close()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            content TEXT,
            embedding VECTOR({EMBED_DIM})
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

embed_model = SentenceTransformer(EMBED_MODEL)

# ---------- UTILITIES ----------
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

def docx_to_text(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# ---------- AGENTIC CHUNKING ----------
def agentic_chunking(text: str):
    """
    Agentic chunking dynamically determines chunk boundaries
    based on sentence embeddings and semantic similarity.
    """
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]
    if len(sents) == 0:
        return []

    embeddings = embed_model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
    chunks, current_chunk = [], [sents[0]]

    for i in range(1, len(sents)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if sim < SENT_SIM_THRESHOLD or len(" ".join(current_chunk).split()) > 250:
            # break chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sents[i]]
        else:
            current_chunk.append(sents[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # remove duplicates or too short chunks
    chunks = [c for c in chunks if len(c.split()) > 20]
    print(f"✂️  Agentic chunking created {len(chunks)} chunks.")
    return chunks

# ---------- RAG INDEX HANDLING ----------
class RAGIndex:
    def __init__(self):
        init_database()
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cur = self.conn.cursor()
        self.buffer = []  # temporary memory for unsaved embeddings

    def add_pdf(self, path: str):
        """Read PDF, chunk it, and prepare embeddings but don't save yet."""
        from pypdf import PdfReader
        print(f"\n📘 Processing PDF: {path}")
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]

        # Agentic chunking
        chunks = agentic_chunking(text)
        if not chunks:
            print("⚠️ No valid chunks extracted.")
            return

        print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        vectors = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

        # Store in buffer (not DB)
        for chunk_text, vector in zip(chunks, vectors):
            self.buffer.append({
                "source": path,
                "content": chunk_text,
                "embedding": vector
            })

        print(f"✅ Buffered {len(chunks)} chunks. Use ':save' to commit them to PostgreSQL.")

    def save_to_db(self):
        """Commit buffered embeddings to PostgreSQL."""
        if not self.buffer:
            print("⚠️ No new embeddings to save.")
            return

        print(f"💾 Saving {len(self.buffer)} chunks to PostgreSQL...")
        for item in tqdm(self.buffer, desc="Saving to DB"):
            vec_list = ",".join(map(str, item["embedding"].tolist()))
            self.cur.execute(
                "INSERT INTO documents (source, content, embedding) VALUES (%s, %s, %s::vector)",
                (item["source"], item["content"], f"[{vec_list}]")
            )
        self.conn.commit()
        self.buffer.clear()
        print("✅ All buffered chunks saved successfully.")
        
    def retrieve(self, query: str, top_k=4):
        q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
        vec_list = ",".join(map(str, q_vec.tolist()))
        self.cur.execute(
            """
            SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (f"[{vec_list}]", f"[{vec_list}]", top_k)
        )
        results = self.cur.fetchall()
        return [(float(r[2]), r[0], r[1]) for r in results]

# ---------- GENERATION ----------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_answer(query, contexts, scores=None, sim_threshold=0.35):
    """RAG answer generator that forbids pretrained knowledge."""

    # 1️⃣ Reject if no retrieved context
    if not contexts or len(contexts) == 0:
        return "I don't know."


    # 2️⃣ Reject if top retrieval score is too low
    if scores is not None and len(scores) > 0 and scores[0] < 0.25:
        return "I don't know."

    # 3️⃣ Join context
    context_text = "\n\n".join(contexts)

    # 4️⃣ Strict prompt (forbids pretraining usage)
    prompt = (
        "You are a retrieval-augmented assistant.\n"
        "Answer ONLY using the information in the context below.\n"
        "If the context does not contain the answer, reply exactly with:\n"
        "\"I don't know.\"\n\n"
        "You are strictly forbidden from using any pretrained or external knowledge.\n"
        "Never add facts, definitions, or reasoning not found verbatim in the context.\n"
        "Answer briefly (1–3 sentences max).\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # 5️⃣ Prepare body with temperature=0 (no creative recall)
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0}  # deterministic, context-bound
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=body, stream=True, timeout=180)
        if resp.status_code != 200:
            print(f"Ollama generation failed (HTTP {resp.status_code})")
            return "I don't know."

        answer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "response" in data:
                text = data["response"]
                answer += text
            if data.get("done"):
                break

        answer = answer.strip()
        if not answer or "i don't know" in answer.lower():
            return "I don't know."

        # 6️⃣ Post-generation validation: check similarity between context & answer
        ans_vec = embed_model.encode(answer, convert_to_numpy=True)
        ctx_vec = embed_model.encode(context_text, convert_to_numpy=True)
        sim = cosine_sim(ans_vec, ctx_vec)

        # 7️⃣ Block if the answer doesn’t semantically match context
        if sim < sim_threshold:
            print(f"⚠️ Context–answer similarity too low ({sim:.2f}), discarding.")
            return "I don't know."

        return answer

    except Exception as e:
        print("⚠️ Ollama connection failed:", e)
        return "I don't know."

# ---------- MAIN CHAT LOOP ----------
def main():
    rag = RAGIndex()

    print("\n🤖 Terminal RAG Chatbot (Agentic Chunking + DB-Only Mode)")
    print("Commands:")
    print("  :add <pdf_path>   → Add and index a PDF (kept in memory)")
    print("  :save             → Save buffered PDFs to PostgreSQL")
    print("  :exit             → Quit chatbot\n")

    while True:
        query = input("🧑 You: ").strip()
        if not query:
            continue

        if query.startswith(":add"):
            try:
                _, path = query.split(" ", 1)
                rag.add_pdf(path)
            except Exception as e:
                print(f"⚠️ Failed to add PDF: {e}")
            continue

        if query == ":save":
            rag.save_to_db()
            continue

        if query == ":exit":
            print("👋 Goodbye!")
            break

        # Retrieval + generation (strictly DB-based)
        results = rag.retrieve(query)
        if not results:
            print("🤖 Bot: I don't know.\n")
            continue

        top_score = results[0][0]
        if top_score < SIM_THRESHOLD:
            print("🤖 Bot: Low similarity. Provide more details or clarify your question.\n")
            continue

        contexts = [r[1] for r in results]
        sources = [r[2] for r in results]
        scores = [r[0] for r in results]
        answer = generate_answer(query, contexts, scores=scores)
        print(f"🤖 Bot: {answer}")
        if "i don't know" in answer.lower():
            pass
        else:
            print(f"📄 Source: {sources[0]}\n")
if __name__ == "__main__":
    main()
