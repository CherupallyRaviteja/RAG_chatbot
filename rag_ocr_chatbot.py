#import required modules 

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
from pdf2image import convert_from_path
from ocr_text import ocr_image_text_and_tables  
import tempfile
from docx.oxml.ns import qn
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import fitz
from PIL import Image
from io import BytesIO
import cv2

nltk.download('punkt', quiet=True)

# ---------- CONFIG ----------
OLLAMA_URL = 'http://localhost:11434'
MODEL = 'phi3:mini'   
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

import os

POPPLER_BIN = r"C:\poppler\Library\bin"

if POPPLER_BIN not in os.environ["PATH"]:
    os.environ["PATH"] = POPPLER_BIN + os.pathsep + os.environ["PATH"]

#Intializing Database if not exist 
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

def pad_image(img, pad=20):
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def pdf_to_text_with_inline_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = []

    for page in doc:
        items = []

        # 1️⃣ Extract text blocks with positions
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, _, block_type = b
            if block_type == 0 and text.strip():  # text block
                items.append((y0, x0, text.strip()))

        # 2️⃣ Extract images with positions
        for img in page.get_images(full=True):
            xref = img[0]

            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                continue

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert to NumPy
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)

            # 🔧 Padding to avoid missing first character
            image_np = pad_image(image_np, pad=20)

            ocr_text = ocr_image_text_and_tables(image_np)
            if ocr_text and ocr_text.strip():
                items.append((bbox.y0, bbox.x0, ocr_text.strip()))

        # 3️⃣ Sort everything by reading order (top → bottom, left → right)
        items.sort(key=lambda x: (x[0], x[1]))

        # 4️⃣ Append to final text
        for _, _, content in items:
            final_text.append(content)

    doc.close()
    return "\n\n".join(final_text)

def pdf_has_images(pdf_path):
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        resources = page.get("/Resources")
        if not resources:
            continue

        xobjects = resources.get("/XObject")
        if not xobjects:
            continue

        for obj in xobjects.values():
            try:
                if obj.get("/Subtype") == "/Image":
                    return True
            except:
                pass

    return False


def docx_to_text_with_exact_inline_ocr(docx_path):
    """
    Extracts DOCX text and OCRs images at their exact inline positions.
    Uses in-memory OCR (no temp files).
    """

    doc = Document(docx_path)
    final_text = []

    for para in doc.paragraphs:
        for run in para.runs:
            # 1️⃣ Normal text
            if run.text:
                final_text.append(run.text)

            # 2️⃣ Detect inline image inside run
            drawings = run._element.xpath('.//w:drawing')
            if not drawings:
                continue

            for d in drawings:
                blip = d.xpath('.//a:blip')
                if not blip:
                    continue

                embed = blip[0].get(qn('r:embed'))
                image_part = doc.part.related_parts.get(embed)
                if not image_part:
                    continue

                image_bytes = image_part.blob

                # 3️⃣ Convert image → NumPy (PaddleOCR compatible)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_np = np.array(image)

                # 🔧 Padding fixes missing first character
                image_np = pad_image(image_np, pad=20)

                # 4️⃣ OCR
                ocr_text = ocr_image_text_and_tables(image_np)
                if ocr_text and ocr_text.strip():
                    final_text.append(ocr_text.strip())

        final_text.append("\n")  # paragraph break

    return "".join(final_text)

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_to_text_with_inline_ocr(path)
    if ext == ".docx":
        return docx_to_text_with_exact_inline_ocr(path)
    raise ValueError("Unsupported file type")

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
        print(f"\n📘 Processing PDF: {path}")
        text = load_document(path)
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