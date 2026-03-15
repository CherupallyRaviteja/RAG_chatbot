import psycopg2
from tqdm import tqdm
from config import DB_CONFIG, embed_model
from db_init import init_database
from document_loader import load_document
from chunking import agentic_chunking

class RAGIndex:
    def __init__(self):
        init_database()
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cur = self.conn.cursor()
        self.buffer = []

    def add_pdf(self, path):
        pages = load_document(path)

        for page_num, page_text in pages:

            chunks = agentic_chunking(page_text)

            vectors = embed_model.encode(chunks, convert_to_numpy=True)

            for c, v in zip(chunks, vectors):
                self.buffer.append({
                    "source": path,
                    "page": page_num,
                    "content": c,
                    "embedding": v
                })
            print("PDF ADDED ✅")

    def save_to_db(self):
        for item in tqdm(self.buffer):
            vec = ",".join(map(str, item["embedding"].tolist()))
            self.cur.execute(
                "INSERT INTO documents (source, page, content, embedding) VALUES (%s,%s,%s,%s::vector)",
                (item["source"], item["page"], item["content"], f"[{vec}]")
            )
        self.conn.commit()
        self.buffer.clear()
        print("DB SAVED ✅")

    def retrieve(self, query, top_k=4):
        qv = embed_model.encode([query], convert_to_numpy=True)[0]
        vec = ",".join(map(str, qv.tolist()))
        self.cur.execute("""
            SELECT content, source, page, 1-(embedding <=> %s::vector)
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """, (f"[{vec}]", f"[{vec}]", top_k))
        return self.cur.fetchall()

if __name__ == "__main__":
    print("✅ RAG index ready")
