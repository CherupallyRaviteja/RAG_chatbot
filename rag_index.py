import psycopg2
from tqdm import tqdm
from config import DB_CONFIG, embed_model
from db_init import init_database
from chunking import agentic_chunking, watson_chunking
from document_loader import test_ocr

class RAGIndex:
    def __init__(self):
        init_database()
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cur = self.conn.cursor()
        self.buffer = []

    def add_pdf(self, path):
        pages = test_ocr(path)
        total_chunks = 0
        for page_num, page_text in pages:

            chunks = watson_chunking(page_text)
            total_chunks += len(chunks)
            vectors = embed_model.encode(chunks, convert_to_numpy=True)

            for c, v in zip(chunks, vectors):
                self.buffer.append({
                    "source": path,
                    "page": page_num,
                    "content": c,
                    "embedding": v
                })
        
        print(f"PDF ADDED ✅ (Total chunks: {total_chunks})")

    def save_to_db(self):
        for item in tqdm(self.buffer):
            vec = ",".join(map(str, item["embedding"].tolist()))
            self.cur.execute(
            """
            INSERT INTO documents (source, page, content, embedding, tsv)
            VALUES (%s,%s,%s,%s::vector,to_tsvector('english',%s))
            """,
            (item["source"], item["page"], item["content"], f"[{vec}]", item["content"])
            )
        self.conn.commit()
        self.buffer.clear()
        print("DB SAVED ✅")


    def retrieve(self, query, top_k=5):
        qv = embed_model.encode([query], convert_to_numpy=True)[0]
        vec = ",".join(map(str, qv.tolist()))

        self.cur.execute("""
        SELECT content, source, page,
            1-(embedding <=> %s::vector) AS vec_score,
            ts_rank(tsv, plainto_tsquery(%s)) AS text_score
        FROM documents
        WHERE tsv @@ plainto_tsquery(%s)
        OR embedding <=> %s::vector < 0.8
        ORDER BY (0.6 * (1-(embedding <=> %s::vector)) +
                0.4 * ts_rank(tsv, plainto_tsquery(%s))) DESC
        LIMIT %s
        """,
        (f"[{vec}]", query, query, f"[{vec}]", f"[{vec}]", query, top_k))

        return self.cur.fetchall()
"""
def retrieve(self, query, top_k=4):
        qv = embed_model.encode([query], convert_to_numpy=True)[0]
        vec = ",".join(map(str, qv.tolist()))
        self.cur.execute(""
            SELECT content, source, page, 1-(embedding <=> %s::vector)
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            "", (f"[{vec}]", f"[{vec}]", top_k))
        return self.cur.fetchall()
"""
    

if __name__ == "__main__":
    print("✅ RAG index ready")
