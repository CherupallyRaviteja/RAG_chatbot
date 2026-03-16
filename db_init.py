import psycopg2
from config import DB_CONFIG, EMBED_DIM

def init_database():
    dbname = DB_CONFIG["dbname"]

    # connect to default postgres DB
    sys_conn = psycopg2.connect(
        dbname="postgres",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    sys_conn.autocommit = True
    cur = sys_conn.cursor()

    # create database if not exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{dbname}"')

    cur.close()
    sys_conn.close()

    # connect to project DB
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # create table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            page INTEGER,
            content TEXT,
            embedding VECTOR({EMBED_DIM}),
            tsv tsvector
        )
    """)

    # index for vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding
        ON documents
        USING ivfflat (embedding vector_cosine_ops)
    """)

    # index for keyword search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tsv
        ON documents
        USING GIN (tsv)
    """)

    conn.commit()
    cur.close()
    conn.close()
    
if __name__ == "__main__":
    init_database()
    print("✅ Database initialized")
