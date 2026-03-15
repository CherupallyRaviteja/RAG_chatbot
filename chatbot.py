import os

# --- Paddle / PaddleX hard suppression ---
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_LOG_LEVEL"] = "ERROR"
os.environ["FLAGS_minloglevel"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"


from rag_index import RAGIndex
from generator import generate_answer
from config import SIM_THRESHOLD
from query_rewriter import rewrite_query

def main():
    rag = RAGIndex()
    print("🤖 Agentic RAG Chatbot")

    while True:
        q = input("You: ").strip()
        
        if q == ":exit":
            break

        elif q.startswith(":add"):
            try:
                _, path = q.split(" ", 1)
                rag.add_pdf(path)
            except Exception as e:
                print(f"⚠️ Failed to add PDF: {e}")
            continue

        elif q == ":save":
            rag.save_to_db()
            continue

        else :
            q = rewrite_query(q)
            results = rag.retrieve(q)
            if not results or results[0][2] < SIM_THRESHOLD:
                print("Bot: I don't know.")
                continue

        contexts = []
        sources = []

        for content, source, page, score in results:
            contexts.append(content)
            sources.append((source, page))

        answer = generate_answer(q, contexts)

        print("Bot:", answer)
        print("\nSources:")
        for s, p in sources:
            print(f"{s} - Page {p}")
        print("- " * 30)
        
if __name__ == "__main__":
    main()
