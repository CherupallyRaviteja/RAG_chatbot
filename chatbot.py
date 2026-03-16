from rag_index import RAGIndex
from generator import generate_answer
from config import SIM_THRESHOLD
from query_rewriter import rewrite_query

def main():
    rag = RAGIndex()
    print("🤖 Agentic RAG Chatbot")
    print("Commands:")
    print(":exit - Exit the chatbot")
    print(":add <file_path> - Add a new document (PDF, DOCX, or image)")
    print(":save - Save the current state to the database\n")
    print("- " * 30)

    while True:
        q = input("You: ").strip()
        
        if q == ":exit":
            break

        elif q.startswith(":add"):
            try:
                _, path = q.split(" ", 1)
                rag.add_pdf(path)
            except Exception as e:
                print(f"⚠️ Failed to add Document: {e}")
            continue

        elif q == ":save":
            rag.save_to_db()
            continue

        else :
            q = rewrite_query(q)
            results = rag.retrieve(q)
            print(f"Retrieved {len(results)} results. Top score: {results[0][3] if results else 'N/A'}")
            if not results or results[0][3] < SIM_THRESHOLD:
                print("Bot: I don't know.")
                continue

        contexts = []
        sources = []

        for content, source, page, score, *_ in results:
            contexts.append(content)
            sources.append((source, page))

        answer = generate_answer(q, contexts)

        print("Bot:", answer)
        print("\nSources:")
        for s, p in set(sources):
            print(f"{s} - Page {p}")
        print("- " * 30)
        
if __name__ == "__main__":
    main()
