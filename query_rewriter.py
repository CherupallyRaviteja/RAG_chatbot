import requests
from config import OLLAMA_URL, MODEL

def rewrite_query(query: str) -> str:
    """
    Uses Ollama to fix spelling, grammar and clarity.
    Returns a rewritten query.
    """

    prompt = f"""
You are a query rewriting assistant.

Your job:
- Correct spelling mistakes
- Fix grammar
- Rewrite the query clearly
- Keep the meaning identical
- Return ONLY the corrected query

Query:
{query}

Rewritten query:
"""

    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=60)
        result = r.json()["response"].strip()
        return result

    except Exception as e:
        print("Rewrite error:", e)
        return query

if __name__ == "__main__":
    query=input("Enter your query: ")
    rewritten_query = rewrite_query(query)
    print("Rewritten query:", rewritten_query)