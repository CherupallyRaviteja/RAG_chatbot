import numpy as np
import requests, json
from config import OLLAMA_URL, MODEL, embed_model
"""
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_answer(query, contexts):
    if not contexts:
        return "I don't know."

    context_text = "\n\n".join(contexts)
    prompt = f"Context:\n{context_text}\n\nQuestion:{query}\nAnswer:"

    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body)
    return r.json().get("response", "I don't know.")

"""

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

if __name__ == "__main__":
    print("🧠 Generator ready")
