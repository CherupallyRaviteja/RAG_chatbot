from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from config import embed_model, SENT_SIM_THRESHOLD

def agentic_chunking(text):
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sents:
        return []

    embeddings = embed_model.encode(sents, convert_to_tensor=True)
    chunks, current = [], [sents[0]]

    for i in range(1, len(sents)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        if sim < SENT_SIM_THRESHOLD or len(" ".join(current).split()) > 250:
            chunks.append(" ".join(current))
            current = [sents[i]]
        else:
            current.append(sents[i])

    chunks.append(" ".join(current))
    chunks = [c for c in chunks if len(c.split()) > 20]
    print(f"✂️ Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    agentic_chunking("This is a test sentence. Another related sentence.")
