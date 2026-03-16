from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from config import embed_model, SENT_SIM_THRESHOLD
import re
def watson_chunking(text, max_words=250):
    blocks, current = [], []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines:
        if re.match(r"^[A-Z][A-Za-z\s]{3,}$", line) or re.match(r"^\d+[\.\)]", line):
            if current:
                blocks.append(" ".join(current))
                current = []
        current.append(line)
        if len(" ".join(current).split()) > max_words:
            blocks.append(" ".join(current))
            current = []

    if current:
        blocks.append(" ".join(current))

    chunks = [c for c in blocks if len(c.split()) > 5 or "\n" in c]
    return chunks

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
    chunks = [c for c in chunks if len(c.split()) > 5 or "\n" in c]
    print(f"✂️ Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":

    pdf_path = "Mineral_Resources.pdf"
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    chunks = watson_chunking(text)
        
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)  # Print the first 500 characters of each chunk
        print("\n")
