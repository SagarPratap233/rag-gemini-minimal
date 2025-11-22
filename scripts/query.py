import os
from dotenv import load_dotenv
from chromadb import PersistentClient
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load local Chroma DB
chroma = PersistentClient(path="chroma_store")
collection = chroma.get_collection("docs")

def embed_query(q: str):
    """Embed the user query using Gemini."""
    res = genai.embed_content(
        model="text-embedding-004",
        content=q
    )
    return res["embedding"]

def ask(q: str):
    """Main RAG pipeline."""
    q_emb = embed_query(q)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )

    docs = results["documents"]

    if not docs or len(docs[0]) == 0:
        print("⚠️ No matching documents found.")
        return

    context = "\n\n".join(docs[0])

    prompt = f"""
Use ONLY the following context to answer.
If the answer is not found in context, say: "Not found in documents."

CONTEXT:
{context}

QUESTION:
{q}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    print("\nANSWER:\n", response.text)

if __name__ == "__main__":
    print("RAG system ready. Ask anything.\n")
    while True:
        q = input("\nAsk: ")
        ask(q)
