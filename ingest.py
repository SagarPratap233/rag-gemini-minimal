import os
from dotenv import load_dotenv
from uuid import uuid4
import google.generativeai as genai
from chromadb import PersistentClient

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create Chroma client the NEW WAY (no Settings)
from chromadb import PersistentClient

chroma = PersistentClient(path="chroma_store")
collection = chroma.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)


def embed_text(text: str):
    """Generate embedding using Gemini."""
    res = genai.embed_content(
        model="text-embedding-004",
        content=text
    )
    return res["embedding"]

def ingest_folder():
    folder = "data"
    if not os.path.exists(folder):
        print("❌ ERROR: 'data' folder does not exist")
        return

    files = os.listdir(folder)
    if not files:
        print("⚠️ No files found inside /data")
        return

    for file in files:
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            continue

        print(f"📄 Reading: {file}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk text into 800-char segments
        chunks = [text[i:i+800] for i in range(0, len(text), 800)]

        for chunk in chunks:
            collection.add(
                ids=[str(uuid4())],
                documents=[chunk],
                embeddings=[embed_text(chunk)]
            )

        print(f"✅ Ingested {file}: {len(chunks)} chunks")

if __name__ == "__main__":
    ingest_folder()
    print("\n🎉 All documents indexed successfully!")
