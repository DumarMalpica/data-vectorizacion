import os
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

chroma_client = PersistentClient(path="data")
collection = chroma_client.get_or_create_collection("docs")

def embed_texts(texts, metadatas):
    for i, chunk in enumerate(tqdm(texts, desc="Embedding chunks")):
        emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        ).data[0].embedding
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[emb],
            metadatas=[metadatas[i]],
            documents=[chunk]
        )
