from openai import OpenAI
from chromadb import PersistentClient
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = PersistentClient(path="data")
collection = chroma_client.get_or_create_collection("docs")

def query_rag(question, top_k=3):
    q_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    ).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    context = "\n".join(results["documents"][0])

    prompt = f"""
Responde con precisión técnica basándote en el siguiente contexto:
{context}

Pregunta: {question}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\n🧠 Respuesta:")
    print(completion.choices[0].message.content)
