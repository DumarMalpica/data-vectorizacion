import os
from src.parse_docs import parse_pdf
from src.embed_store import embed_texts
from src.query_rag import query_rag

def main():
    docs_path = "docs"
    texts, metas = [], []

    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            print(f"Procesando {file}...")
            text = parse_pdf(os.path.join(docs_path, file))
            texts.append(text)
            metas.append({"source": file})

    embed_texts(texts, metas)
    print("\n✅ Vectorización completada.")

    while True:
        q = input("\n❓ Pregunta ('exit' para salir): ")
        if q.lower() == "exit":
            break
        query_rag(q)

if __name__ == "__main__":
    main()
