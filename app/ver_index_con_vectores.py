from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Cargar embeddings desde Ollama
embedding_model = OllamaEmbeddings(model="llama3")

# Cargar el índice FAISS y el archivo .pkl con autorización
db = FAISS.load_local(r"C:\Users\harly.munoz\OneDrive - Perficient, Inc\rag_finanzas_ollama\faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Acceder a los documentos y sus vectores
documents = db.docstore._dict
index = db.index
vectors = index.reconstruct_n(0, index.ntotal)

# Imprimir texto y vector
for i, (doc_id, doc) in enumerate(documents.items()):
    print(f"\n--- Documento {i+1} ---")
    print("Texto:")
    print(doc.page_content)
    print("\nVector:")
    print(vectors[i])
    print("-" * 80)
