from app.rag_engine import embed_and_store

if __name__ == "__main__":
    print("Creando y guardando el almacén de vectores...")
    embed_and_store()
    print("¡Almacén de vectores creado exitosamente!")
