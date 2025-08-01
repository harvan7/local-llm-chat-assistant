# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requerimientos al contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia los directorios de la aplicación y los datos
COPY ./app /app/app
COPY ./data /app/data

# Crea el almacén de vectores. 
# Esto asegura que el índice FAISS esté disponible cuando se inicie el contenedor.
RUN python -c "from app.rag_engine import embed_and_store; embed_and_store()"

# Expone el puerto 8000 para que la API sea accesible
EXPOSE 8000

# Comando para iniciar la aplicación usando uvicorn
# El host 0.0.0.0 hace que sea accesible desde fuera del contenedor
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]