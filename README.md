# RAG Finanzas Ollama

Este proyecto es un servicio de API de **Preguntas y Respuestas (Q&A)** que utiliza un modelo de **Generación Aumentada por Recuperación (RAG)** para responder preguntas sobre finanzas personales. La API está construida con **FastAPI** y el motor de RAG utiliza **LangChain**, **Ollama** y **FAISS**.

## Características

-   **API de Q&A:** Expone un endpoint `/ask` para hacer preguntas sobre finanzas personales.
-   **Motor RAG:** Utiliza un modelo de lenguaje (a través de Ollama) y una base de datos de vectores (FAISS) para proporcionar respuestas precisas basadas en un conjunto de documentos.
-   **Seguridad:** Protege el endpoint de la API con una clave de API.
-   **Contenerizado con Docker:** Se proporciona un `Dockerfile` para construir y ejecutar fácilmente el servicio.

## Cómo empezar (con Docker)

### Prerrequisitos

-   [Docker](https://www.docker.com/get-started) instalado en tu máquina.

### Construir y ejecutar la imagen de Docker

1.  **Clona este repositorio:**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd rag_finanzas_ollama
    ```

2.  **Crea un archivo `.env`:**

    En la raíz del proyecto, crea un archivo llamado `.env` y añade tu clave de API:

    ```
    API_KEY=tu_super_secreta_api_key
    ```

3.  **Construye la imagen de Docker:**

    ```bash
    docker build -t rag-finanzas-api .
    ```

4.  **Ejecuta el contenedor de Docker:**

    ```bash
    docker run -d -p 8000:8000 --env-file .env --name rag-finanzas-container rag-finanzas-api
    ```

    Esto iniciará el contenedor en segundo plano y mapeará el puerto 8000 del contenedor al puerto 8000 de tu máquina.

### Usar la API

Una vez que el contenedor esté en funcionamiento, puedes enviar peticiones `POST` al endpoint `/ask`.

-   **URL:** `http://localhost:8000/ask`
-   **Método:** `POST`
-   **Headers:**
    -   `Authorization`: `Bearer tu_super_secreta_api_key`
    -   `Content-Type`: `application/json`
-   **Body (raw JSON):**

    ```json
    {
        "query": "¿Cómo puedo empezar a invertir?"
    }
    ```

#### Ejemplo con `curl`

```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer tu_super_secreta_api_key" \
-d '{
    "query": "¿Cuáles son algunos consejos para ahorrar dinero?"
}'
```

---

# 🧠 RAG Finanzas Personales con Ollama + LLaMA3:8B (Desarrollo Local)

Sistema RAG (Retrieval-Augmented Generation) usando documentos `.txt` y el modelo local `llama3:8b` de **Ollama**, especializado en responder preguntas sobre **finanzas personales**.

## 📦 Requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado y ejecutando `llama3:8b`
- Modelo cargado con:

```bash
ollama run llama3:8b
```