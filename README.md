# iPhone User Guide RAG Chatbot

## Project Overview

An AI-powered chatbot that answers questions using the iPhone User Guide PDF.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the document and generate answers with source page references.

## Technologies

- Python 3.12
- LangChain
- LangChain Groq
- HuggingFace Sentence Transformers
- Qdrant Cloud
- Chainlit
- Docker

## Features

- Answers questions based only on the iPhone User Guide.
- Retrieves relevant document chunks from Qdrant Cloud.
- Includes source page references.
- Supports multi-turn conversations within a session.
- Responds when information is not found in the document.
- Provides answers in English.

## Environment Variables

Create a `.env` file using `.env.example` and provide:

- GROQ_API_KEY
- QDRANT_URL
- QDRANT_API_KEY
- COLLECTION_NAME
- PORT

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the Qdrant Cloud collection is already populated and the environment variables are configured.

Start the chatbot:

```bash
chainlit run app.py --host 0.0.0.0 --port 8000
```

Open:

http://localhost:8000

## Run with Docker

Build the image:

```bash
docker build -t chatbot:1.0 .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env chatbot:1.0
```

Open:

http://localhost:8000

## Important Notes

- The Qdrant Cloud collection has already been populated with the document data.
- The application connects to the existing collection at runtime.
- The ingestion script is not executed when the application starts.
- The evaluator does not need to run the ingestion script.
- The `.env` file must contain valid API credentials and must never be committed to GitHub.

## Document

Source document: iPhone User Guide PDF.
