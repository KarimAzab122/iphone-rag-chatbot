import os
import pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

PDF_PATH = "data/iPhone User Guide 1.pdf"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "iphone_user_guide")

def load_pdf():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf = pymupdf.open(PDF_PATH)
    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(PDF_PATH),
                        "page": page_number
                    }
                )
            )

    pdf.close()
    return documents


def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Please set QDRANT_URL and QDRANT_API_KEY in .env")

    print("Loading PDF...")
    documents = load_pdf()
    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Uploading chunks to Qdrant...")
    QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    ).add_documents(chunks)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()