
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "iphone_user_guide")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def ask_question(question, chat_history=None):
    chat_history = chat_history or []

    history_text = "\n".join(
        f"{role}: {message}"
        for role, message in chat_history[-6:]
    )

    # Rewrite follow-up questions using conversation context.
    rewrite_prompt = f"""
Rewrite the latest user question as a standalone search query.

Use conversation history only to resolve references
such as "it", "this", or "where is it saved?".

Do not answer the question.
Do not add facts that are not in the conversation.
Return only the rewritten search query.

Conversation history:
{history_text}

Latest question:
{question}

Standalone search query:
"""

    search_query = llm.invoke(rewrite_prompt).content.strip()

    if not search_query:
        search_query = question

    # Retrieve document chunks using the standalone query.
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    docs = retriever.invoke(search_query)

    if not docs:
        return (
            "I couldn't find this information in the iPhone User Guide.",
            []
        )

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'Unknown')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    # Keep only page numbers that exist in retrieved documents.
    retrieved_pages = sorted({
        int(doc.metadata["page"])
        for doc in docs
        if doc.metadata.get("page") is not None
    })

    answer_prompt = f"""
You are an assistant for the iPhone User Guide.

Answer the user's question using ONLY the retrieved document context.

Strict rules:
1. Never use outside knowledge.
2. Never invent or guess information.
3. If the context does not clearly contain the answer,
   say exactly:
   "I couldn't find this information in the iPhone User Guide."
4. Conversation history may clarify the question,
   but it is NOT evidence.
5. Every factual statement must be supported by the retrieved context.
6. Do not write page citations in the answer.
7. Always answer in English only.
8. Be concise and direct.

Conversation history:
{history_text}

Standalone search query:
{search_query}

Retrieved document context:
{context}

Latest user question:
{question}

Answer:
"""

    answer = llm.invoke(answer_prompt).content.strip()

    if not answer:
        return (
            "I couldn't find this information in the iPhone User Guide.",
            []
        )

    # Ask the model to identify supporting pages only.
    verification_prompt = f"""
Identify which retrieved pages directly support the answer.

Use ONLY the page numbers and text in the retrieved context.
Do not select a page unless its text directly supports the answer.
Return page numbers only, separated by commas.
If no page directly supports the answer, return NONE.

Retrieved context:
{context}

Answer:
{answer}

Page numbers:
"""

    verification = llm.invoke(verification_prompt).content.strip()

    if "NONE" in verification.upper():
        return (
            "I couldn't find this information in the iPhone User Guide.",
            []
        )

    mentioned_pages = {
        int(page)
        for page in re.findall(r"\d+", verification)
    }

    sources = sorted(
        set(retrieved_pages).intersection(mentioned_pages)
    )

    if not sources:
        return (
            "I couldn't find this information in the iPhone User Guide.",
            []
        )

    return answer, sources
