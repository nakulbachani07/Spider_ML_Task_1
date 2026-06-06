import os
import chromadb
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

load_dotenv()


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "./data/vector_store"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings"}
        )


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })

        return retrieved_docs


groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024
)

embedding_manager = EmbeddingManager()

vector_store = VectorStore(
    collection_name="pdf_documents",
    persist_directory="./data/vector_store"
)

rag_retriever = RAGRetriever(vector_store, embedding_manager)


def rag_simple(query, retriever, llm, top_k=5):
    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [doc["content"] for doc in results]
    ) if results else ""

    if not context:
        return "No relevant context found to answer the question."

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    sources = []

    for doc in results:
        metadata = doc["metadata"]

        filename = (
            metadata.get("source_file")
            or metadata.get("sorce_file")
            or "Unknown File"
        )

        page = metadata.get("page", "Unknown")
        similarity = doc.get("similarity_score", 0.0)

        sources.append(
            f"- {filename} | Page: {page} | Similarity: {similarity:.3f}"
        )

    final_response = f"""
{answer}

---

Sources:
{chr(10).join(sources)}
"""

    return final_response