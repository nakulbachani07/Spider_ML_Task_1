# PDF Question Answering System using RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based PDF Question Answering System.

The chatbot allows users to ask questions related to a collection of research papers and receive context-aware answers generated using a Large Language Model (LLM). Relevant document chunks are retrieved from a vector database and provided to the LLM as context before answer generation.

A Streamlit-based web interface is used for interaction.

---

## Features

* PDF document ingestion
* Text chunking and preprocessing
* Semantic search using vector embeddings
* ChromaDB vector database
* Groq LLM integration
* Streamlit chatbot interface
* Source attribution with similarity scores
* Retrieval-Augmented Generation (RAG) pipeline

---

## Tech Stack

### Embedding Model

* Sentence Transformers
* Model: `all-MiniLM-L6-v2`

### Vector Database

* ChromaDB

### Large Language Model

* Groq
* Model: `llama-3.3-70b-versatile`

### Frameworks and Libraries

* Python
* LangChain
* Streamlit
* Sentence Transformers
* ChromaDB
* NumPy
* PyMuPDF
* python-dotenv

---

## Workflow

### 1. Document Ingestion

Research papers are loaded from PDF files.

### 2. Text Chunking

Documents are split into smaller chunks for efficient retrieval.

### 3. Embedding Generation

Each chunk is converted into dense vector embeddings using Sentence Transformers.

### 4. Vector Storage

Embeddings and metadata are stored in ChromaDB.

### 5. Retrieval

For every user query:

* Query embeddings are generated.
* Top-k relevant chunks are retrieved from ChromaDB.
* Similarity scores are calculated.

### 6. Answer Generation

Retrieved context is passed to the Groq LLM, which generates an answer grounded only in the retrieved documents.

---


---



---

## Example Questions

* What is Retrieval-Augmented Generation?
* How does self-attention differ from recurrence?
* What is the difference between BERT and GPT?
* Explain Multi-Head Attention.
* What are the advantages of LoRA?
* Explain Sentence-BERT.

---

## Sample Output

```text
Answer:
Self-attention differs from recurrence because it allows parallel computation across sequence positions, while recurrent networks process tokens sequentially.

Sources:

1. Attention_is_all_you_need.pdf (Page 5)
   Similarity: 0.82

2. Attention_is_all_you_need.pdf (Page 4)
   Similarity: 0.79
```

---




