Spider ML Task 1

This repository contains my submission for Spider ML Task 1.

Contents
1. Base Task - Fashion MNIST Classification

The base_task/ folder contains the complete implementation of the Fashion-MNIST classification task using PyTorch.

It includes:

Complete training and evaluation notebook
Saved model weights
submission.csv
Accuracy and loss plots inside the notebook


2. Applied ML Domain - RAG Chatbot

The applied_ml_domain/ folder contains a Retrieval-Augmented Generation chatbot built for answering questions from research papers related to NLP and Large Language Models.

The chatbot supports:

PDF ingestion

Text chunking

SentenceTransformer embeddings

ChromaDB vector storage

Semantic retrieval

Groq LLM-based answer generation

Streamlit interface

Source paper and similarity score display

Tech Stack

Python

PyTorch

Streamlit

ChromaDB

SentenceTransformers

LangChain

Groq API

Pandas, NumPy, Matplotlib

How to Run Applied ML Chatbot - cd applied_ml_domain
                                pip install -r requirements.txt
                                streamlit run app.py


Create a .env file using .env.example:

GROQ_API_KEY=your_groq_api_key_here

Repository Structure

Spider_ML_Task_1/
│
├── README.md
├── base_task/
├── applied_ml_domain/
├── .env.example
├── .gitignore
└── requirements / config files
