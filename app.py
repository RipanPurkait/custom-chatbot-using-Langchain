from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define FastAPI app
app = FastAPI()

# Global variable for vector store
vector_store = None

# Request model for query input
class QueryRequest(BaseModel):
    query: str
    k: int = 5  # Default to top 5 answers

# Function to extract and process data from a URL
def extract_data_from_url(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split documents into chunks (to handle token limits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create vector store using Hugging Face Embeddings
def create_vector_store(texts):
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # Create and return FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Function to retrieve answers based on a query
def retrieve_answer(query: str, vector_store, k: int = 5):
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    docs = vector_store.similarity_search_by_vector(query_embedding, k=k)

    # Extract and return the top documents
    top_docs = [doc.page_content for doc in docs]
    return top_docs

# Endpoint to load data from URL and create vector store
@app.post("/load-url/")
def load_url(url: str):
    global vector_store
    try:
        texts = extract_data_from_url(url)
        vector_store = create_vector_store(texts)
        return {"message": "Vector store created successfully", "document_count": len(texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {e}")

# Endpoint to retrieve answers based on a query
@app.post("/retrieve-answer/")
def get_answer(query_request: QueryRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Vector store is not initialized. Please load data first.")

    try:
        answers = retrieve_answer(query_request.query, vector_store, query_request.k)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving answer: {e}")



