from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

# Initialize FastAPI app
app = FastAPI()

# Create a class to handle the input structure for texts
class TextItem(BaseModel):
    page_content: str

class QueryRequest(BaseModel):
    query: str
    k: int = 5  # Optional, default to 5 results

# Initialize vector store globally (in memory)
vector_store = None

# Step 1: API to create vector store from documents
@app.post("/create-vector-store")
def create_vector_store_api(texts: List[TextItem]):
    global vector_store
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # Convert list of TextItem to the required format (list of dicts with "page_content")
    documents = [{"page_content": text.page_content} for text in texts]
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return {"message": "Vector store created successfully!"}

# Step 2: API to retrieve answers based on query
@app.post("/retrieve-answer")
def retrieve_answer_api(query_request: QueryRequest):
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Vector store not initialized. Please create it first.")

    # Embed the query
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)
    query_embedding = embeddings.embed_query(query_request.query)

    # Retrieve top k results
    docs = vector_store.similarity_search_by_vector(query_embedding, k=query_request.k)
    top_docs = [doc.page_content for doc in docs]

    return {"query": query_request.query, "results": top_docs}

# Step 3: API health check
@app.get("/health")
def health_check():
    return {"status": "API is running"}

