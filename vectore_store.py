from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from web_scrape import *

# Create embeddings and store in a vector store using Hugging Face
def create_vector_store(texts):
    # Use Hugging Face Embeddings 
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # Create vector store (FAISS)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Retrieve answers based on a query
def retrieve_answer(query, vector_store, k=5):
    # Embed the query using the same Hugging Face model
    hf_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # Create an embedding for the query
    query_embedding = embeddings.embed_query(query)

    # Search the vector store for similar documents (returns top k matches)
    # The similarity_search_by_vector method by default returns only documents.
    # To get the scores, use similarity_search_with_score_by_vector instead.
    docs = vector_store.similarity_search_by_vector(query_embedding, k=k)

    # Extract the content of the top documents
    top_docs = [doc.page_content for doc in docs]

    return top_docs


# Create vector store
vector_store = create_vector_store(texts)

# Retrieve answers for a query
query = "Python"
answers = retrieve_answer(query, vector_store, k=3)

for i, answer in enumerate(answers, 1):
    print(f"Answer {i}: {answer}")