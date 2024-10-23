from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#Load data from URL
def extract_data_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split documents into chunks (to handle token limits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

url_schema = "https://brainlox.com/courses/category/technical"
texts = extract_data_from_url(url_schema)