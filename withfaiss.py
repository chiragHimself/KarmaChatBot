import streamlit as st
import os
from dotenv import load_dotenv
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import ollama
#from langchain_groq import ChatGroq

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__88d55440990d4b7184cffbb3324a6209"

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def load_documents(directory):
    loader = DirectoryLoader(directory, loader_cls=TextLoader)
    text_documents = loader.load()
    return text_documents

@st.cache_resource
def split_documents(_documents, chunk_size=800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(_documents)

@st.cache_resource
def create_embeddings(model_name):
    return OllamaEmbeddings(model=model_name)

@st.cache_resource
def create_db(_documents, _embeddings):
    return FAISS.from_documents(_documents, _embeddings)

def ollama_llm(question, context):#this is the prompt temp for llama3
    
    formatted_prompt = f""" Your goal is to respond as a official chatbot for KarmaPoints. 
    All the answers generated must be from the provided context only.
    Answer in very short and precise way within 50 words but for major info like explaining redeeming process, answer might be longer,
    . the karma points also have a physical office located at F9, Tower 15, Type 3, East Kidwai Nagar, New Delhi - 110023 
    you may start with a very small greeting. Do not repeat based on provided context and never repeat users question.
    Try to be friendly , Refuse politely if you are not 100 percent certain about the info, don't hallucinate.
    Never specify the redemption process if not asked in question.
Tell them to visit the official website: https://thekarmapoints.com/ for further info after each response. Question: {question}\n\nContext: {context}"""
    response = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# RAG Setup
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Main Streamlit App
st.title("Karma Bot (Powered by Phi3)") 

# Load documents and set up the database only once
documents = load_documents("C:\\Users\\MSI GAMING\\OneDrive\\Desktop\\KarmaBot\\RagTextFiles")
split_docs = split_documents(documents)
embeddings = create_embeddings("llama3")
db = create_db(split_docs, embeddings)

retriever = db.as_retriever()

# Text input for user query
query = st.text_input("How may we help?(AI response - waiting time 15-20s)")

# Button to submit the query
if st.button("Submit"):
    if query:
        response = rag_chain(query, retriever)
        st.write(response)



