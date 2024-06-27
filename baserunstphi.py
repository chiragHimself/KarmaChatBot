import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import ollama

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
    return Chroma.from_documents(_documents, _embeddings)

def ollama_llm(question, context):#this is the prompt temp for llama3
    
    formatted_prompt = f""" You are the official customer support chatbot for Karma Points . Any response generated by you must be from provided context only.
Answer precisely within 100 words , your response shall not contain any text except basic answer like code or extended info and never provide any Note: to user at the end of response.
Various available job positions are provided in context only , you may answer it if user asks from context.
You may start with a small greeting. Never say based on provided context and never repeat user questions.
Only if user asks for physical address you can tell it is located at F9, Tower 15, Type 3, East Kidwai Nagar, New Delhi - 110023.
Only talk about KP redeeming process and points when asked and not other wise dont talk about kp redemption. 
you may specify company website only once in your answer.
 .You might try to formulate a story type response based on provided context only
 .Answer to what is asked in a efficient and accurate manner.You will only answer what is asked not a word more even its just one line. Question: {question}\n\nContext: {context}"""
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
documents = load_documents("RagTextFiles")
split_docs = split_documents(documents)
embeddings = create_embeddings("phi3")
db = create_db(split_docs, embeddings)

retriever = db.as_retriever()

# Text input for user query
query = st.text_input("How may we help?(AI response - waiting time 5-10s)")

# Button to submit the query
if st.button("Submit"):
    if query:
        response = rag_chain(query, retriever)
        st.write(response)



