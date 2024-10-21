import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import re

load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("ChatBot on Documents Q&A")

# Initialize the LLM model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Gemma-7b-it",
    temperature=0.5,
    max_tokens=500,
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Load vectors from PDFs
def load_pdf_to_vectors(pdf_paths):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)
    
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, all_docs

# Function to load conversation history from a file
def load_conversation_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

# Function to save conversation history to a file
def save_conversation_history(file_path, conversation_history):
    with open(file_path, 'w') as file:
        json.dump(conversation_history, file)

# PDF file paths (Update with your paths)
pdf_paths = [
    r"C:\Users\sujata.shelke\OneDrive - Acuvate Software Private Limited\Python\Nucor_Model_YOLO\Chat-with-PDF-Chatbot-main\mydata\Final_FREQUENTLY_ASKED_QUESTIONS_-PATENT (3).pdf",
    r"C:\Users\sujata.shelke\OneDrive - Acuvate Software Private Limited\Python\Nucor_Model_YOLO\Chat-with-PDF-Chatbot-main\mydata1\FINAL_FAQs_June_2018 (1).pdf"
]

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors, st.session_state.docs = load_pdf_to_vectors(pdf_paths)

# Initialize conversation history
conversation_history_file = "conversation_history.json"
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_conversation_history(conversation_history_file)

# Input for user query
prompt1 = st.text_input("What do you want to ask from the documents?")

# Function for F1-score calculation
def compute_f1(prediction, ground_truth):
    prediction_tokens = re.findall(r'\b\w+\b', prediction.lower())
    ground_truth_tokens = re.findall(r'\b\w+\b', ground_truth.lower())

    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Ground truth for comparison (In practice, this could be an annotated dataset)
ground_truth_answer = "This is the correct answer."  # Replace with real ground truth

# Process user input
if prompt1:
    st.session_state.conversation_history.append({"role": "user", "content": prompt1})

    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        # Add the bot's response to the conversation history
        st.session_state.conversation_history.append({"role": "bot", "content": response['answer']})
        
        # Display answer
        st.write("Answer:")
        st.write(response['answer'])

        # Calculate and display F1 Score
        f1 = compute_f1(response['answer'], ground_truth_answer)
        st.markdown(f"<p style='font-size:16px'>F1-Score: {f1:.2f}</p>", unsafe_allow_html=True)

        # Save the conversation history to a file
        save_conversation_history(conversation_history_file, st.session_state.conversation_history)

        # Display the conversation history
        st.write("Conversation History:")
        for entry in st.session_state.conversation_history:
            st.write(f"{entry['role']}: {entry['content']}")
        
        if response.get("context"):
            related_docs = [doc.metadata.get("source", "Unknown") for doc in response["context"]]
            unique_related_docs = list(set(related_docs))
            st.write("Related Documents:")
            st.write(", ".join(unique_related_docs))
        
    else:
        st.error("Vector store not initialized.")
