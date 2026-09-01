import streamlit as st
import tempfile
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot (Groq + PDF)")

# Sidebar setup
with st.sidebar:
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to start.", icon="🔑")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=api_key)

# Cache embedding model to speed up re-runs
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Process PDF into FAISS Vector Store
@st.cache_resource(show_spinner="Processing PDF...")
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    return FAISS.from_documents(chunks, embedding_model)

vector_db = None
if uploaded_file:
    vector_db = process_pdf(uploaded_file.getvalue())
    st.sidebar.success("PDF indexed successfully!")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context if document is loaded
    context = ""
    if vector_db:
        retrieved_docs = vector_db.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Construct System Prompt
    if context:
        system_instruction = f"""Answer the question using ONLY the provided document context. If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}"""
    else:
        system_instruction = "You are a helpful general AI assistant."

    # Build messages array for API payload
    messages_payload = [{"role": "system", "content": system_instruction}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    # Stream response from Groq
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_payload,
                stream=True,
            )
            
            full_response = ""
            for chunk in completion:
                full_response += (chunk.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")
