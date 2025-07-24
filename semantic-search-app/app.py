# semantic-search/app.py

import asyncio
import sys
if sys.platform.startswith('win') and asyncio.get_event_loop_policy().__class__.__name__ != 'WindowsSelectorEventLoopPolicy':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Custom UI Styling
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1a2a6c, #16213e 80%);
        color: #fff;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        color: #f1f1f1;
    }

    h1, h2, h3 {
        color: #82B1FF;
        font-weight: 700;
    }

    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        color: #000;
        border: 2px solid #82B1FF;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stFileUploader > div {
        background-color: #f1f1f1;
        border: 2px dashed #2196F3;
        border-radius: 10px;
        padding: 1rem;
    }

    .stButton > button {
        background-color: #1976D2;
        color: #fff;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0D47A1;
        transform: scale(1.03);
    }

    .answer-container {
        background-color: rgba(255, 255, 255, 0.95);
        color: #000;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
    }

    .context-container {
        background-color: #f0f8ff;
        color: #333;
        border-left: 5px solid #90CAF9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        font-style: italic;
    }

    footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #90CAF9;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class='main-card'>
    <h1 style='text-align:center;'>🧠 MindSeek</h1>
    <p style='text-align:center; font-size:1.1rem;'>Upload PDFs and ask smart, context-aware questions with Gemini AI.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📘 How to use")
    st.markdown("1. Upload a PDF\n2. Ask a question\n3. View context-based AI answer")
    st.header("✨ Features")
    st.markdown("- Gemini AI Semantic Search\n- PDF Text Embedding\n- Confidence Score")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    st.markdown(f"<div class='answer-container'><b>📄 File:</b> {uploaded_file.name}<br><b>Size:</b> {uploaded_file.size / 1024 / 1024:.2f} MB</div>", unsafe_allow_html=True)

    with st.spinner("🔄 Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = Chroma.from_documents(chunks, embeddings)

            st.success(f"✅ Processed {len(chunks)} chunks")

            query = st.text_input("🤔 Ask something about the document:")

            if query:
                matches = db.similarity_search_with_score(query, k=3)
                if matches:
                    context = "\n\n".join([match[0].page_content for match in matches])
                    confidence = 1 - matches[0][1]

                    model = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = f"""
You are a document analysis assistant. Use the context provided below to answer the user's question as accurately as possible.

Context:
{context}

Question:
"{query}"

Instructions:
- Focus on word-by-word meaning and precision.
- Do not make assumptions beyond the context.
- If the context is insufficient, say so.

Answer:
"""
                    response = model.generate_content(prompt)

                    st.markdown("### 🤖 AI Answer")
                    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                    st.markdown(f"*Confidence:* {confidence_color} {confidence*100:.0f}%")
                    st.markdown(f"<div class='answer-container'><h4>💬 Answer:</h4><p>{response.text}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='context-container'>{context[:500]}{'...' if len(context) > 500 else ''}</div>", unsafe_allow_html=True)
                else:
                    st.warning("⚠ No relevant content found. Try rephrasing your question.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            os.remove(tmp_path)
else:
    st.markdown("<div class='answer-container'><h4>👋 Welcome!</h4><p>Upload a PDF to start exploring it with AI.</p></div>", unsafe_allow_html=True)

st.markdown("<footer>🚀 Powered by Gemini AI • Built with Streamlit • Enhanced in BlueGlass UI</footer>", unsafe_allow_html=True)
