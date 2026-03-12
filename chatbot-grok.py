import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ Missing API keys. Please set OPENAI_API_KEY and GROQ_API_KEY in your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with Your Doctor",
    page_icon="💬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #c3d9cb; }
    .stApp {
        background-color: #c3d9cb;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .main-header {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px;
    }
    .stTextInput, .stFileUploader, .stButton > button { border-radius: 8px; }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton > button:hover { background-color: #a3a5cf; }
    .answer-box {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        border-radius: 10px;
        padding: 18px 22px;
        font-size: 1rem;
        line-height: 1.7;
        color: #1b5e20;
    }
    .badge-ready   { background:#d4edda; color:#155724; padding:6px 14px; border-radius:20px; font-size:.85rem; font-weight:600; }
    .badge-missing { background:#fff3cd; color:#856404; padding:6px 14px; border-radius:20px; font-size:.85rem; font-weight:600; }
    .card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>💬 Chat with Your Doctor</div>", unsafe_allow_html=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
    )

llm = get_llm()

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Answer the question below using ONLY the provided context.
If the answer is not in the context, say: "I couldn't find relevant information in the provided documents."

<context>
{context}
</context>

Question: {question}

Provide a clear, accurate, and helpful answer.
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever):
    """Pure LCEL chain — no langchain.chains dependency."""
    return (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    uploaded_pdfs = st.file_uploader(
        "📄 Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more medical PDF documents.",
    )

    chunk_size    = st.slider("Chunk size",    500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0,   500,  200,  step=50)
    max_docs      = st.slider("Max documents", 10,  200,  50,   step=10)

    st.markdown("---")
    st.markdown("### 📊 Status")

    if "vectors" in st.session_state:
        st.markdown('<span class="badge-ready">✅ Vector DB ready</span>', unsafe_allow_html=True)
        st.caption(f"Chunks loaded: {len(st.session_state.get('final_documents', []))}")
    else:
        st.markdown('<span class="badge-missing">⚠️ Not embedded yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** LLaMA 3.3 70B  \n**Embeddings:** OpenAI  \n**Vector DB:** FAISS")

# ── Embedding function ────────────────────────────────────────────────────────
def build_vector_store(uploaded_files, chunk_size, chunk_overlap, max_docs):
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        return False

    tmp_dir = "_uploaded_pdfs"
    os.makedirs(tmp_dir, exist_ok=True)

    for f in os.listdir(tmp_dir):
        try:
            os.remove(os.path.join(tmp_dir, f))
        except OSError:
            pass

    for up in uploaded_files:
        with open(os.path.join(tmp_dir, up.name), "wb") as out_f:
            out_f.write(up.read())

    with st.spinner("📄 Loading PDF documents…"):
        docs = PyPDFDirectoryLoader(tmp_dir).load()

    with st.spinner("✂️ Splitting into chunks…"):
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).split_documents(docs[:max_docs])

    with st.spinner("🔢 Generating embeddings…"):
        embeddings = OpenAIEmbeddings()
        vectors    = FAISS.from_documents(chunks, embeddings)

    st.session_state.embeddings      = embeddings
    st.session_state.vectors         = vectors
    st.session_state.final_documents = chunks
    return True

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the symptoms of diabetes?",
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_btn = st.button("🔍 Get Answer", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("last_response", None)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if ask_btn and question:
        if "vectors" not in st.session_state:
            st.warning("⚠️ Please build the Vector DB first using the button on the right.")
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    retriever = st.session_state.vectors.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    chain = build_chain(retriever)

                    t0      = time.perf_counter()
                    answer  = chain.invoke(question)
                    elapsed = time.perf_counter() - t0

                    # Also fetch the source docs for display
                    source_docs = retriever.invoke(question)

                    st.session_state.last_answer  = answer
                    st.session_state.last_elapsed = elapsed
                    st.session_state.last_docs    = source_docs

                except Exception as e:
                    st.error(f"❌ Error during retrieval: {e}")

    if "last_answer" in st.session_state:
        st.markdown(f"**Answer** *(responded in {st.session_state.last_elapsed:.2f}s)*")
        st.markdown(
            f'<div class="answer-box">{st.session_state.last_answer}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📚 Source Chunks Used"):
            for i, doc in enumerate(st.session_state.last_docs, 1):
                src = doc.metadata.get("source", "Unknown")
                pg  = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** — `{os.path.basename(src)}` · page {pg}")
                st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
                st.markdown("---")

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗄️ Vector Database")
    st.caption("Upload and embed your PDF documents before asking questions.")

    if st.button("⚡ Build / Rebuild Vector DB", use_container_width=True):
        success = build_vector_store(uploaded_pdfs, chunk_size, chunk_overlap, max_docs)
        if success:
            st.success(f"✅ Vector DB ready! {len(st.session_state.final_documents)} chunks indexed.")
            st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use")
    st.markdown("""
1. Upload one or more PDF files using the uploader in the sidebar
2. Click **Build / Rebuild Vector DB** to index them
3. Type your medical question and click **Get Answer**
4. Expand *Source Chunks Used* to see which document passages were referenced
""")
    st.markdown("</div>", unsafe_allow_html=True)
