import os
import streamlit as st

# --- Load secrets from .env or Streamlit Cloud ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("⚠️ Hugging Face API token not found. Please set it in `.env` or Streamlit Secrets.")
    st.stop()

# --- LangChain / HuggingFace imports ---
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# --- Configuration ---
HUGGINGFACE_LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"


def setup_llm(repo_id, token):
    """Setup HuggingFace LLM endpoint."""
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.4,
        huggingfacehub_api_token=token,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm_endpoint)


SYSTEM_PROMPT_TEMPLATE = """
🩺 You are a trusted digital health assistant. Use the context from health documents to answer the user's questions in a clear and compassionate way. 
If you are unsure about something, say "I'm not sure about that" rather than guessing.

Context: {context}
"""


def set_custom_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", "{input}")
        ]
    )


# --- Streamlit UI Settings ---
st.set_page_config(page_title="🩺 Health AI Assistant", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8fbfd; }
    .stChatMessage { border-radius: 15px; padding: 12px; }
    .user { background-color: #e1f0f6; }
    .assistant { background-color: #f1f9f4; }
    .title { color: #007b83; font-size: 36px; font-weight: 800; text-align:center; }
    .subtitle { text-align:center; font-size:16px; color:#555; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🩺 Health AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your trusted AI-powered health support based on verified documents.</p>', unsafe_allow_html=True)

# --- Initialize LLM and Vector DB ---
if "qa_chain" not in st.session_state:
    try:
        with st.spinner("🔄 Initializing health assistant..."):
            llm_model = setup_llm(HUGGINGFACE_LLM_REPO_ID, HF_TOKEN)

            # FIX: Pass HF token to embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL,
                huggingfacehub_api_token=HF_TOKEN,
                cache_folder="models"
            )

            # Load FAISS
            if not os.path.exists(DB_FAISS_PATH):
                st.error(f"⚠️ FAISS index not found at `{DB_FAISS_PATH}`. Please build it first.")
                st.stop()

            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={'k': 3})

            custom_prompt = set_custom_prompt()
            document_combiner_chain = create_stuff_documents_chain(llm_model, custom_prompt)
            qa_chain = create_retrieval_chain(retriever, document_combiner_chain)

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
        st.success("✅ Assistant is ready!")
    except Exception as e:
        st.error(f"❌ Error initializing assistant: {e}")
        st.stop()


# --- Chat UI ---
user_query = st.chat_input("💬 Ask a health-related question...")

if user_query:
    # Display user message
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("🤔 Thinking..."):
        try:
            response = st.session_state.qa_chain.invoke({'input': user_query})
            answer = response['answer']
            st.session_state.chat_history.append(("assistant", answer))
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"❌ Error: {e}"))

# Display chat history with styling
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 <b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant'>🤖 <b>Assistant:</b> {message}</div>", unsafe_allow_html=True)

# Show sources if available
if user_query and "context" in response:
    st.markdown("### 📚 Sources from health documents:")
    for i, doc in enumerate(response["context"], 1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)
            if doc.metadata:
                st.json(doc.metadata)

