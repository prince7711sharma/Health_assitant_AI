import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face API token not found. Please set HF_TOKEN in your .env file.")
    st.stop()

HUGGINGFACE_LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"


# --- Functions ---
def setup_llm(huggingface_repo_id, hf_token):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.4,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm_endpoint)


SYSTEM_PROMPT_TEMPLATE = """
You are a trusted digital health assistant. Use the context from health documents to answer the user's questions in a clear and compassionate way. 
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


# --- Streamlit UI ---
st.set_page_config(page_title="🩺 Health AI Assistant", page_icon="🩺", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Background */
    .main { background-color: #f8fbfd; }

    /* Decorative header bar */
    .header-bar {
        background: linear-gradient(to right, #007b83, #00b8a9);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in-out;
    }
    .header-title {
        font-size: 36px;
        font-weight: 900;
        color: white;
        letter-spacing: 1px;
    }
    .header-subtitle {
        font-size: 16px;
        color: #e9fdfd;
        margin-top: 5px;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        background-color: #e1f0f6; 
        padding: 12px; 
        border-radius: 15px; 
        margin-bottom: 10px; 
        max-width: 80%;
        margin-left: auto;
        color: #003344;
    }
    .chat-bubble-assistant {
        background-color: #f1f9f4; 
        padding: 12px; 
        border-radius: 15px; 
        margin-bottom: 10px; 
        max-width: 80%;
        margin-right: auto;
        color: #003314;
    }

    /* Sources styling */
    .source-title {
        font-weight: bold;
        color: #005c4b;
        font-size: 18px;
        margin-top: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 12px;
        color: #777;
        margin-top: 20px;
    }

    /* Animation */
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# --- Decorative Heading ---
st.markdown("""
<div class="header-bar">
    <div class="header-title">🩺 Health AI Assistant</div>
    <div class="header-subtitle">Your trusted companion for health-related information</div>
</div>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "qa_chain" not in st.session_state:
    try:
        with st.spinner("🔄 Loading health documents and AI model..."):
            llm_model = setup_llm(HUGGINGFACE_LLM_REPO_ID, HF_TOKEN)

            # ✅ Set token to environment for embeddings
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

            # ✅ Initialize embeddings (without passing token)
            embedding_model = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL
            )

            if not os.path.exists(DB_FAISS_PATH):
                st.error(f"⚠️ FAISS index not found at {DB_FAISS_PATH}. Please build it first.")
                st.stop()

            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={'k': 3})

            custom_prompt = set_custom_prompt()
            document_combiner_chain = create_stuff_documents_chain(llm_model, custom_prompt)
            qa_chain = create_retrieval_chain(retriever, document_combiner_chain)

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
        st.success("✅ Assistant is ready to chat!")
    except Exception as e:
        st.error(f"Error initializing assistant: {e}")
        st.stop()

# --- Chat input ---
user_query = st.chat_input("💬 Type your health question here...")

if user_query:
    # Save user message
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("💭 Thinking..."):
        response = st.session_state.qa_chain.invoke({'input': user_query})
        answer = response['answer']

    # Save assistant response
    st.session_state.chat_history.append(("assistant", answer))

# --- Display chat history ---
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">🧑‍💻 <b>You:</b> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-assistant">🤖 <b>Assistant:</b> {message}</div>', unsafe_allow_html=True)

# --- Show sources ---
if user_query:
    st.markdown('<p class="source-title">📚 Sources from Health Documents:</p>', unsafe_allow_html=True)
    for i, doc in enumerate(response["context"], 1):
        with st.expander(f"🔹 Source {i}"):
            st.write(doc.page_content)
            if doc.metadata:
                st.json(doc.metadata)

# --- Footer ---
st.markdown(
    '<p class="footer">⚠️ This chatbot is for educational purposes only and is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True)
