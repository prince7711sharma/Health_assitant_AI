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
    st.error("⚠️ Hugging Face API token not found. Please set HF_TOKEN in your .env file or Streamlit secrets.")
    st.stop()

HUGGINGFACE_LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"


# --- Functions ---
def setup_llm(huggingface_repo_id, hf_token):
    """
    Sets up the Hugging Face LLM endpoint.
    """
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.4,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm_endpoint)


SYSTEM_PROMPT_TEMPLATE = """
You are MediBot AI, a trusted digital health assistant. Always introduce yourself as MediBot AI at the beginning of your responses. Use the context from health documents to answer the user's questions in a clear and compassionate way. 
If you are unsure about something, say "I'm not sure about that" rather than guessing.

Context: {context}
"""


def set_custom_prompt():
    """
    Defines the custom chat prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", "{input}")
        ]
    )


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="🤖 Medibot AI Health Assistant", page_icon="🩺", layout="wide")

# --- Custom CSS for Styling ---
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
st.markdown("""
<div class="header-bar">
    <div class="header-title">🩺 Health AI Assistant</div>
    <div class="header-subtitle">Your trusted companion for health-related information</div>
</div>
""", unsafe_allow_html=True)
# --- Initialize session state for QA chain and chat history ---
if "qa_chain" not in st.session_state:
    try:
        with st.spinner("🔄 Loading health documents and AI model..."):
            llm_model = setup_llm(HUGGINGFACE_LLM_REPO_ID, HF_TOKEN)

            # Set token to environment for HuggingFaceEmbeddings to pick up
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

            # Initialize embeddings (token is picked from environment)
            embedding_model = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL
            )

            if not os.path.exists(DB_FAISS_PATH):
                st.error(f"⚠️ FAISS index not found at {DB_FAISS_PATH}. Please build it first and ensure it's committed to your repository.")
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

# Process user query and generate response only if a query is submitted
if user_query:
    # Save user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("💭 Thinking..."):
        # Invoke the QA chain to get the response
        response = st.session_state.qa_chain.invoke({'input': user_query})
        answer = response['answer']

    # Save assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": response.get("context", [])})

# --- Display chat history ---
# This loop runs on every rerun to display all messages
for message_obj in st.session_state.chat_history:
    role = message_obj["role"]
    content = message_obj["content"]

    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">🧑‍💻 <b>You:</b> {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-assistant">🤖 <b>Assistant:</b> {content}</div>', unsafe_allow_html=True)
        # Display sources if available for the assistant's response
        if "context" in message_obj and message_obj["context"]:
            st.markdown('<p class="source-title">📚 Sources from Health Documents:</p>', unsafe_allow_html=True)
            for i, doc in enumerate(message_obj["context"], 1):
                with st.expander(f"🔹 Source {i}"):
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.json(doc.metadata)

# --- Footer ---
st.markdown(
    '<p class="footer">⚠️ This chatbot is for educational purposes only and is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True)









