🩺 MediBot AI Health Assistant

🩺 MediBot AI Health Assistant
MediBot AI is a compassionate and knowledgeable digital health assistant built with Streamlit and LangChain. It leverages a powerful Large Language Model (LLM) from Hugging Face and a FAISS vector store to provide clear, concise, and helpful information based on a corpus of health documents.

✨ Features
Intelligent Q&A: Answers user questions based on provided health document context.

Contextual Responses: Utilizes a retrieval-augmented generation (RAG) approach to fetch relevant information.

Source Citation: Displays the specific health document snippets used to generate answers, promoting transparency.

Personalized Interaction: Addresses the user by a predefined name (e.g., "Prince") and introduces itself as "MediBot AI."

Responsive UI: Designed with a clean, modern Streamlit interface and custom CSS for a pleasant user experience.

Secure Token Handling: Uses environment variables/Streamlit secrets for API token management.

🚀 Technologies Used
Streamlit: For building the interactive web application.

LangChain: Framework for developing applications powered by language models.

Hugging Face Hub: Provides the LLM (Mistral-7B-Instruct-v0.3) and embedding model (sentence-transformers/all-MiniLM-L6-v2).

FAISS: For efficient similarity search and storage of vector embeddings.

Python-dotenv: For local management of environment variables.

⚙️ Setup Instructions
Follow these steps to set up and run the MediBot AI Health Assistant locally or deploy it.

Prerequisites
Python 3.9+
Git

1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>
2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Create a requirements.txt file in your project root with the following content:

streamlit
python-dotenv
langchain==0.2.5
langchain-community==0.2.5
langchain-core==0.2.14
langchain-huggingface==0.0.3
faiss-cpu
sentence-transformers

Then install them:

pip install -r requirements.txt

4. Obtain Hugging Face API Token
Go to Hugging Face.

Create a new User Access Token with read role.

Copy this token.

5. Set Up Environment Variables
Locally:
Create a file named .env in the root of your project directory (where app.py is located) and add your Hugging Face token:

HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN_HERE"

Important: Make sure your .gitignore file includes .env to prevent accidentally committing your token to GitHub.

On Streamlit Cloud (for deployment):
Do NOT commit your .env file. Instead, use Streamlit Cloud's secrets management:

Go to your app's dashboard on Streamlit Cloud.

Click the three dots (⋮) next to your app and select "Settings."

Go to the "Secrets" tab.

Add your token in TOML format:

HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN_HERE"

Save the secrets.

6. Build the FAISS Vector Store
The project relies on a FAISS index located at vectorstore/db_faiss. You need to build this index from your health documents. This typically involves:

Loading your health documents (e.g., PDFs, text files).

Splitting them into chunks.

Generating embeddings for these chunks using HuggingFaceEmbeddings.

Creating and saving a FAISS index.

Note: The current app.py assumes this index already exists. If you don't have it, you'll need a separate script to create it. A simplified example of how you might create db_faiss (assuming you have a docs folder with .txt files):

# create_vectorstore.py (example script, not part of app.py)
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set your Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_YOUR_HUGGING_FACE_TOKEN_HERE"

# Define paths
DOCS_PATH = "docs" # Folder containing your health documents
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_faiss_index():
    documents = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"): # Adjust for other file types
            file_path = os.path.join(DOCS_PATH, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    if not os.path.exists(DOCS_PATH):
        print(f"Please create a '{DOCS_PATH}' directory and add your health documents (e.g., .txt files) inside it.")
    else:
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        create_faiss_index()


Run this script: python create_vectorstore.py

Important: After creating vectorstore/db_faiss, commit the vectorstore directory to your Git repository. This is crucial for deployment on Streamlit Cloud, as it needs these files to be present.

▶️ How to Run Locally
Once all dependencies are installed and the FAISS index is built and present:

streamlit run app.py

This will open the application in your web browser.

☁️ Deployment on Streamlit Cloud
Push your entire project (including app.py, requirements.txt, and the vectorstore directory) to a GitHub repository.

Go to Streamlit Cloud and click "New app."

Connect your GitHub repository and select the branch and app.py file.

Ensure your HF_TOKEN is set up in the Streamlit Cloud secrets (as described in step 5 of Setup).

Click "Deploy!"

⚠️ Important Notes
FAISS Index: The vectorstore/db_faiss directory must be present and committed to your GitHub repository for the app to work on Streamlit Cloud.

API Token: Never hardcode your Hugging Face API token directly in app.py. Always use environment variables or Streamlit secrets.

Medical Advice Disclaimer: This chatbot is for educational purposes only and is not a substitute for professional medical advice.

📄 License
This project is open-source and available under the MIT License. (You can replace this with your preferred license).
