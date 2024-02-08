from flask import Flask, jsonify, request
import os
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Initialize Flask app
app = Flask(__name__)

# Load documents from CSV file
def load_docs(directory):
    loader = CSVLoader(directory)
    documents = loader.load()
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize Chroma vector store
def initialize_chroma(directory):
    documents = load_docs(directory)
    docs = split_docs(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    return db

# Load OpenAI model
def load_openai_model():
    os.environ["OPENAI_API_KEY"] = ""
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    return chain

# Initialize Chroma vector store and OpenAI model
db = initialize_chroma('./train12.csv')
chain = load_openai_model()

# Define route for recommendation API
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from JSON request
    data = request.get_json()
    user_input = data.get('query')
    query = "What are all news title and description match to " + user_input
    
    # Perform similarity search
    matching_docs = db.similarity_search(query, k=6)

    # Prepare response JSON
    response = []
    for doc in matching_docs[1:]:
        response.append({'Recommended Articles': doc.page_content})

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
