from flask import Flask, request, jsonify
from llama_index.core import Document, SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import requests
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

API_KEY = os.getenv('PINECONE_API_KEY')
API_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')
index_name = "docs-quickstart-index"
namespace = "user1"
pc = Pinecone(api_key=API_KEY)

def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks

def getEmbeddingModel():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def getChunkedDocs():
    documents = SimpleDirectoryReader('docs').load_data()
    chunked_documents = []
    for doc in documents:
        chunks = chunk_text(doc.text)
        for chunk in chunks:
            chunked_documents.append(Document(text=chunk))
    return chunked_documents

def getEmbeddings():
    embedding_model = getEmbeddingModel()
    chunked_documents = getChunkedDocs()
    embeddings = embedding_model.encode([doc.text for doc in chunked_documents], convert_to_tensor=True)
    return embeddings

def getIndex():
    index = pc.Index(index_name)
    return index

# Initialize Pinecone
def insertEmbeddingsIntoDB():
    embeddings = getEmbeddings()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.shape[1],
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 
    chunked_documents = getChunkedDocs()
    index = pc.Index(index_name)

    vectors = [
        {"id": str(i), "values": embedding.tolist(), "metadata": {"text": chunked_documents[i].text}}
        for i, embedding in enumerate(embeddings)
    ]

    index.upsert(vectors,namespace=namespace)

def callLLM(payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def retrieve_documents(query, index, embedding_model, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    result = index.query(vector=query_embedding.tolist(),namespace=namespace, top_k=top_k, include_metadata=True)
    retrieved_docs = [Document(text=item['metadata']['text']) for item in result['matches']]
    return retrieved_docs

def generate_response(query, retrieved_docs):
    context = " ".join([doc.text for doc in retrieved_docs])
    input_text = f"Context: {context}\n Answer the below query using the above context. Keep Answer to the point. No Irrelevant content. \nQuery: {query}"
    jsonq = {"inputs": input_text,
              "parameters":{
                 "max_new_tokens":100 
              }}
    data = callLLM(jsonq)
    return data[0]['generated_text']


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        # Save the file to 'articles' folder
        os.makedirs('docs', exist_ok=True)
        save_path = os.path.join('docs', file.filename)
        file.save(save_path)
        return jsonify({"message": "File processed and saved successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type, please upload a PDF file"}), 400

@app.route('/embeddings', methods=['POST'])
def create_embeddings():
    insertEmbeddingsIntoDB()
    return jsonify({"message": "File processed and saved successfully"}), 200

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    index = getIndex()
    embedding_model = getEmbeddingModel()
    retrieved_docs = retrieve_documents(query, index, embedding_model)
    response = generate_response(query, retrieved_docs)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
