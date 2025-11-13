from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, filter_to_minimal_docs, text_split, get_bm25_retriever
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from werkzeug.utils import secure_filename
from pypdf.errors import PdfReadError
from langchain.document_loaders import PyPDFLoader

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2,google_api_key=GEMINI_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

UPLOAD_FOLDER = 'uploaded_pdfs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('chat.html')

bm25_retriever = None

def hybrid_retrieve(query):
    # If bm25_retriever is not initialized, use only vector retriever
    if bm25_retriever is None:
        return retriever.get_relevant_documents(query)
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    vector_docs = retriever.get_relevant_documents(query)
    # Combine and deduplicate (by page_content)
    all_docs = {doc.page_content: doc for doc in bm25_docs + vector_docs}
    return list(all_docs.values())

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    # Use hybrid retrieval
    docs = hybrid_retrieve(msg)
    response = question_answer_chain.invoke({"input": msg, "context": docs})
    print("Response : ", response)
    return str(response)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(save_path)
        try:
            loader = PyPDFLoader(save_path)
            docs = loader.load()
            minimal_docs = filter_to_minimal_docs(docs)
            chunks = text_split(minimal_docs)
            docsearch.add_documents(documents=chunks)
            global bm25_retriever
            bm25_retriever = get_bm25_retriever(chunks)
            return jsonify({"success": True, "filename": filename}), 200
        except PdfReadError:
            os.remove(save_path)
            return jsonify({"error": "Uploaded PDF is corrupted or unreadable."}), 400
        except Exception as e:
            os.remove(save_path)
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
