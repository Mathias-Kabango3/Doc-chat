import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException #type:ignore
from fastapi.middleware.cors import CORSMiddleware #type:ignore
from pydantic import BaseModel #type:ignore
from dotenv import load_dotenv #type:ignore
from supabase import create_client, Client #type:ignore

# LangChain components
from langchain_community.document_loaders import PyPDFLoader #type:ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #type:ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type:ignore
from langchain_community.vectorstores import FAISS #type:ignore
from langchain.chains.combine_documents import create_stuff_documents_chain #type:ignore
from langchain.prompts import ChatPromptTemplate #type:ignore

# Load environment variables
load_dotenv()

# --- Initialize Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
PDF_BUCKET_NAME = "pdfs"
INDEX_BUCKET_NAME = "indexes"

# --- Initialize FastAPI App ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    document_id: str
    question: str

# --- Helper Functions ---
def process_and_store_pdf(file_path: str, document_id: str):
    """Loads a PDF, splits it, creates embeddings, and stores it in FAISS."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to a local file first
    index_path = f"temp_indexes/{document_id}"
    vector_store.save_local(index_path)

    # Upload the index files to Supabase Storage
    for file_name in os.listdir(index_path):
        supabase.storage.from_(INDEX_BUCKET_NAME).upload(
            file=f"{index_path}/{file_name}",
            path=f"{document_id}/{file_name}",
        )
    
    # Clean up local temp files
    shutil.rmtree(index_path)
    os.remove(file_path)

def load_vector_store_from_supabase(document_id: str) -> FAISS:
    """Downloads a FAISS index from Supabase and loads it."""
    index_path = f"temp_indexes/{document_id}"
    os.makedirs(index_path, exist_ok=True)

    # Download all files for the given index
    files = supabase.storage.from_(INDEX_BUCKET_NAME).list(path=document_id)
    for file_obj in files:
        response = supabase.storage.from_(INDEX_BUCKET_NAME).download(f"{document_id}/{file_obj['name']}")
        with open(f"{index_path}/{file_obj['name']}", "wb+") as f:
            f.write(response)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True) # Important flag for FAISS
    
    # Clean up
    shutil.rmtree(index_path)
    return vector_store

# --- API Endpoints ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handles PDF upload, processing, and storage.
    Returns a unique document_id for querying later.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Use filename as a simple unique ID (in a real app, use UUIDs)
    document_id = os.path.splitext(file.filename)[0]
    
    # Save PDF temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Upload original PDF to Supabase
        with open(temp_file_path, "rb") as f:
            supabase.storage.from_(PDF_BUCKET_NAME).upload(
                file=f, path=f"{document_id}.pdf", file_options={"content-type": "application/pdf"}
            )
        
        # Process the PDF and store its vector index
        process_and_store_pdf(temp_file_path, document_id)
        
        return {"document_id": document_id, "filename": file.filename}
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/chat")
async def chat_with_doc(request: QueryRequest):
    """
    Handles chat queries for a specific document.
    """
    try:
        # Load the specific vector store for the document
        vector_store = load_vector_store_from_supabase(request.document_id)
        retriever = vector_store.as_retriever()

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If you don't know the answer, just say that you don't know.
        
        <context>
        {context}
        </context>

        Question: {input}
        """)
        
        # Create the chain
        llm = ChatOpenAI(model="gpt-4o")
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Get relevant documents from the retriever
        docs = retriever.invoke(request.question)
        
        # Get the answer
        response = document_chain.invoke({
            "input": request.question,
            "context": docs
        })
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    


@app.get("/documents")
async def get_all_documents():
    """
    Retrieves a list of all uploaded documents from the 'pdfs' bucket.
    """
    try:
        files = supabase.storage.from_(PDF_BUCKET_NAME).list()
        # The list contains more data, we just want the names
        # filter out any potential system files like .emptyFolderPlaceholder
        document_list = [
            {"id": os.path.splitext(file['name'])[0], "name": file['name']}
            for file in files if file['name'] != '.emptyFolderPlaceholder'
        ]
        return document_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

