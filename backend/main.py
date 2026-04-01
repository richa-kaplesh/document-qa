from fastapi import FastAPI , UploadFile, File
from fastapi.middleware.cors import CORSMiddelware
from pydantic import BaseModel
from rag import process_document,answer_question
import os
import shutil
from dotenv import load_dotenv
from eval import run_eval

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddelware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None

class Question(BaseModel):
    question: str

class EvalRequest(BaseModel):
    golden_dataset_path:str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore,doc_info

    file_path = f"temp_{file.filename}"
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    vectorstore,chunk_count = process_document(file_path)

    os.remove(file_path)
    doc_info = {
        "filename":file.filename,
        "chunks":chunk_count
    }
    return {
        "message":"Document processed successfully",
        "filename":file.filename,
        "chunks_created":chunk_count
    }

@app.post("/ask")
async def ask(body: Question):
    if vectorstore is None:
        return {"error": "No Document uploaded yet"}
    answer = answer_question(vectorstore, body.question,groq_api_key)
    return {"answer":answer}

@app.post("/Eval")
async def evaluate(body:EvalRequest):
    if vectorstore is None:
        return {"error":"No document uploaded yet. Please upload a PDF first."}
    results = run_eval(vectorstore,groq_api_key,body.golden_dataset_path)
    return results
   
@app.get("/status")
def status():
    if vectorstore is None:
        return {"status":"No document loaded"}
    return {
        "status":"Ready",
        "document":doc_info
    }

