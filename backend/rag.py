from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from config import EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS


embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the following
context to answer the question.
If the answer is not in the context, say "I dont know based on the provided document."
Do not make up information.

Context:
{context}

Question: {question}

Answer: """
prompt= PromptTemplate(
    template = PromptTemplate,
    input_variales=["context","question"]
)


def process_document(file_path:str):

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks , embeddings)

    return vectorstore

def answer_question(vectorstore , question:str , groq_api_key:str):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name =LLM_MODEL
    ),

    qa_chain= RetrievalQA.from_chain_type(
        llm = llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k",TOP_K_CHUNKS}
        ),
        chain_type_kwargs={"prompt":prompt},
        return_source_documents=True
    )
    result = qa_chain.invoke({"query":question})
    source_chunks = [doc.page_content for doc in result["source_documents"]]

    return {
        "answer":result["result"],
        "source_chunks":source_chunks
    }