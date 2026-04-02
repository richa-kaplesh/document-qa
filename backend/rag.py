from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import config 



PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""You are a helpful assisytant. Use only the following context to answer the question.
If the answer is not in the context, say "I dont know based on the provided document."
Do not make up information.

Context:
{context}

Question: {input}

Answer: """)
    

def process_document(file_path:str):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks , embeddings)

    return vectorstore , len(chunks)

def answer_question(vectorstore , question:str , groq_api_key:str):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name =config.LLM_MODEL
    )

    document_chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k":config.TOP_K_CHUNKS}),
        document_chain
    )

    
    result = retrieval_chain.invoke({"input":question})
    source_chunks = [doc.page_content for doc in result["context"]]

    return {
        "answer":result["result"],
        "source_chunks":source_chunks
    }