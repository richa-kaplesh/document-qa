from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

def answer_question(vectorstore, question: str, groq_api_key: str):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=config.LLM_MODEL
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K_CHUNKS})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    source_chunks = [doc.page_content for doc in docs]

    return {
        "answer": answer,
        "source_chunks": source_chunks
    }