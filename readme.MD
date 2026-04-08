# Document Q&A — RAG System

A RAG (Retrieval Augmented Generation) system built from scratch. Upload any PDF, ask questions, get answers grounded in your document.

Built with LangChain + FAISS + Groq Llama 3 + FastAPI + React.

---

## What This Project Does

- Upload a PDF document
- Ask questions about it in natural language
- Get answers with source chunks shown so you can verify


---

## Why RAG?

LLMs like Llama 3 are trained on data up to a certain date. They don't know about your specific documents — your company manual, your research paper, your product FAQ. You can't retrain the model every time you have new information — that's expensive and slow.

RAG solves this by giving the model relevant information at query time:
1. Store your document as vectors in a database
2. When a question comes in, find the most relevant parts
3. Send those parts + the question to the LLM
4. LLM answers based on your specific document, not general training

---

## Architecture

```
User uploads PDF
      ↓
PyPDFLoader extracts text
      ↓
RecursiveCharacterTextSplitter splits into 500 char chunks
      ↓
HuggingFace all-MiniLM-L6-v2 converts chunks to vectors
      ↓
FAISS stores vectors locally
      ↓
User asks question
      ↓
Same embedding model converts question to vector
      ↓
FAISS finds top 3 most similar chunks (cosine similarity)
      ↓
format_docs joins chunks into one string
      ↓
LangChain fills context + question into prompt template
      ↓
Groq Llama 3.3 70B generates answer from context
      ↓
Answer + source chunks returned to frontend
```

---

## Project Structure

```
document-qa/
    backend/
        main.py        ← FastAPI endpoints only
        rag.py         ← core RAG pipeline
        config.py      ← all configuration in one place
        Dockerfile     ← containerization
        requirements.txt
    frontend/
        src/
            App.tsx    ← upload + Q&A interface
    data/
        golden_dataset.json   ← your test Q&A pairs
    .env                      ← GROQ_API_KEY (never commit this)
```

---

## Tech Stack — And Why

| Tool | What it does | Why this one |
|---|---|---|
| LangChain 0.2.16 | Orchestration | Connects all pieces, industry standard |
| FAISS | Vector database | Runs locally, no server, no cost |
| all-MiniLM-L6-v2 | Embedding model | Small, fast, free, runs locally |
| Groq Llama 3.3 70B | LLM | Free API, very fast inference |
| FastAPI | Backend | Async support, automatic docs |
| React + TypeScript | Frontend | Type safety, component based |
| Docker | Containerization | Runs identically on any machine |

---

## Key Concepts Explained

**What is an embedding?**
Text converted to a vector of numbers where similar meaning = similar numbers = close in space. "The dog ran fast" and "The puppy sprinted quickly" would have very similar vectors even though they share no words.

**What is a vector database?**
A database designed for storing and searching vectors. Instead of "find rows where name = X", you do "find vectors most similar to this query vector". FAISS does this using cosine similarity — measuring the angle between vectors.



**What is chunking and why does it matter?**
You can't send a 100-page PDF to an LLM — too many tokens. You split it into small chunks of 500 characters each and only send the relevant ones. Chunk size affects quality:
- Too large → chunk has too much irrelevant content mixed in
- Too small → chunk doesn't have enough context to be useful

**What is chunk overlap?**
Consecutive chunks share some text. Without overlap, a sentence that falls at the boundary between two chunks gets split — neither chunk has the full context. Overlap prevents this.

---

## Things That Confused Me (And The Answers)

**Where does the question get converted to a vector? I don't see it in the code.**

It happens inside `chain.invoke()` — LangChain does it automatically. The retriever knows which embedding model to use because when we built the vectorstore:
```python
vectorstore = FAISS.from_documents(chunks, embeddings)
```
We passed `embeddings` to FAISS. So FAISS remembers — "I was built with this model." When the retriever searches, it uses the same model to convert the question. You don't see it explicitly because LangChain handles it internally.

**Where are `{context}` and `{input}` being filled? Who fills them and when?**

LangChain fills them automatically when the chain runs. `{context}` gets replaced with the retrieved chunks joined into one string. `{input}` gets replaced with the user's question. You define the template once — LangChain handles the filling at runtime when `chain.invoke(question)` is called.

**Why `k` in `search_kwargs={"k": 3}`?**

`k` stands for "k nearest neighbors" — a standard computer science term meaning "find the k most similar items." It's FAISS's parameter name, not something we invented. You just pass it as a dictionary. `k=3` means return the 3 most similar chunks.

**What is `RunnablePassthrough()`?**

In the chain, `context` goes through the retriever and format_docs — it gets transformed. But `input` should just be the question unchanged. `RunnablePassthrough()` means — pass this value through without any transformation. Think of it as a direct wire with no processing.

**What is `StrOutputParser()`?**

The LLM returns a LangChain message object, not a plain string. `StrOutputParser()` converts that message object into a regular Python string. Without it you'd get something like `AIMessage(content="The answer is...")` instead of just `"The answer is..."`.

**Why do we call retriever.invoke AND chain.invoke separately?**

`chain.invoke` gives us the final answer but doesn't easily expose the individual source chunks.
`retriever.invoke` gives us the raw chunks directly.
We need both — the answer to show the user, and the source chunks to show which parts of the document were used. So we call both separately.

**Why must the same embedding model be used for chunks and questions?**

The vectors must be in the same "space" to be comparable. If chunks are embedded with model A and questions with model B, the similarity scores mean nothing — like comparing temperatures in Celsius and Fahrenheit without converting. The model must be identical for both.

**What is `format_docs` doing? Why not just pass the list directly?**

The retriever returns a list of Document objects. But the prompt template needs a single string for `{context}`. `format_docs` joins all chunks into one string separated by double newlines. Without it, the prompt would receive a Python list object instead of readable text.

**What is the `|` pipe operator in LangChain?**

It chains steps together — output of one step becomes input of the next. Like Unix pipes. So:
```python
retriever | format_docs
```
means: question goes into retriever → gets chunks → chunks go into format_docs → comes out as one string. The full chain works the same way — each step's output feeds the next step.

**Why does the chain need a dictionary as input?**

```python
{"context": retriever | format_docs, "input": RunnablePassthrough()}
```
The prompt template has two placeholders — `{context}` and `{input}`. The dictionary maps each placeholder to its source. `context` gets the retrieved chunks. `input` gets the original question. LangChain uses this mapping to fill the template.

**What happened to `RetrievalQA`? Why did we use a different approach?**

`RetrievalQA` got deprecated in newer LangChain versions. The modern approach uses `create_retrieval_chain` and `create_stuff_documents_chain` instead. Same concept, different API. This is a common frustration with LangChain — it evolves fast and breaks old code. Always check the version when following tutorials.

**Why did we pin LangChain to version 0.2.16?**

LangChain changes its API frequently. Newer versions moved imports around. Pinning the version ensures the code works consistently. In production you always pin dependency versions for this reason.

**What does "stuff" mean in `create_stuff_documents_chain`?**

"Stuff" means take all retrieved documents and stuff them all into the prompt at once. It's the simplest retrieval strategy — just concatenate everything. Alternative strategies exist for very long documents (map-reduce, refine) but for most use cases stuffing works fine.

**Why does the model get decommissioned? What happened with llama3-8b-8192?**

AI providers regularly retire older models and replace them with newer ones. `llama3-8b-8192` was replaced by `llama-3.3-70b-versatile`. This is why model names should always live in `config.py` — when a model gets deprecated, you change one line and everything works again. If the model name was hardcoded in 10 different places, you'd have to find and update all of them.

---

## The LangChain Chain — Full Picture

```python
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)
```

Step by step when `chain.invoke(question)` is called:

1. Build dictionary — retriever finds top 3 chunks, format_docs joins them into one string for `context`. Question passes through unchanged for `input`.
2. PROMPT_TEMPLATE fills `{context}` and `{input}` — produces a complete prompt string
3. `llm` receives the prompt — Groq Llama 3.3 70B generates an answer — returns LangChain message object
4. `StrOutputParser` converts message object to plain Python string

Result: the answer as a plain string.

---

**What is a golden dataset?**
A set of question-answer pairs you create manually from your document, where you already know the correct answer. Like a test with an answer key.

**Why cosine similarity instead of exact match?**
"The return policy is 30 days" and "You have thirty days to return items" mean the same thing but share few words. Cosine similarity compares meaning, not words.

**How to create your golden dataset:**
1. Upload your PDF
2. Read through it manually
3. Write 10-20 questions with answers you can verify directly from the document
4. Save in `data/golden_dataset.json`

---

## Configuration

Everything tunable lives in `config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # embedding model
LLM_MODEL = "llama-3.3-70b-versatile"    # Groq LLM model
CHUNK_SIZE = 500                           # characters per chunk
CHUNK_OVERLAP = 50                         # overlap between chunks
TOP_K_CHUNKS = 3                           # chunks retrieved per question
```

---

## Running Locally

**1. Clone and set up backend**
```bash
git clone https://github.com/yourusername/document-qa
cd document-qa/backend
pip install -r requirements.txt
```

**2. Create .env file**
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at console.groq.com

**3. Start backend**
```bash
uvicorn main:app --reload
```

**4. Set up and start frontend**
```bash
cd ../frontend
npm install
npm run dev
```

**5. Open app**
Go to `http://localhost:5173`

---

## Running with Docker

**Build the image:**
```bash
cd backend
docker build -t document-qa .
```

**Run the container:**
```bash
docker run -p 8000:8000 --env-file .env document-qa
```

`-p 8000:8000` maps port 8000 on your machine to port 8000 inside the container.
`--env-file .env` passes your Groq API key into the container.

**What Docker does here:**
Packages your entire backend — Python version, all dependencies, all code — into one container. Runs identically on any machine. No "works on my machine" problems.

---

## Alternative Approaches

Every choice we made has alternatives. Here's what else exists:

**Instead of FAISS:**
- Chroma — easier to persist to disk
- Pinecone — cloud hosted, scales to millions of vectors
- Qdrant — fast, production grade, runs locally
- pgvector — vector search inside PostgreSQL

**Instead of all-MiniLM-L6-v2:**
- text-embedding-3-small (OpenAI) — better quality, costs money
- bge-large-en (BAAI) — better than MiniLM, still free
- embed-english-v3.0 (Cohere) — very good, has free tier

**Instead of LangChain:**
- LlamaIndex — better for complex document hierarchies
- Haystack — production grade RAG framework
- Raw API calls — full control, no abstraction

**Instead of Groq:**
- OpenAI API — GPT-4, better quality, costs money
- Ollama — run models completely locally, free
- Together AI — free tier, many model options

**Instead of RecursiveCharacterTextSplitter:**
- TokenTextSplitter — splits by tokens, more accurate for context limits
- SemanticChunker — uses embeddings to find natural boundaries, best quality
- MarkdownHeaderTextSplitter — splits by headers, good for structured docs

---

## Limitations and What's Next

**Current limitations:**
- Vectorstore lives in memory — restarting server clears it
- Only supports PDF files
- Single document at a time

**What would make it production ready:**
- Persist vectorstore to disk so it survives restarts
- Support multiple documents simultaneously
- Add reranking for better retrieval accuracy
- Add hybrid search (semantic + keyword) for better coverage
- Stream LLM response instead of waiting for full answer

---

## Author

Built by Richa
