# Document Q&A — RAG System

A production-grade RAG (Retrieval Augmented Generation) system built from scratch. Upload any PDF, ask questions, get answers grounded in your document — with a full evaluation harness to measure accuracy.

Built with LangChain + FAISS + Groq Llama 3 + FastAPI + React.

---

## What This Project Does

- Upload a PDF document
- Ask questions about it in natural language
- Get answers with source chunks shown (so you can verify)
- Run an evaluation harness against a golden dataset to measure RAG accuracy

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
LangChain builds prompt: context + question
      ↓
Groq Llama 3 generates answer from context
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
        eval.py        ← evaluation harness
        config.py      ← all configuration in one place
        Dockerfile
        requirements.txt
    frontend/
        src/
            App.tsx    ← upload + Q&A interface
            Eval.tsx   ← evaluation dashboard
    data/
        golden_dataset.json   ← your test Q&A pairs
    .env                      ← GROQ_API_KEY (never commit this)
```

---

## Tech Stack — And Why

| Tool | What it does | Why this one |
|---|---|---|
| LangChain | Orchestration | Connects all pieces, industry standard |
| FAISS | Vector database | Runs locally, no server, no cost |
| all-MiniLM-L6-v2 | Embedding model | Small, fast, free, runs locally |
| Groq Llama 3 | LLM | Free API, very fast inference |
| FastAPI | Backend | You already know it, async support |
| React + TypeScript | Frontend | You already know it |

---

## Key Concepts Explained

**What is an embedding?**
Text converted to a vector of numbers where similar meaning = similar numbers = close together in space. "The dog ran fast" and "The puppy sprinted quickly" would have very similar vectors even though they share no words.

**What is a vector database?**
A database designed for storing and searching vectors. Instead of "find rows where name = X", you do "find vectors most similar to this query vector". FAISS does this using cosine similarity — measuring the angle between vectors.

**What is cosine similarity?**
A measure of how similar two vectors are. Ranges from 0 (completely different) to 1 (identical). Used both for retrieval (finding relevant chunks) and evaluation (comparing expected vs actual answers).

**What is chunking and why does it matter?**
You can't send a 100-page PDF to an LLM — too many tokens. You split it into small chunks and only send the relevant ones. Chunk size affects quality:
- Too large → chunk has too much irrelevant content mixed in
- Too small → chunk doesn't have enough context to be useful
500 characters with 50 character overlap works well for most documents.

**What is chunk overlap?**
Consecutive chunks share some text. Without overlap, a sentence that falls at the boundary between two chunks gets split — neither chunk has the full context. Overlap prevents this.

**Why do we use the same embedding model for chunks and questions?**
The vectors must be in the same "space" to be comparable. If chunks are embedded with model A and questions with model B, the similarity scores mean nothing — like comparing temperatures in Celsius and Fahrenheit without converting.

**What is the system prompt and why does it matter?**
We tell Llama 3: "Only answer from the provided context. Say I don't know if the answer isn't there." Without this, the LLM uses its general training knowledge to fill gaps — which leads to hallucination (making up plausible-sounding but wrong answers).

---

## Evaluation Harness

This is the part most RAG tutorials skip. We measure our system's accuracy systematically.

**What is a golden dataset?**
A set of question-answer pairs you create manually from your document, where you already know the correct answer. Like a test with an answer key.

**How evaluation works:**
1. For each question in the golden dataset, ask the RAG system
2. Compare the RAG answer to the expected answer using cosine similarity
3. If similarity >= threshold (0.7 by default), it passes
4. Report overall pass rate

**Why cosine similarity instead of exact match?**
"The return policy is 30 days" and "You have thirty days to return items" mean the same thing but share few words. Cosine similarity compares meaning, not words.

**How to create your golden dataset:**
1. Upload your PDF
2. Read through it manually
3. Write 10-20 questions with answers you can verify directly from the document
4. Save in `data/golden_dataset.json`

---

## Configuration

Everything tunable lives in `config.py`. Change these to improve performance:

```python
CHUNK_SIZE = 500        # increase for more context per chunk
CHUNK_OVERLAP = 50      # increase if answers get cut off at boundaries
TOP_K_CHUNKS = 3        # increase to give LLM more context
SIMILARITY_THRESHOLD = 0.7  # lower to be more lenient in eval
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

```bash
cd backend
docker build -t document-qa .
docker run -p 8000:8000 --env-file .env document-qa
```

---

## Limitations and What's Next

**Current limitations:**
- Vectorstore lives in memory — restarting the server clears it
- Only supports PDF files
- Single document at a time

**What would make it production-ready:**
- Persist vectorstore to disk so it survives restarts
- Support multiple documents simultaneously
- Add reranking for better retrieval accuracy
- Add hybrid search (semantic + keyword) for better coverage
- Stream the LLM response instead of waiting for full answer

---

## Author

Built by Richa