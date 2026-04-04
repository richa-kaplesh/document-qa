from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL= "all-MiniLM-L6-v2"

CHUNK_SIZE=500
CHUNK_OVERLAP=50

TOP_K_CHUNKS=3

SIMILARITY_THRESHOLD = 0.7