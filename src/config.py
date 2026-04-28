import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Models
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "intfloat/multilingual-e5-base"
    )
    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "gpt-4o-mini"
    )

    # Paths
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")
    PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "indexes/faiss.index")
    METADATA_PATH = os.getenv("METADATA_PATH", "indexes/metadata.pkl")
    LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")

    # RAG configs
    TOP_K = int(os.getenv("TOP_K", 5))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

    # App
    APP_ENV = os.getenv("APP_ENV", "development")