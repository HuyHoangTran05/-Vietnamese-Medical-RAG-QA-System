from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Testing project setup")
    logger.info(f"Embedding model: {Config.EMBEDDING_MODEL}")
    logger.info(f"Top K: {Config.TOP_K}")
    logger.info(f"Chunk size: {Config.CHUNK_SIZE}")
    logger.info(f"Log path: {Config.LOG_PATH}")

    print("Setup is working!")


if __name__ == "__main__":
    main()