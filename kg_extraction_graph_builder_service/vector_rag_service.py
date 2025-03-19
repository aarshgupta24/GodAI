import logging
import os

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

TEXT_FIELD = "chunk"
EMBEDDING_FIELD = "embeddings"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorRAGService:
    """Manages vector-based retrieval using OpenSearch."""

    def __init__(self, uri: str, collection_name: str, embedding_model: str):
        try:
            if os.getenv("LLM_CONFIG") == "gemini":
                embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)
            else:
                embedding = OpenAIEmbeddings(model=embedding_model)

            self.vectorstore_retriever = OpenSearchVectorSearch(
                index_name=collection_name,
                embedding_function=embedding,
                opensearch_url=uri,
                vector_field=EMBEDDING_FIELD,  # The field containing your vector embeddings
                text_field=TEXT_FIELD,  # The field containing your text chunks
            ).as_retriever(search_kwargs={"vector_field": EMBEDDING_FIELD,
                                          "text_field": TEXT_FIELD,"k":10})
        except Exception as e:
            logger.error(f"Error initializing Milvus vector store: {e}")
            raise

    def retrieve(self, query: str) -> list:
        """Retrieves documents based on a query."""
        try:
            return self.vectorstore_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
