import logging
import os
from abc import ABC, abstractmethod

from injector import inject
from sentence_transformers import SentenceTransformer

from clients import OpenAIClient, GeminiClient

# Setup logging
logger = logging.getLogger(__name__)


class Embedder(ABC):
    @abstractmethod
    def get_embeddings(self, text):
        pass


class OpenAIEmbedder(Embedder):
    @inject
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize the OpenAIEmbedder with a specified OpenAI model.
        Default model is 'text-embedding-ada-002'.
        """
        self.openai_client = openai_client
        self.model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL")

    def get_embeddings(self, text):
        """
        Get embeddings for a single text string using OpenAI's model.

        :param text: A string to embed.
        :return: Embedding as a list of floats.
        """
        try:
            return self.openai_client.create_embeddings(text, self.model_name)
        except Exception as e:
            logger.error("Error occurred while getting embeddings: {0}".format(e))
            raise

class GeminiEmbedder(Embedder):
    @inject
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the GeminiEmbedder with a specified Gemini model.
        Default model is 'text-embedding-004'.
        """
        self.gemini_client = gemini_client
        self.model_name = os.getenv("GEMINI_EMBEDDINGS_MODEL")

    def get_embeddings(self, text):
        """
        Get embeddings for a single text string using Gemini's model.

        :param text: A string to embed.
        :return: Embedding as a list of floats.
        """
        try:
            return self.gemini_client.create_embeddings(text, self.model_name)
        except Exception as e:
            logger.error("Error occurred while getting embeddings: {0}".format(e))
            raise


class SentenceTransformerEmbedder(Embedder):
    def __init__(self):
        """
        Initialize the SentenceTransformerEmbedder with a specified SentenceTransformer model.
        """
        self.model = SentenceTransformer(os.getenv("EMBEDDINGS_MODEL_NAME_SENTENCE_TRANSFORMER"))

    def get_embeddings(self, text):
        """
        Get embeddings for a single text string using SentenceTransformer.

        :param text: A string to embed.
        :return: Embedding as a numpy array.
        """
        # Get embeddings for the provided text
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding
