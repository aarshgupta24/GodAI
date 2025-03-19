import os

from injector import Module, provider, singleton, Injector
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from chunking_embedding_indexing_service import Embedder, OpenAIEmbedder, SentenceTransformerEmbedder, GeminiEmbedder
from clients import DocumentMongoDBClient, OpenAIClient, GeminiClient, OpenSearchClient
from configs.prompt import plain_rag_prompt
from kg_extraction_graph_builder_service.llm_retrieval_service import PlainRAGService
from kg_extraction_graph_builder_service.vector_rag_service import VectorRAGService
from tasks.chunking_embedding_indexing_task import ChunkingEmbeddingIndexingTask
from tasks.extraction_task import ExtractionTask

class DependencyInjector(Module):

    @singleton
    @provider
    def provide_document_mongo_db_client(self) -> DocumentMongoDBClient:
        return DocumentMongoDBClient(os.getenv('MONGODB_URI'), os.getenv('MONGO_DB'),
                                     os.getenv('MONGO_DOCUMENTS_COLLECTION'))

    @singleton
    @provider
    def provide_opensearch_client(self) -> OpenSearchClient:
        return OpenSearchClient(
            os.getenv('OPENSEARCH_HOST'), os.getenv('OPENSEARCH_PORT'), os.getenv('OPENSEARCH_USER'),
            os.getenv('OPENSEARCH_PASSWORD'), self.get_opensearch_index_name())

    @singleton
    @provider
    def provide_openai_client(self) -> OpenAIClient:
        return OpenAIClient(os.getenv('OPENAI_API_KEY'))
    
    @singleton
    @provider
    def provide_gemini_client(self) -> GeminiClient:
        return GeminiClient(os.getenv('GOOGLE_API_KEY'))

    @singleton
    @provider
    def provide_embedder(self) -> Embedder:
        if os.getenv("EMBEDDER_NAME") == "sentence_transformer":
            return SentenceTransformerEmbedder()

        if os.getenv("EMBEDDER_NAME") == "gemini_embedder":
            return GeminiEmbedder(self.provide_gemini_client())
        
        return OpenAIEmbedder(self.provide_openai_client())
    
    def get_opensearch_index_name(self):
        prefix = os.getenv('OPENSEARCH_CHUNK_INDEX_PREFIX')
        index_name = f'{prefix}'

        return index_name

    @singleton
    @provider
    def provide_opensearch_vector_store(self) -> VectorRAGService:
        if os.getenv("LLM_CONFIG") == "gemini":
            return VectorRAGService(
                os.getenv('OPENSEARCH_URI'),
                self.get_opensearch_index_name(),
                os.getenv('GEMINI_EMBEDDINGS_MODEL'))
        else:
            return VectorRAGService(
                os.getenv('OPENSEARCH_URI'),
                self.get_opensearch_index_name(),
                os.getenv('OPENAI_EMBEDDINGS_MODEL'))
            

    @singleton
    @provider
    def get_document_extraction_task(self) -> ExtractionTask:
        return ExtractionTask(
            self.provide_document_mongo_db_client()
        )

    @singleton
    @provider
    def get_chunk_embedding_task(self) -> ChunkingEmbeddingIndexingTask:
        return ChunkingEmbeddingIndexingTask(
            self.provide_document_mongo_db_client(),
            self.provide_opensearch_client(),
            embedder=self.provide_embedder()
        )

    @singleton
    @provider
    def get_plain_rag_service(self) -> PlainRAGService:
        llm = self.get_chat_llm()
        return PlainRAGService(
            plain_rag_prompt, self.provide_opensearch_vector_store(),llm)
    
    def get_chat_llm(self):
        if os.getenv("LLM_CONFIG") == "gemini":
            return ChatGoogleGenerativeAI(temperature=os.getenv('EXTRACTION_TEMP'), verbose=True,
                          model=os.getenv('GEMINI_EXTRACTION_MODEL'))
        else:
            return ChatOpenAI(temperature=os.getenv('EXTRACTION_TEMP'), verbose=True,
                          model_name=os.getenv('OPENAI_EXTRACTION_MODEL'))

# Initialize the injector
injector = Injector([DependencyInjector()])
