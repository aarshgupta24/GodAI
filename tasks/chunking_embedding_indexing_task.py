import logging
import os

from injector import inject

from chunking_embedding_indexing_service import Embedder, SemanticParagraphSplitter
from clients import DocumentMongoDBClient, OpenSearchClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingEmbeddingIndexingTask:

    @inject
    def __init__(self, document_mongodb_client: DocumentMongoDBClient,
                 opensearch_client: OpenSearchClient, *args,
                 embedder: Embedder, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_mongodb_client = document_mongodb_client
        self.opensearch_client = opensearch_client
        self.embedder = embedder
        self.splitter = SemanticParagraphSplitter(chunk_size=int(os.getenv("TEXT_CHUNK_SIZE", 2000)))

    def process(self, doc_id):
        logger.info(f'Processing document: {doc_id} for chunking, embedding & indexing...')
        documents = [self.get_document_from_db(doc_id)] if doc_id else self.document_mongodb_client.get_documents()
        return [self.process_doc(doc_id, document) for document in documents]

    def process_doc(self, doc_id, document):
        if document:
            chunks = self.create_chunks(doc_id, document['content'])

            chunks_to_index = []
            for index, chunk in enumerate(chunks, start=1):
                embeddings = self.create_embeddings(doc_id, index, chunk)
                chunk_index_doc = {
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "chunk": chunk,
                    "embeddings": embeddings,
                }
                chunks_to_index.append(chunk_index_doc)
            indexed_chunk_ids = self.index_chunks(doc_id, chunks_to_index)
            logger.debug(f"Indexed chunk ids for doc #{doc_id}: {indexed_chunk_ids}")
            return indexed_chunk_ids
        else:
            logger.error(f'Document not found with id: {doc_id} in mongo.')
            return []

    def get_document_from_db(self, doc_id):
        # Fetch the document using the MongoDB client
        return self.document_mongodb_client.get_document_by_id(doc_id)

    def create_chunks(self, doc_id, content):
        # Split the content into chunks
        chunks = self.splitter.split_text(content)
        logger.info(f'Created {len(chunks)} chunks for document: {doc_id}.')
        return chunks

    def create_embeddings(self, doc_id, chunk_index, chunk):
        # Generate the vector embeddings for chunk
        embeddings = self.embedder.get_embeddings(chunk)
        logger.info(f'Generated embeddings for the chunk index: {chunk_index} with doc_id: {doc_id}')
        return embeddings

    def index_chunks(self, doc_id, chunks):
        # Logic to index chunks in OpenSearch
        indexed_chunk_ids = []
        for i, chunk in enumerate(chunks, start=1):
            logger.debug(f'Indexing chunk {i} for document {doc_id} in opensearch')
            id = self.opensearch_client.index_doc(chunk)
            indexed_chunk_ids.append(id)
        logger.info(f'Indexed {len(chunks)} chunks for document: {doc_id} in opensearch.')
        return indexed_chunk_ids
