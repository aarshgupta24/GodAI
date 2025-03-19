import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from dependency_injection.dependency_injector import injector
from kg_extraction_graph_builder_service.llm_retrieval_service import PlainRAGService
from tasks.chunking_embedding_indexing_task import ChunkingEmbeddingIndexingTask
from tasks.extraction_task import ExtractionTask

# Initialize logger
logger = logging.getLogger(__name__)


class ExtractContentAPI(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = injector.get(ExtractionTask)
        logger.info(self.task)

    def post(self, request):
        # get the file from the request
        if 'file' not in request.FILES:
            return Response({"error": "No file provided"}, status=400)

        # Get the uploaded file
        uploaded_file = request.FILES['file']

        # Extract content from file and save to mongo
        doc_id = self.task.process(uploaded_file)

        return Response({"status": "success", "doc_id": str(doc_id)})


class EmbedContentAPI(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = injector.get(ChunkingEmbeddingIndexingTask)

    def post(self, request):
        doc_id = request.query_params.get('doc_id')
        indexed_chunk_ids = self.task.process(doc_id)
        return Response({"status": "success", "chunk_ids": indexed_chunk_ids})


class RunPlainRagQueryView(APIView):
    """
    API endpoint to run a plain RAG query using vector storage.
    """

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plain_rag_service = injector.get(PlainRAGService)

    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            logger.info(f"Running plain RAG query: {query}")
            result = self.plain_rag_service.invoke(query)
            return Response({"result": result})
        except Exception as e:
            logger.error(f"Error running plain RAG query: {e}")
            return Response({"error": "Failed to process the query"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
