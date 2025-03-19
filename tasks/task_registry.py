from django.conf import settings

from .chunking_embedding_indexing_task import ChunkingEmbeddingIndexingTask

def chunking_embedding_indexing_task(self, *args):
    settings.di.get(ChunkingEmbeddingIndexingTask).process(*args)
