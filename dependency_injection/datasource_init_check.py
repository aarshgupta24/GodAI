import json

from clients import DocumentMongoDBClient, OpenSearchClient
from dependency_injection.dependency_injector import injector

os_client = injector.get(OpenSearchClient)
mongo_client = injector.get(DocumentMongoDBClient)


def check_index_collection():
    # Opensearch index check
    with open("scripts/mappings/embeddings_index_mapping_openai.json") as f:
            index_body = json.load(f)

    os_client.check_and_create_opensearch_index(index_body)

    # MongoDB collection check
    mongo_client.check_and_create_mongo_collection()
