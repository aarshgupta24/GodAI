import argparse
import json
import logging
import os

from clients.opensearch_client import OpenSearchClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def create_index(index_name):
    if not index_name:
        logger.error("index name is must")
        return
    
    logger.info("creating threat index")
    with open(f"scripts/mappings/embeddings_index_mapping.json") as f:
        index_body = json.load(f)

    os_client = OpenSearchClient(
            os.getenv('OPENSEARCH_HOST'), os.getenv('OPENSEARCH_PORT'), os.getenv('OPENSEARCH_USER'),
            os.getenv('OPENSEARCH_PASSWORD'), index_name)

    response = os_client.create_index(index_body)
    logger.info(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Opensearch index creation script")
    parser.add_argument("--index_name", type=str, help="name of new index which needs to be created")
    args = parser.parse_args()
    index_name = args.index_name
    create_index(index_name)
