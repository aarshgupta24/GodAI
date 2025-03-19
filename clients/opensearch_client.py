import logging

from opensearchpy import OpenSearch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenSearchClient:
    def __init__(self, host, port, user, password, index):
        self.client = OpenSearch(hosts=[{'host': host, 'port': port}],
                                 http_compress=True,
                                 http_auth=(user, password) if user else None,
                                 use_ssl=False,
                                 verify_certs=False,
                                 ssl_show_warn=False)
        self.index = index

    def get_chunk_by_id(self, chunk_id):
        response = self.client.search(index=self.index, body={
            "size": 1,
            "query": {"term": {"id": chunk_id}}
        })
        logger.debug(f"Response from Opensearch for chunk {chunk_id}: {response}")
        if response['hits']['hits']:
            return response['hits']['hits'][0]['_source']
        return ""

    def get_max_id(self):
        logger.debug(f"Querying Opensearch for max id")
        response = self.client.search(index=self.index, body={
            "size": 1,
            "_source": "id",
            "sort": [{"id": "desc"}]
        })
        logger.debug(f"Response from Opensearch for max id: {response}")
        if response['hits']['hits']:
            max_id = response['hits']['hits'][0]['_source']['id']
            return int(max_id)
        else:
            return 0

    def index_doc(self, doc):
        if doc.get('id') is None or doc.get('id', 0) == 0:
            doc_id = self.get_max_id() + 1
            doc['id'] = doc_id
        response = self.client.index(index=self.index, id=doc['id'], body=doc, refresh=True)
        return int(response['_id'])
    
    def check_and_create_opensearch_index(self, index_body):
        if not self.client.indices.exists(index=self.index):
            self.create_index(index_body)
        else:
            logger.info(f"index '{self.index}' already exists.")

    def create_index(self, index_body):
        self.client.indices.create(self.index, body=index_body)
