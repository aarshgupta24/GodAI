import logging
from bson import ObjectId

from clients.abstract_mongo_client import AbstractMongoDBClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentMongoDBClient(AbstractMongoDBClient):
    def __init__(self, uri, db, documents_collection):
        super().__init__(uri, db)
        self.documents_collection = documents_collection

    def get_document_by_id(self, doc_id):
        return self.find_one(self.documents_collection, {"_id": ObjectId(doc_id)})

    def insert_document(self, document):
        # Insert the document into the collection
        result = self.insert_one(self.documents_collection, document)

        # Return the ObjectId of the newly inserted document
        return result.inserted_id

    def get_documents(self):
        return list(self.find_all(self.documents_collection))

    def check_and_create_mongo_collection(self):
        if self.documents_collection not in self.list_collections():
            self.create_collection(self.documents_collection)
        else:
            logger.info(f"collection {self.documents_collection} already exist")
