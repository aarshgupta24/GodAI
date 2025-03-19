from abc import ABC
from datetime import datetime, timezone

from injector import singleton
from pymongo import MongoClient


@singleton
class AbstractMongoDBClient(ABC):
    def __init__(self, uri, db):
        self.client = MongoClient(uri)
        self.db = self.client[db]

    def find_one(self, collection, filter):
        return self.db[collection].find_one(filter)

    def find_all(self, collection):
        return self.db[collection].find({})

    def insert_one(self, collection, document):
        # Add timestamps
        document['created_at'] = datetime.now(timezone.utc)
        document['updated_at'] = datetime.now(timezone.utc)

        # Insert the document into the collection
        return self.db[collection].insert_one(document)
    
    def list_collections(self):
        return self.db.list_collection_names()

    def create_collection(self, collection):
        self.db.create_collection(collection)
