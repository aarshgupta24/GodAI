import os

from injector import inject
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader

from api.models import DocumentsCollection
from clients import DocumentMongoDBClient


class ExtractionTask:
    LOADER_MAPPING = {
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        # Add more mappings for other file extensions and loaders as needed
    }

    @inject
    def __init__(self, document_mongodb_client: DocumentMongoDBClient, **kwargs):
        super().__init__(**kwargs)
        self.document_mongodb_client = document_mongodb_client

    def process(self, uploaded_file):
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Extract the content from the file
        content = self.extract_doc(file_path)
        if not content:
            raise ValueError(f"No content found")
        content = content.strip()

        # Save it to the mongo
        doc_id = self.save_file_to_mongo(content, uploaded_file.name)

        # Remove temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        return doc_id

    def save_file_to_mongo(self, content, uploaded_file_name):
        content_obj = DocumentsCollection(content, uploaded_file_name)

        doc_id = self.document_mongodb_client.insert_document(content_obj.to_dict())
        return doc_id

    def extract_doc(self, file_path) -> str:
        # Extract the file extension from the provided file path
        ext = "." + file_path.rsplit(".", 1)[-1]

        # Check if the file extension is supported by the loader mapping
        if not ext in self.LOADER_MAPPING:
            # Raise an error if the file extension is not supported by the loader mapping
            raise ValueError(f"Unsupported file extension '{ext}'")

        try:
            # Retrieve the appropriate loader class and arguments from the mapping
            loader_class, loader_args = self.LOADER_MAPPING[ext]

            # Initialize the loader with the file path and any additional arguments
            loader = loader_class(file_path, **loader_args)

            # Load the content from the file, which may be divided into pages
            pages = loader.load()

            # Combine the content from all pages into a single string
            return "".join([page.page_content for page in pages])

        # If an error occurs, attempt an alternative handling for .docx files
        except Exception as e:
            if ext == ".docx":
                from docx import Document

                # Use python-docx to read the .docx file content
                document = Document(file_path)

                # Extract and join all paragraph texts with line breaks
                return "".join([paragraph.text + "\n" for paragraph in document.paragraphs])
