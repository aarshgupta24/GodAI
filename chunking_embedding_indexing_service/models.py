from datetime import datetime, timezone


class DocumentsCollection:
    def __init__(self, content: str, file_name: str, created_at: datetime = None):
        self.content = content
        self.created_at = created_at or datetime.now(timezone.utc)
        self.file_name = file_name

    def to_dict(self):
        # Convert the object to a dictionary to be compatible with MongoDB insertion
        return {
            "content": self.content,
            "created_at": self.created_at,
            "file_name": self.file_name
        }
