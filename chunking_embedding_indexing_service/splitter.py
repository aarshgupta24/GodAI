import logging

import tiktoken

# Setup logging
logger = logging.getLogger(__name__)


class SemanticParagraphSplitter:
    def __init__(self, chunk_size=2000, model="gpt-3.5-turbo", byte_limit=10000):
        """
        Initialize the TextChunker with a specified token threshold and byte limit for embeddings.

        :param chunk_size: The maximum token size allowed per chunk.
        :param byte_limit: The maximum byte size for each chunk when sending to the embedding service.
        """
        self.chunk_size = chunk_size
        self.model = model
        self.byte_limit = byte_limit

    def split_text(self, text):
        logging.info("Splitting the text into chunks")
        # Initialize the tokenizer based on the model
        enc = tiktoken.encoding_for_model(self.model)

        # Initialize a list to hold the chunks of text
        chunks = []
        current_chunk = []
        current_token_count = 0
        current_byte_size = 0

        # Split the text into paragraphs
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            paragraph_tokens = enc.encode(paragraph)
            paragraph_token_count = len(paragraph_tokens)
            paragraph_byte_size = len(enc.decode(paragraph_tokens).encode('utf-8'))  # Get byte size

            # Check if adding this paragraph exceeds the token limit or byte size limit
            if current_token_count + paragraph_token_count > self.chunk_size or current_byte_size + paragraph_byte_size > self.byte_limit:
                # If it exceeds, store the current chunk and reset the chunk variables
                chunks.append(enc.decode(current_chunk))
                current_chunk = paragraph_tokens
                current_token_count = paragraph_token_count
                current_byte_size = paragraph_byte_size
            else:
                # Otherwise, add the paragraph to the current chunk
                current_chunk.extend(paragraph_tokens)
                current_token_count += paragraph_token_count
                current_byte_size += paragraph_byte_size

        # Append any remaining tokens as a final chunk
        if current_chunk:
            chunks.append(enc.decode(current_chunk))
        return chunks
