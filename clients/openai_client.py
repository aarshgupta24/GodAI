import logging

import openai
import os
from injector import singleton

# Setup logging
logger = logging.getLogger(__name__)


@singleton
class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def create_embeddings(self, text, model):
        response = openai.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def execute_prompt(self, prompt, model="gpt-4o", temperature=0, max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS_LLM", 2500)), response_format=None):
        if response_format is None:
            response_format = {"type": "json_object"}

        try:
            # https://platform.openai.com/docs/api-reference/chat/create  # chat-create-seed
            response = openai.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                messages=[{"role": "user", "content": prompt}],
            )

            output = response.choices[0].message.content
            return output
        except Exception as e:
            logger.error("error occurred in openai: {0}".format(e))
            raise

    def mock_nodes_and_relationships(self):
        # Example: Mockup extracted graph
        nodes = [{"name": "John Doe", "label": "Person"},
                 {"name": "Tech Corp", "label": "Organization"}]
        relationships = [{"source": "John Doe", "target": "Tech Corp", "type": "works_at"}]
        return nodes, relationships
