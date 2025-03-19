import logging
import os
import google.generativeai as genai
from injector import singleton

# Setup logging
logger = logging.getLogger(__name__)


@singleton
class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def create_embeddings(self, text, model="text-embedding-004"):
        response = genai.embed_content(model=model, content=text)
        return response.get("embedding")

    def execute_prompt(self, prompt, model="gemini-2.0-flash-exp", temperature=0, max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS_LLM", 2500)), response_format=None):
        if response_format is None:
            response_format = "application/json"
        try:
            model = genai.GenerativeModel(model)
            genai_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type=response_format
            )
            response = model.generate_content(prompt, generation_config=genai_config)
            return response.text if response else ""
        except Exception as e:
            logger.error(f"Error occurred while executing prompt: {e}")
            raise

    def mock_nodes_and_relationships(self):
        """
        Mock method to generate nodes and relationships.
        """
        nodes = [{"name": "Jane Doe", "label": "Person"},
                 {"name": "Tech Innovations Ltd.", "label": "Organization"}]
        relationships = [{"source": "Jane Doe", "target": "Tech Innovations Ltd.", "type": "employed_by"}]
        return nodes, relationships
