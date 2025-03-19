import logging

from langchain_core.output_parsers import StrOutputParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlainRAGService:

    def __init__(self, plain_rag_prompt, retriever, llm):
        self.plain_rag_chain = plain_rag_prompt | llm | StrOutputParser()
        self.retriever = retriever
        self.llm = llm
        self.plain_rag_prompt = plain_rag_prompt

    def invoke(self, query: str) -> str:
        docs = self.retriever.retrieve(query)

        generation = self.plain_rag_chain.invoke(
            {"context": "\n".join([doc.page_content for doc in docs]), "question": query})
        return generation

