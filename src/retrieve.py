import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder


query_template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

def make_retrieve_pipeline(document_store):
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=query_template))
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipeline.add_component("llm", OpenAIGenerator(api_key= os.getenv("OPENAI_API_KEY")))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder")
    pipeline.connect("prompt_builder", "llm")
    return pipeline