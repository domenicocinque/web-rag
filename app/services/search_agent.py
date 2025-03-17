from googlesearch import search
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack import component, default_to_dict


@component
class GoogleSearch:
    def __init__(self, top_k: int = 10, lang: str = "en"):
        self.top_k = top_k
        self.lang = lang

    def to_dict(self):
        return default_to_dict(self, top_k=self.top_k, lang=self.lang)

    @component.output_types(urls=list[str])
    def run(self, query: str):
        results = search(query, num_results=self.top_k, lang=self.lang, advanced=False)
        return {"urls": [link for link in results]}


class SearchAgent:
    def __init__(self, api_key: str, embedding_model: str = "text-embedding-3-small"):
        self.api_key = Secret.from_token(api_key)
        self.embedding_model = embedding_model
        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine"
        )
        self.preprocess_pipeline = self._make_preprocess_pipeline()
        self.search_pipeline = self._make_search_pipeline()
        self.retrieve_pipeline = self._make_retrieve_pipeline()

    @property
    def _search_template(self):
        return """
        Rewrite the following user question for an optimized web search.\n
        Original User Question: {{ query }}

        Instructions:
        - Identify the main topic and key details in the user question.
        - Rewrite the question in a clear, concise format suitable for web search queries.
        - Ensure the rewritten question maintains the original intent and context of the user question.
        - Avoid ambiguous or overly broad terms that might lead to irrelevant search results.
        - Break down complex or multi-part questions into simpler queries if necessary.

        Rewritten question for web search:
        """

    @property
    def _query_template(self):
        return """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    def _make_preprocess_pipeline(self):
        preprocess_pipeline = Pipeline()
        preprocess_pipeline.add_component(
            "prompt_builder", PromptBuilder(template=self._search_template)
        )
        preprocess_pipeline.add_component("llm", OpenAIGenerator(api_key=self.api_key))
        preprocess_pipeline.connect("prompt_builder", "llm")

        return preprocess_pipeline

    def _make_search_pipeline(self):
        search_pipeline = Pipeline()
        search_pipeline.add_component("searcher", GoogleSearch(top_k=10, lang="en"))
        search_pipeline.add_component("fetcher", LinkContentFetcher())
        search_pipeline.add_component("converter", HTMLToDocument())
        search_pipeline.add_component("cleaner", DocumentCleaner())
        search_pipeline.add_component(
            "splitter", DocumentSplitter(split_length=100, split_overlap=0)
        )
        search_pipeline.add_component(
            "embedder",
            OpenAIDocumentEmbedder(api_key=self.api_key, model=self.embedding_model),
        )
        search_pipeline.add_component(
            "writer", DocumentWriter(document_store=self.document_store)
        )

        search_pipeline.connect("searcher", "fetcher")
        search_pipeline.connect("fetcher", "converter")
        search_pipeline.connect("converter", "cleaner")
        search_pipeline.connect("cleaner", "splitter")
        search_pipeline.connect("splitter", "embedder")
        search_pipeline.connect("embedder", "writer")
        return search_pipeline

    def _make_retrieve_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", PromptBuilder(template=self._query_template)
        )
        pipeline.add_component(
            "text_embedder",
            OpenAITextEmbedder(self.api_key, model=self.embedding_model),
        )
        pipeline.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=self.document_store)
        )
        pipeline.add_component("llm", OpenAIGenerator(api_key=self.api_key))
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder")
        pipeline.connect("prompt_builder", "llm")
        return pipeline

    def run(self, query: str) -> str:
        refactored_query = self.preprocess_pipeline.run(
            {"prompt_builder": {"query": query}}
        )["llm"]["replies"][0]

        self.search_pipeline.run({"searcher": {"query": refactored_query}})

        result = self.retrieve_pipeline.run(
            {"prompt_builder": {"question": query}, "text_embedder": {"text": query}}
        )
        llm_reply = result["llm"]["replies"][0]
        return llm_reply
