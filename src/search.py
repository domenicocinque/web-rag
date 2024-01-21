from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from googlesearch import search
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
    

def make_search_pipeline(document_store):
    search_pipeline = Pipeline()
    search_pipeline.add_component("searcher", GoogleSearch(top_k=10, lang="en"))
    search_pipeline.add_component("fetcher", LinkContentFetcher())
    search_pipeline.add_component("converter", HTMLToDocument())
    search_pipeline.add_component("cleaner", DocumentCleaner())
    search_pipeline.add_component("splitter", DocumentSplitter(split_length=100, split_overlap=0))
    search_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    search_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    search_pipeline.connect("searcher", "fetcher")
    search_pipeline.connect("fetcher", "converter")
    search_pipeline.connect("converter", "cleaner")
    search_pipeline.connect("cleaner", "splitter")
    search_pipeline.connect("splitter", "embedder")
    search_pipeline.connect("embedder", "writer")

    return search_pipeline




