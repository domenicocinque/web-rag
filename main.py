import logging

import typer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from src.preprocess import make_preprocess_pipeline
from src.search import make_search_pipeline
from src.retrieve import make_retrieve_pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(query: str) -> None:
    """
    Given a user query, reply with an answer given a corpus of documents.

    Args:
        query (str): The user query.
    """
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    preprocess_pipeline = make_preprocess_pipeline()
    search_pipeline = make_search_pipeline(document_store=document_store)
    retrieve_pipeline = make_retrieve_pipeline(document_store=document_store)

    logger.info("Rewriting query...")
    refactored_query = preprocess_pipeline.run({"prompt_builder": {"query": query}})["llm"]["replies"][0]

    logger.info("Searching for relevant documents...")
    search_pipeline.run({"searcher": {"query": refactored_query}})

    logger.info("Building answer...")
    result = retrieve_pipeline.run(
        {"prompt_builder": {"question": query},
         "text_embedder": {"text": query}}
    )
    llm_reply = result["llm"]["replies"][0]
    response = {"reply": llm_reply}

    print(response)


if __name__ == "__main__":
    typer.run(main) 
