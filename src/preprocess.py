import os
from haystack import Pipeline 
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

search_template = """
Rewrite the following user question for an optimized web search.\n
Original User Question: {{ query }}

Instructions:
- Identify the main topic and key details in the user question.
- Rewrite the question in a clear, concise format suitable for web search queries.
- Ensure the rewritten question maintains the original intent and context of the user question.
- Avoid ambiguous or overly broad terms that might lead to irrelevant search results.
- Break down complex or multi-part questions into simpler queries if necessary.

Rewritten Question for Web Search:
"""


def make_preprocess_pipeline():
    preprocess_pipeline = Pipeline()
    preprocess_pipeline.add_component("prompt_builder", PromptBuilder(template=search_template))
    preprocess_pipeline.add_component("llm", OpenAIGenerator(api_key=os.getenv("OPENAI_API_KEY")))
    preprocess_pipeline.connect("prompt_builder", "llm")

    return preprocess_pipeline
