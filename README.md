# Web RAG 

## Description

This project is a question-answering system that uses a pipeline of components to process a user's query, search for relevant documents, and generate a response.
Powered by Haystack 2.0 and Hugging Face Transformers.

The component that searches on the web uses `googlesearch`, which has will return 429 errors if used too much in a short period of time (related issue: [here](https://github.com/Nv7-GitHub/googlesearch/issues/61))


## Installation

1. Clone this repository.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Set your `OPENAI_API_KEY` environment variable to your OpenAI API key.

## Usage

Run the main script with a query as an argument:

```sh
python main.py "your query here"
```




