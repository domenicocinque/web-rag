# Web RAG 

## Description

Example of FastAPI for web-based RAG using Haystack. 

The component that searches on the web uses the `googlesearch` library, which will return 429 errors if used too much in a short period of time (related issue: [here](https://github.com/Nv7-GitHub/googlesearch/issues/61))


## Installation

1. Clone this repository.
2. Install the required Python packages by running `pip install -r requirements.txt`, or `uv sync`.
3. Set your `OPENAI_API_KEY` environment variable to your OpenAI API key.

## Usage

Run the app with 

```sh
python -m uvicorn app.main:app
```

or with `just run dev`. 





