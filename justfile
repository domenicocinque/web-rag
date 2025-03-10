# List all the available commands.
help:
    just --list

# Run the app with the specified environment.
run *ENV:
    ENV={{ ENV }} uv run python -m uvicorn app.main:app --reload

# Run the test and optionally specify the specific test folder.
test *FOLDER:
    ENV="test" uv run pytest test/{{ FOLDER }} --cov -vv

# Run ruff format
fmt:
    uvx ruff format

# Run ruff check
lint:
    uvx ruff check --fix