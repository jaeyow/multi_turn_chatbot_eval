# Multi-turn Chatbot Evaluation - Justfile
# Common commands for development and deployment

# List all available commands
default:
    @just --list

# Install project dependencies
install:
    pip install -r requirements.txt

# Create and setup virtual environment
venv:
    python -m venv .venv
    @echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Run the Streamlit UI (recommended for quick start)
streamlit:
    streamlit run streamlit_app.py

# Run the FastAPI server
server:
    uvicorn server:app --reload

# Run the Burr monitoring UI
burr:
    burr

# Generate state machine diagram
diagram:
    python application.py

# Run all tests (if tests exist)
test:
    pytest

# Format code with black
format:
    black .

# Lint code with flake8
lint:
    flake8 .

# Type check with mypy
typecheck:
    mypy .

# Clean up Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Create .env file from template (if it doesn't exist)
env:
    @if [ ! -f .env ]; then \
        echo "OPENAI_API_KEY=your_openai_api_key_here" > .env; \
        echo ".env file created. Please update with your OpenAI API key."; \
    else \
        echo ".env file already exists."; \
    fi

# Install development dependencies
dev-install:
    pip install black flake8 mypy pytest pytest-asyncio

# Run a complete setup (venv + install + env)
setup: venv
    .venv/bin/pip install -r requirements.txt
    @just env
    @echo "Setup complete! Activate venv with: source .venv/bin/activate"

# Run the application in development mode (server with auto-reload)
dev: server

# Check code quality (lint + typecheck)
check: lint typecheck

# Full CI check (format check + lint + typecheck + test)
ci:
    black --check .
    flake8 .
    mypy .
    pytest
