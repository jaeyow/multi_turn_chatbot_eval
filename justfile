# Multi-turn Chatbot Evaluation - Justfile
# Common commands for development and deployment

# List all available commands
default:
    @just --list

# Install project dependencies
install:
    uv sync

# Install including dev dependencies
dev-install:
    uv sync --dev

# Create and setup virtual environment + install deps + create .env
setup:
    uv sync --dev
    @just env
    @echo "Setup complete! Run: source .venv/bin/activate"

# Create .env file from template (if it doesn't exist)
env:
    @if [ ! -f .env ]; then \
        echo "OPENAI_API_KEY=your_openai_api_key_here" > .env; \
        echo ".env file created. Please update with your OpenAI API key."; \
    else \
        echo ".env file already exists."; \
    fi

# Run the Streamlit UI (recommended for quick start)
streamlit:
    uv run streamlit run streamlit_app.py

# Run the FastAPI server
server:
    uv run uvicorn server:app --reload

# Run the Burr monitoring UI
burr:
    uv run burr

# Generate state machine diagram
diagram:
    uv run python application.py

# Run all tests
test:
    uv run pytest

# Format code with black
format:
    uv run black .

# Lint code with flake8
lint:
    uv run flake8 .

# Type check with mypy
typecheck:
    uv run mypy .

# Check code quality (lint + typecheck)
check: lint typecheck

# Full CI check (format check + lint + typecheck + test)
ci:
    uv run black --check .
    uv run flake8 .
    uv run mypy .
    uv run pytest

# Clean up Python cache files and uv build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Run the application in development mode (server with auto-reload)
dev: server

# Run all evaluations against the default chatbot model (gpt-4o via OpenAI)
# Saves: error_analysis/baseline_scores_gpt-4o.json
# Override judge model : just eval --judge-model gpt-4o
# Resume from a step   : just eval --start-step 4
eval *ARGS:
    uv run python eval_runner.py {{ARGS}}

# Run evaluations with Gemma 4 e2b (local Ollama — must be running on port 11434)
# Saves: error_analysis/baseline_scores_gemma4-e2b.json
eval-gemma2b *ARGS:
    CHATBOT_MODEL=gemma4:e2b CHATBOT_BASE_URL=http://localhost:11434/v1 uv run python eval_runner.py {{ARGS}}

# Run evaluations with Gemma 4 e4b (local Ollama — must be running on port 11434)
# Saves: error_analysis/baseline_scores_gemma4-e4b.json
eval-gemma4b *ARGS:
    CHATBOT_MODEL=gemma4:e4b CHATBOT_BASE_URL=http://localhost:11434/v1 uv run python eval_runner.py {{ARGS}}

# Compare two baselines — accepts model names OR file paths
# Examples:
#   just compare gpt-4o gemma4:e2b
#   just compare gpt-4o gemma4:e4b
#   just compare gemma4:e2b gemma4:e4b
#   just compare error_analysis/baseline_scores_gpt-4o.json error_analysis/baseline_scores_gemma4-e2b.json
compare OLD NEW:
    uv run python compare_baselines.py {{OLD}} {{NEW}}
