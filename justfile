# Thermal-Flow developer tasks. Run `just` (or `just --list`) to see all recipes.

# List all available recipes
default:
    @just --list

# Install/refresh the uv-managed environment
sync:
    uv sync

# Run the test suite
test *args:
    uv run pytest {{ args }}

# Lint with ruff
lint:
    uv run ruff check .

# Auto-fix lint issues where possible
lint-fix:
    uv run ruff check --fix .

# Check formatting without modifying files
format-check:
    uv run ruff format --check .

# Format the codebase
format:
    uv run ruff format .

# Run everything CI runs: lint, format check, tests
ci: lint format-check test

# Log in to Weights & Biases (only needed once per machine)
wandb-login:
    uv run wandb login

# Remove caches, coverage data and build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

# Remove the virtual environment too (forces a full resync next time)
nuke: clean
    rm -rf .venv
