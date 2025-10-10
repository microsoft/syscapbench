# Code Formatting
 
This doc lists some suggested code checks before committing code. We recommend running these checks to ensure code quality and consistency across the project.

## Suggested Checks before Committing

### 1. Linting

```bash
# Install ruff
curl -LsSf https://astral.sh/ruff/install.sh | sh

# Run ruff
ruff check --fix  # Lint all files in the current directory.
ruff format       # Format all files in the current directory.
```

### 2. Spelling

1. Misspell

```bash
# Install misspell
curl -L https://git.io/misspell | sh -s -- -b "$HOME/.local/bin"

# check all files and overwrite file with corrections
$HOME/.local/bin/misspell -w *   
```

### 3. Markdown linting

```bash
# Install markdownlint-cli2
npm install markdownlint-cli2 --global
markdownlint-cli2 "**/*.md" "#node_modules" "#bin" "#venv" "#.venv"
```

### 4. Unit tests

```bash
pytest

```
