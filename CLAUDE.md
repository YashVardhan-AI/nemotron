For changes involving html files, please use MCP pupeeteer to test.

Package Management
- ONLY use uv, NEVER pip
- Installation: uv add package
- Running tools: uv run tool
- Upgrading: uv add --dev package --upgrade-package package
- FORBIDDEN: uv pip install, @latest syntax

Formatting
- Format: uv run --frozen ruff format *.py
- Check: uv run --frozen ruff check *.py
- Fix: uv run --frozen ruff check *.py --fix
- Sort imports: uv run --frozen ruff check --select I *.py --fix
- Type checking: uv run --frozen mypy *.py

Version Control
- Use git at each stage: commit after completing each step/task, not in one big batch
- Make a checkpoint commit before destructive or hard-to-reverse changes (deletions, bulk edits)
- One logical change per commit, with a clear message describing what and why
- Verify (tests/format pass) before committing

