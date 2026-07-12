#!/usr/bin/env bash
set -e
uv run ruff format .
uv run ruff check .
uv run pytest
docker build -q -t agentic-rag:ci .
echo "✅ all gates green — safe to push"