#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export SOURCE_MODE=browser_upload
if [ ! -f .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
