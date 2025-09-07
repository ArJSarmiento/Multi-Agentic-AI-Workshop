import os

# Block accidental live model calls in tests (local only)
os.environ.setdefault("ALLOW_MODEL_REQUESTS", "false")
