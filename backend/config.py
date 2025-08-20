# backend/config.py

import os

# Define directories for uploads and vector storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "..", "data", "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "..", "data", "vectors")

def ensure_dirs():
    """Ensure necessary directories exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
