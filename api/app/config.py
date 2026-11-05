import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'guitar.db'}")
REDIS_URL = os.environ.get("REDIS_URL")  # unset => in-process worker thread (dev)
SECRET = os.environ.get("UPLOAD_SECRET", "dev-secret-change-me")
PUBLIC_API_BASE = os.environ.get("PUBLIC_API_BASE", "")  # e.g. https://api.example.com; "" = relative
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

DATA_DIR.mkdir(parents=True, exist_ok=True)
