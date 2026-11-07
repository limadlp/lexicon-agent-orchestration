"""Explicit configuration loading for command-line entry points."""

import os
from pathlib import Path


def load_dotenv(env_path: Path | str | None = None) -> None:
    """Carrega variaveis de um arquivo .env para os.environ sem bibliotecas externas."""
    candidates = [
        Path(env_path) if env_path else None,
        Path.cwd() / ".env",
        Path.cwd() / "lexicon" / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",  # lexicon/.env
        Path(__file__).resolve().parent.parent.parent.parent / ".env",  # root/.env
    ]
    for path in filter(None, candidates):
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line in lines:
                line = line.strip()
                key, separator, value = line.partition("=")
                if separator and key.strip() and not line.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            return
