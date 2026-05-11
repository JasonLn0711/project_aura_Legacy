import os
import tempfile
from pathlib import Path


RUNTIME_DIR_ENV = "AURA_RUNTIME_DIR"


def runtime_dir() -> Path:
    base_dir = os.environ.get(RUNTIME_DIR_ENV)
    if base_dir:
        path = Path(base_dir)
    else:
        path = Path(tempfile.gettempdir()) / "project_aura"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_normalized_path(worker_id) -> Path:
    return runtime_dir() / f"normalized_{worker_id}.wav"


def transcript_backup_path() -> Path:
    return runtime_dir() / "temp_transcript.txt"


def append_transcript_backup(text: str):
    with transcript_backup_path().open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def remove_transcript_backup():
    path = transcript_backup_path()
    if path.exists():
        path.unlink()
