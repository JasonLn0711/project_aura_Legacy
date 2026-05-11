SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
VAD_LEVEL = 3

MODEL_ID = "SoybeanMilk/faster-whisper-Breeze-ASR-25"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
GITHUB_REPOSITORY = "JasonLn0711/project_aura"

DEFAULT_PROMPT = "這是一份專業的繁體中文會議紀錄，請務必根據語氣加上正確的全形標點符號。"
DEFAULT_LIVE_PROMPT = "The following is a professional meeting record."

SUPPORTED_SPLIT_EXTENSIONS = {
    "mp3",
    "wav",
    "m4a",
    "ogg",
    "flac",
    "aac",
    "wma",
    "aiff",
    "opus",
}
