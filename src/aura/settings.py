from dataclasses import dataclass

from aura.config import COMPUTE_TYPE, DEFAULT_LIVE_PROMPT, DEFAULT_PROMPT, DEVICE, MODEL_ID


@dataclass(frozen=True)
class AppSettings:
    model_id: str = MODEL_ID
    device: str = DEVICE
    compute_type: str = COMPUTE_TYPE
    target_dbfs: float = -20.0
    beam_size: int = 5
    language: str | None = "zh"
    file_initial_prompt: str | None = DEFAULT_PROMPT
    live_initial_prompt: str | None = DEFAULT_LIVE_PROMPT
    denoise_enabled: bool = False
    splitter_target_minutes: int = 40
    splitter_tolerance_minutes: int = 5


DEFAULT_SETTINGS = AppSettings()
