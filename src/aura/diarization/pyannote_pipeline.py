import os
from dataclasses import dataclass
from pathlib import Path

from aura.diarization.speaker_assignment import SpeakerTurn


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
HUGGINGFACE_TOKEN_ENV = "HUGGINGFACE_TOKEN"
HF_TOKEN_ENV = "HF_TOKEN"


class DiarizationDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiarizationSettings:
    enabled: bool = False
    min_speakers: int = 2
    max_speakers: int = 6
    model_id: str = DEFAULT_DIARIZATION_MODEL
    device: str = "cuda"
    use_exclusive: bool = True

    def __post_init__(self):
        if self.min_speakers < 1:
            raise ValueError("min_speakers must be at least 1")
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers must be greater than or equal to min_speakers")


def huggingface_token() -> str | None:
    return os.environ.get(HUGGINGFACE_TOKEN_ENV) or os.environ.get(HF_TOKEN_ENV)


def pipeline_kwargs(settings: DiarizationSettings) -> dict:
    if settings.min_speakers == settings.max_speakers:
        return {"num_speakers": settings.min_speakers}
    return {
        "min_speakers": settings.min_speakers,
        "max_speakers": settings.max_speakers,
    }


def _load_pyannote_pipeline(settings: DiarizationSettings):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationDependencyError(
            "Speaker diarization requires the optional `pyannote.audio` dependency. "
            "Install it with `python -m pip install -e .[diarization]`."
        ) from exc

    token = huggingface_token()
    if not token:
        raise DiarizationDependencyError(
            "Speaker diarization requires a Hugging Face access token. "
            f"Set `{HUGGINGFACE_TOKEN_ENV}` or `{HF_TOKEN_ENV}` after accepting the pyannote model terms."
        )

    pipeline = Pipeline.from_pretrained(settings.model_id, token=token)

    if settings.device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except ImportError:
            pass

    return pipeline


def _annotation_from_output(output, use_exclusive: bool):
    if use_exclusive and hasattr(output, "exclusive_speaker_diarization"):
        return output.exclusive_speaker_diarization
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output


def speaker_turns_from_annotation(annotation) -> list[SpeakerTurn]:
    turns = []

    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for segment, _track, speaker in iterator:
            turns.append(SpeakerTurn(float(segment.start), float(segment.end), str(speaker)))
        return turns

    for turn, speaker in annotation:
        turns.append(SpeakerTurn(float(turn.start), float(turn.end), str(speaker)))
    return turns


def diarize_audio_file(audio_path: str | Path, settings: DiarizationSettings) -> list[SpeakerTurn]:
    pipeline = _load_pyannote_pipeline(settings)
    output = pipeline(str(audio_path), **pipeline_kwargs(settings))
    annotation = _annotation_from_output(output, settings.use_exclusive)
    return speaker_turns_from_annotation(annotation)
