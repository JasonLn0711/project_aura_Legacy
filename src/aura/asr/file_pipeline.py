import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from aura.audio.denoise import OFF_DENOISE_PRESET, normalize_denoise_preset, reduce_audio_segment_noise
from aura.settings import DEFAULT_SETTINGS
from aura.system.cuda import is_cuda_runtime_error
from aura.system.runtime_paths import append_transcript_backup, temp_normalized_path


class FileTranscriptionCancelled(RuntimeError):
    pass


@dataclass
class CancellationToken:
    cancelled: bool = False

    def request_cancel(self):
        self.cancelled = True

    def raise_if_cancelled(self):
        if self.cancelled:
            raise FileTranscriptionCancelled("File transcription cancelled.")


@dataclass(frozen=True)
class FileTranscriptionSettings:
    target_dbfs: float = DEFAULT_SETTINGS.target_dbfs
    beam_size: int = DEFAULT_SETTINGS.beam_size
    initial_prompt: str | None = DEFAULT_SETTINGS.file_initial_prompt
    language: str | None = DEFAULT_SETTINGS.language
    enable_denoise: bool = DEFAULT_SETTINGS.denoise_enabled
    denoise_preset: str = DEFAULT_SETTINGS.denoise_preset

    def __post_init__(self):
        object.__setattr__(
            self,
            "denoise_preset",
            normalize_denoise_preset(self.enable_denoise, self.denoise_preset),
        )


@dataclass(frozen=True)
class FileTranscriptionResult:
    file_name: str
    lines: list[str] = field(default_factory=list)
    cancelled: bool = False


def resolve_initial_prompt(prompt, default_prompt=DEFAULT_SETTINGS.file_initial_prompt):
    """Use the default prompt only when the caller did not provide a value."""
    if prompt is None:
        return default_prompt
    return str(prompt).strip()


def build_transcribe_kwargs(beam_size=5, language="zh", initial_prompt=None, condition_on_previous_text=True):
    kwargs = {
        "beam_size": int(beam_size) if beam_size else 5,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    return kwargs


def format_segment(segment) -> str:
    h, m = divmod(int(segment.start), 3600)
    m, s = divmod(m, 60)
    timestamp = f"{h:02d}:{m:02d}:{s:02d}"
    return f"[{timestamp}] {segment.text}"


def normalize_file_transcription_error(error: Exception) -> str:
    error_msg = str(error)
    lower_msg = error_msg.lower()
    if is_cuda_runtime_error(error_msg):
        return (
            "CUDA runtime is incomplete on this machine.\n"
            "The app tried to use the GPU model, but a CUDA runtime library is missing.\n\n"
            "Quick fix: open Advanced Settings, choose `int8`, click Reload Model, "
            "then import the file again."
        )
    if "ffmpeg" in lower_msg or "ffprobe" in lower_msg:
        return (
            f"{error_msg}\n\nImported media decoding depends on ffmpeg/ffprobe. "
            "Please install them and try again."
        )
    return error_msg


def prepare_import_audio(
    file_path: str,
    settings: FileTranscriptionSettings,
    temp_path: Path,
    cancellation: CancellationToken | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> Path:
    cancellation = cancellation or CancellationToken()
    file_name = os.path.basename(file_path)
    audio = None
    normalized = None
    try:
        if status_callback:
            status_callback(f"🔊 Preparing {file_name} for transcription...")
        cancellation.raise_if_cancelled()
        with open(file_path, "rb") as source:
            audio = AudioSegment.from_file(source)

        if settings.denoise_preset != OFF_DENOISE_PRESET:
            if status_callback:
                status_callback(f"🧹 Applying {settings.denoise_preset} denoise to {file_name}...")
            cancellation.raise_if_cancelled()
            audio = reduce_audio_segment_noise(audio, preset=settings.denoise_preset)

        if status_callback:
            status_callback(f"🔉 Normalizing volume for {file_name}...")
        cancellation.raise_if_cancelled()
        normalized = audio.apply_gain(settings.target_dbfs - audio.dBFS)
        with temp_path.open("wb") as target:
            normalized.export(target, format="wav")
        cancellation.raise_if_cancelled()
        return temp_path
    finally:
        if audio:
            del audio
        if normalized:
            del normalized
        gc.collect()


def transcribe_prepared_file(model, prepared_path: Path, settings: FileTranscriptionSettings):
    return model.transcribe(
        str(prepared_path),
        **build_transcribe_kwargs(
            beam_size=settings.beam_size,
            language=settings.language,
            initial_prompt=settings.initial_prompt,
            condition_on_previous_text=True,
        ),
    )


def transcribe_file(
    model,
    file_path: str,
    settings: FileTranscriptionSettings,
    worker_id,
    cancellation: CancellationToken | None = None,
    status_callback: Callable[[str], None] | None = None,
    line_callback: Callable[[str], None] | None = None,
) -> FileTranscriptionResult:
    cancellation = cancellation or CancellationToken()
    file_name = os.path.basename(file_path)
    temp_path = temp_normalized_path(worker_id)
    lines = []
    try:
        prepared_path = prepare_import_audio(
            file_path=file_path,
            settings=settings,
            temp_path=temp_path,
            cancellation=cancellation,
            status_callback=status_callback,
        )
        segments, info = transcribe_prepared_file(model, prepared_path, settings)

        for segment in segments:
            cancellation.raise_if_cancelled()
            formatted_text = format_segment(segment)
            lines.append(formatted_text)
            if line_callback:
                line_callback(formatted_text)
            append_transcript_backup(formatted_text)

        if status_callback:
            if cancellation.cancelled:
                status_callback(f"⚠️ Cancelled transcribing {file_name}")
            else:
                status_callback(f"✅ Finished transcribing {file_name}")
        return FileTranscriptionResult(file_name=file_name, lines=lines, cancelled=cancellation.cancelled)
    finally:
        if temp_path.exists():
            temp_path.unlink()
