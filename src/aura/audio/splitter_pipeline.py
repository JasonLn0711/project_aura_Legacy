import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo

from aura.config import SUPPORTED_SPLIT_EXTENSIONS


@dataclass(frozen=True)
class SplitterSettings:
    target_minutes: float = 40
    tolerance_minutes: float = 5
    min_silence_len: int = 1000
    silence_margin_db: int = 16
    fallback_fade_ms: int = 100

    @property
    def target_ms(self) -> int:
        return int(round(self.target_minutes * 60 * 1000))

    @property
    def tolerance_ms(self) -> int:
        return int(round(self.tolerance_minutes * 60 * 1000))


@dataclass(frozen=True)
class SplitDecision:
    cut_point: int
    used_silence: bool
    search_start: int
    search_end: int


@dataclass(frozen=True)
class SplitResult:
    output_paths: list[Path]


def output_extension(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    return ext if ext in SUPPORTED_SPLIT_EXTENSIONS else "mp3"


def source_bitrate(file_path: str):
    try:
        info = mediainfo(file_path)
        return info.get("bit_rate", None)
    except Exception:
        return None


def export_format_for_ext(ext: str) -> str:
    if ext == "m4a":
        return "ipod"
    if ext == "aac":
        return "adts"
    return ext


def choose_cut_point(audio: AudioSegment, current_pos: int, settings: SplitterSettings) -> SplitDecision:
    total_duration_ms = len(audio)
    search_start = max(current_pos, current_pos + settings.target_ms - settings.tolerance_ms)
    search_end = min(current_pos + settings.target_ms + settings.tolerance_ms, total_duration_ms)
    window = audio[search_start:search_end]
    silences = detect_silence(
        window,
        min_silence_len=settings.min_silence_len,
        silence_thresh=audio.dBFS - settings.silence_margin_db,
    )

    if silences:
        best_silence = silences[len(silences) // 2]
        cut_point = search_start + (best_silence[0] + best_silence[1]) // 2
        return SplitDecision(cut_point=cut_point, used_silence=True, search_start=search_start, search_end=search_end)

    return SplitDecision(
        cut_point=min(current_pos + settings.target_ms, total_duration_ms),
        used_silence=False,
        search_start=search_start,
        search_end=search_end,
    )


def export_chunk(chunk: AudioSegment, output_dir: str, base_name: str, index: int, ext: str, bitrate=None) -> Path:
    out_path = Path(output_dir) / f"{base_name}_part{index:02d}.{ext}"
    export_kwargs = {"format": export_format_for_ext(ext)}
    if ext == "mp3":
        export_kwargs["bitrate"] = str(bitrate) if bitrate else "192k"
    with out_path.open("wb") as target:
        chunk.export(target, **export_kwargs)
    return out_path


def split_audio_file(
    file_path: str,
    output_dir: str,
    settings: SplitterSettings,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> SplitResult:
    if settings.target_ms <= 0:
        raise ValueError("target_minutes must be greater than zero")
    if settings.tolerance_ms < 0:
        raise ValueError("tolerance_minutes must be zero or greater")

    audio = None
    output_paths = []
    try:
        if log_callback:
            log_callback(f"⏳ Loading audio file...\nFile: {file_path}")
        if progress_callback:
            progress_callback(5)

        with open(file_path, "rb") as source:
            audio = AudioSegment.from_file(source)
        total_duration_ms = len(audio)
        if log_callback:
            log_callback(f"✅ Load successful! Total duration: {total_duration_ms / 60000:.2f} minutes")

        original_bitrate = source_bitrate(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = output_extension(file_path)
        current_pos = 0
        chunk_index = 1

        while current_pos < total_duration_ms:
            if (total_duration_ms - current_pos) <= (settings.target_ms + settings.tolerance_ms):
                if log_callback:
                    log_callback(f"✂️ Exporting final segment (Part {chunk_index})...")
                final_chunk = audio[current_pos:]
                output_paths.append(export_chunk(final_chunk, output_dir, base_name, chunk_index, ext, original_bitrate))
                if progress_callback:
                    progress_callback(100)
                break

            decision = choose_cut_point(audio, current_pos, settings)
            if log_callback:
                log_callback(
                    f"🔍 Analyzing best cut point for Part {chunk_index} "
                    f"({decision.search_start / 60000:.1f} ~ {decision.search_end / 60000:.1f} min)..."
                )

            if decision.used_silence:
                if log_callback:
                    log_callback(f"🎯 Found suitable pause point! Cutting at {decision.cut_point / 60000:.2f} minutes.")
                chunk = audio[current_pos:decision.cut_point]
            else:
                if log_callback:
                    log_callback(
                        f"⚠️ No obvious pause found, performing smooth cut at {decision.cut_point / 60000:.2f} minutes."
                    )
                chunk = audio[current_pos:decision.cut_point].fade_out(settings.fallback_fade_ms)

            if log_callback:
                log_callback(f"💾 Saving Part {chunk_index}...")
            output_paths.append(export_chunk(chunk, output_dir, base_name, chunk_index, ext, original_bitrate))

            current_pos = decision.cut_point
            chunk_index += 1
            if progress_callback:
                progress_callback(int((current_pos / total_duration_ms) * 90) + 5)

        if log_callback:
            log_callback("🎉 All segments have been split and saved.")
        return SplitResult(output_paths=output_paths)
    finally:
        if audio:
            del audio
        gc.collect()
