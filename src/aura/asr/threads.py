import datetime
import gc
import logging
import os
import queue
import time

from faster_whisper import WhisperModel
from pydub import AudioSegment
from PyQt6.QtCore import QThread, pyqtSignal

from aura.audio.denoise import reduce_audio_segment_noise
from aura.config import COMPUTE_TYPE, DEFAULT_LIVE_PROMPT, DEFAULT_PROMPT, DEVICE, MODEL_ID
from aura.system.cuda import is_cuda_runtime_error, preload_cuda_runtime_libraries
from aura.system.runtime_paths import append_transcript_backup, temp_normalized_path

logger = logging.getLogger(__name__)


def resolve_initial_prompt(prompt, default_prompt):
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


class FileTranscriberThread(QThread):
    text_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        model,
        file_path,
        target_dbfs=-20.0,
        beam_size=5,
        initial_prompt=None,
        language="zh",
        enable_denoise=False,
    ):
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.target_dbfs = target_dbfs
        self.beam_size = beam_size
        self.initial_prompt = resolve_initial_prompt(initial_prompt, DEFAULT_PROMPT)
        self.language = language
        self.enable_denoise = enable_denoise
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def _raise_if_cancelled(self):
        if self.cancel_requested:
            raise RuntimeError("File transcription cancelled.")

    def run(self):
        self.status_updated.emit("⏳ Analyzing audio file, please wait...")
        temp_path = None
        audio = None
        normalized = None
        file_name = os.path.basename(self.file_path)
        try:
            self.status_updated.emit(f"🔊 Preparing {file_name} for transcription...")
            self._raise_if_cancelled()
            audio = AudioSegment.from_file(self.file_path)
            if self.enable_denoise:
                self.status_updated.emit(f"🧹 Applying denoise to {file_name}...")
                self._raise_if_cancelled()
                audio = reduce_audio_segment_noise(audio)

            self.status_updated.emit(f"🔉 Normalizing volume for {file_name}...")
            self._raise_if_cancelled()
            normalized = audio.apply_gain(self.target_dbfs - audio.dBFS)
            temp_path = temp_normalized_path(id(self))
            normalized.export(str(temp_path), format="wav")

            self._raise_if_cancelled()
            segments, info = self.model.transcribe(
                str(temp_path),
                **build_transcribe_kwargs(
                    beam_size=self.beam_size,
                    language=self.language,
                    initial_prompt=self.initial_prompt,
                    condition_on_previous_text=True,
                ),
            )

            for segment in segments:
                h, m = divmod(int(segment.start), 3600)
                m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                formatted_text = f"[{timestamp}] {segment.text}"
                self.text_updated.emit(formatted_text)
                append_transcript_backup(formatted_text)
            if self.cancel_requested:
                self.status_updated.emit(f"⚠️ Cancelled transcribing {file_name}")
            else:
                self.status_updated.emit(f"✅ Finished transcribing {file_name}")
        except Exception as e:
            error_msg = str(e)
            lower_msg = error_msg.lower()
            if is_cuda_runtime_error(error_msg):
                error_msg = (
                    "CUDA runtime is incomplete on this machine.\n"
                    "The app tried to use the GPU model, but a CUDA runtime library is missing.\n\n"
                    "Quick fix: open Advanced Settings, choose `int8`, click Reload Model, "
                    "then import the file again."
                )
            elif "ffmpeg" in lower_msg or "ffprobe" in lower_msg:
                error_msg = (
                    f"{error_msg}\n\nImported media decoding depends on ffmpeg/ffprobe. "
                    "Please install them and try again."
                )
            if self.cancel_requested:
                self.status_updated.emit(f"⚠️ Cancelled transcribing {file_name}")
            else:
                self.status_updated.emit(f"❌ Failed to transcribe {file_name}")
                self.error_signal.emit(f"{file_name}\n\n{error_msg}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            if audio:
                del audio
            if normalized:
                del normalized
            gc.collect()
            self.finished_signal.emit()


class ModelLoaderThread(QThread):
    """Load the Whisper model asynchronously to avoid UI freezes."""

    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, device, compute_type):
        super().__init__()
        self.device = device
        self.compute_type = compute_type
        self.actual_device = device
        self.actual_compute_type = compute_type
        self.runtime_note = ""

    def run(self):
        try:
            if self.device == "cuda":
                runtime_ready, runtime_source = preload_cuda_runtime_libraries()
                if runtime_ready:
                    self.runtime_note = f"CUDA runtime source: {runtime_source}"
                else:
                    self.actual_device = "cpu"
                    self.actual_compute_type = "int8"
                    self.runtime_note = f"CUDA runtime unavailable ({runtime_source}). Falling back to CPU/int8."
                    self.status_signal.emit(f"⚠️ {self.runtime_note}")

            self.status_signal.emit(
                f"🚀 Loading model in background ({self.actual_device}/{self.actual_compute_type})..."
            )
            model = WhisperModel(
                MODEL_ID,
                device=self.actual_device,
                compute_type=self.actual_compute_type,
            )
            self.finished_signal.emit(model)
        except Exception as e:
            error_msg = str(e)
            if self.actual_device == "cuda" and is_cuda_runtime_error(error_msg):
                try:
                    self.actual_device = "cpu"
                    self.actual_compute_type = "int8"
                    self.runtime_note = "CUDA runtime failed during initialization. Retrying on CPU/int8."
                    self.status_signal.emit(f"⚠️ {self.runtime_note}")
                    model = WhisperModel(MODEL_ID, device="cpu", compute_type="int8")
                    self.finished_signal.emit(model)
                    return
                except Exception as fallback_error:
                    error_msg = f"{error_msg} | CPU fallback failed: {fallback_error}"
            if "out of memory" in error_msg.lower():
                error_msg = "Insufficient GPU memory. Try switching to int8 precision or closing other programs."
            self.error_signal.emit(error_msg)


class TranscriberThread(QThread):
    text_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.running = True
        self.model = None
        self.device = DEVICE
        self.compute_type = COMPUTE_TYPE
        self.live_beam_size = 5
        self.live_language = "zh"
        self.live_initial_prompt = DEFAULT_LIVE_PROMPT

    def update_live_settings(self, beam_size=5, language="zh", initial_prompt=None):
        self.live_beam_size = int(beam_size) if beam_size else 5
        self.live_language = language
        self.live_initial_prompt = resolve_initial_prompt(initial_prompt, DEFAULT_LIVE_PROMPT)

    def run(self):
        while self.running:
            try:
                if self.model is None:
                    time.sleep(0.5)
                    continue

                audio_data = self.audio_queue.get(timeout=1)
                transcribe_kwargs = build_transcribe_kwargs(
                    beam_size=self.live_beam_size,
                    language=self.live_language,
                    initial_prompt=self.live_initial_prompt,
                    condition_on_previous_text=False,
                )

                segments, info = self.model.transcribe(audio_data, **transcribe_kwargs)
                text_segment = "".join([s.text for s in segments])
                if text_segment.strip():
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    formatted_text = f"[{timestamp}] {text_segment}"
                    self.text_updated.emit(formatted_text)
                    append_transcript_backup(formatted_text)
            except queue.Empty:
                continue
            except Exception as e:
                err = f"Live transcription error: {e}"
                logger.exception(err)
                self.status_updated.emit(f"⚠️ {err}")

    def add_audio(self, audio_np):
        self.audio_queue.put(audio_np)

    def stop(self):
        self.running = False
