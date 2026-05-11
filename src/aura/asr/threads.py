import datetime
import gc
import os
import queue
import time

from faster_whisper import WhisperModel
from pydub import AudioSegment
from PyQt6.QtCore import QThread, pyqtSignal

from aura.config import COMPUTE_TYPE, DEFAULT_LIVE_PROMPT, DEVICE, MODEL_ID
from aura.system.cuda import is_cuda_runtime_error, preload_cuda_runtime_libraries


class FileTranscriberThread(QThread):
    text_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model, file_path, target_dbfs=-20.0, beam_size=5, initial_prompt="", language="zh"):
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.target_dbfs = target_dbfs
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.language = language

    def run(self):
        self.status_updated.emit("⏳ Analyzing audio file, please wait...")
        temp_path = None
        audio = None
        normalized = None
        try:
            self.status_updated.emit("🔊 Analyzing audio volume...")
            audio = AudioSegment.from_file(self.file_path)
            normalized = audio.apply_gain(self.target_dbfs - audio.dBFS)
            temp_path = os.path.join(os.getcwd(), "temp_normalized.wav")
            normalized.export(temp_path, format="wav")

            segments, info = self.model.transcribe(
                temp_path,
                beam_size=self.beam_size,
                language=self.language,
                condition_on_previous_text=True,
                initial_prompt=self.initial_prompt,
            )

            for segment in segments:
                h, m = divmod(int(segment.start), 3600)
                m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                formatted_text = f"[{timestamp}] {segment.text}"
                self.text_updated.emit(formatted_text)

                with open("temp_transcript.txt", "a", encoding="utf-8") as f:
                    f.write(formatted_text + "\n")
            self.status_updated.emit("✅ File processing completed!")
        except Exception as e:
            self.status_updated.emit(f"❌ File processing failed: {e}")
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

    def update_live_settings(self, beam_size=5, language="zh", initial_prompt=""):
        self.live_beam_size = int(beam_size) if beam_size else 5
        self.live_language = language
        self.live_initial_prompt = initial_prompt.strip() if initial_prompt else ""

    def run(self):
        while self.running:
            try:
                if self.model is None:
                    time.sleep(0.5)
                    continue

                audio_data = self.audio_queue.get(timeout=1)
                transcribe_kwargs = {
                    "beam_size": self.live_beam_size,
                    "condition_on_previous_text": False,
                }
                if self.live_language:
                    transcribe_kwargs["language"] = self.live_language
                if self.live_initial_prompt:
                    transcribe_kwargs["initial_prompt"] = self.live_initial_prompt

                segments, info = self.model.transcribe(audio_data, **transcribe_kwargs)
                text_segment = "".join([s.text for s in segments])
                if text_segment.strip():
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    formatted_text = f"[{timestamp}] {text_segment}"
                    self.text_updated.emit(formatted_text)
                    with open("temp_transcript.txt", "a", encoding="utf-8") as f:
                        f.write(formatted_text + "\n")
            except queue.Empty:
                continue
            except Exception as e:
                err = f"Live transcription error: {e}"
                print(err)
                self.status_updated.emit(f"⚠️ {err}")

    def add_audio(self, audio_np):
        self.audio_queue.put(audio_np)

    def stop(self):
        self.running = False
