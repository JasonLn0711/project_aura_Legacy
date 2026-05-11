import gc
import os

from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from PyQt6.QtCore import QThread, pyqtSignal

from aura.config import SUPPORTED_SPLIT_EXTENSIONS


class SmartSplitterThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, output_dir, target_minutes=40, tolerance_minutes=5):
        super().__init__()
        self.file_path = file_path
        self.output_dir = output_dir
        self.target_ms = target_minutes * 60 * 1000
        self.tolerance_ms = tolerance_minutes * 60 * 1000

    def run(self):
        audio = None
        try:
            self.log_signal.emit(f"⏳ Loading audio file...\nFile: {self.file_path}")
            self.progress_signal.emit(5)

            audio = AudioSegment.from_file(self.file_path)
            total_duration_ms = len(audio)
            self.log_signal.emit(f"✅ Load successful! Total duration: {total_duration_ms / 60000:.2f} minutes")

            try:
                info = mediainfo(self.file_path)
                original_bitrate = info.get("bit_rate", None)
            except Exception:
                original_bitrate = None

            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            ext = os.path.splitext(self.file_path)[1].lower().replace(".", "")
            if ext not in SUPPORTED_SPLIT_EXTENSIONS:
                ext = "mp3"

            current_pos = 0
            chunk_index = 1

            while current_pos < total_duration_ms:
                if (total_duration_ms - current_pos) <= (self.target_ms + self.tolerance_ms):
                    self.log_signal.emit(f"✂️ Exporting final segment (Part {chunk_index})...")
                    final_chunk = audio[current_pos:]
                    self.export_chunk(final_chunk, base_name, chunk_index, ext, original_bitrate)
                    self.progress_signal.emit(100)
                    break

                search_start = current_pos + self.target_ms - self.tolerance_ms
                search_end = min(current_pos + self.target_ms + self.tolerance_ms, total_duration_ms)
                self.log_signal.emit(
                    f"🔍 Analyzing best cut point for Part {chunk_index} "
                    f"({search_start / 60000:.1f} ~ {search_end / 60000:.1f} min)..."
                )

                window = audio[search_start:search_end]
                silences = detect_silence(window, min_silence_len=1000, silence_thresh=audio.dBFS - 16)

                if silences:
                    best_silence = silences[len(silences) // 2]
                    cut_point = search_start + (best_silence[0] + best_silence[1]) // 2
                    self.log_signal.emit(f"🎯 Found suitable pause point! Cutting at {cut_point / 60000:.2f} minutes.")
                else:
                    cut_point = current_pos + self.target_ms
                    self.log_signal.emit(f"⚠️ No obvious pause found, performing smooth cut at {cut_point / 60000:.2f} minutes.")

                chunk = audio[current_pos:cut_point]
                if not silences:
                    chunk = chunk.fade_out(100)

                self.log_signal.emit(f"💾 Saving Part {chunk_index}...")
                self.export_chunk(chunk, base_name, chunk_index, ext, original_bitrate)

                current_pos = cut_point
                chunk_index += 1
                self.progress_signal.emit(int((current_pos / total_duration_ms) * 90) + 5)

            self.log_signal.emit("🎉 All segments have been split and saved.")
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if audio:
                del audio
            gc.collect()

    def export_chunk(self, chunk, base_name, index, ext, bitrate=None):
        out_path = os.path.join(self.output_dir, f"{base_name}_part{index:02d}.{ext}")
        format_str = "ipod" if ext == "m4a" else ("adts" if ext == "aac" else ext)
        export_kwargs = {"format": format_str}
        if ext == "mp3":
            export_kwargs["bitrate"] = str(bitrate) if bitrate else "192k"
        chunk.export(out_path, **export_kwargs)
