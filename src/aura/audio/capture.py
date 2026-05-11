import os
import wave

import numpy as np
import pyaudio
import webrtcvad
from PyQt6.QtCore import QThread, pyqtSignal

from aura.audio.denoise import reduce_noise_safely
from aura.config import CHUNK_MS, CHUNK_SIZE, SAMPLE_RATE, VAD_LEVEL
from aura.system.native_audio import no_alsa_err, suppress_native_stderr


class AudioRecorderThread(QThread):
    waveform_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(str)

    def __init__(self, filename, transcriber_thread, enable_denoise=False):
        super().__init__()
        self.filename = filename
        self.transcriber = transcriber_thread
        self.enable_denoise = enable_denoise
        self.running = True
        self.vad = webrtcvad.Vad(VAD_LEVEL)
        self.full_frames = []
        self.min_speech_len_sec = 0.5
        self.max_segment_len_sec = 8.0
        self.energy_gate_rms = 550.0

    def _flush_speech_buffer(self, speech_buffer):
        if not speech_buffer:
            return []

        audio_np = np.concatenate(speech_buffer).flatten().astype(np.float32) / 32768.0

        if self.enable_denoise:
            try:
                audio_np = reduce_noise_safely(audio_np, SAMPLE_RATE)
            except Exception as e:
                print(f"Denoising failed (skipped): {e}")

        padding_length = int(SAMPLE_RATE * 0.5)
        silence_padding = np.zeros(padding_length, dtype=np.float32)
        padded_audio_np = np.concatenate([audio_np, silence_padding])
        self.transcriber.add_audio(padded_audio_np)
        return []

    def _open_stream(self):
        with no_alsa_err(), suppress_native_stderr():
            pa = pyaudio.PyAudio()
            target_device_index = None
            target_channels = 1

            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if "pulse" in info["name"].lower():
                    target_device_index = i
                    target_channels = int(info["maxInputChannels"]) if info["maxInputChannels"] > 0 else 1
                    break

            if target_device_index is not None:
                print(f"🚀 Mounting PulseAudio virtual device: Index {target_device_index}, Channels: {target_channels}")
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=target_channels,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=target_device_index,
                    frames_per_buffer=CHUNK_SIZE,
                )
            else:
                print("⚠️ Pulse device not found, trying system default device...")
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )
                target_channels = 1

        return pa, stream, target_channels

    def run(self):
        try:
            pa, stream, target_channels = self._open_stream()
        except Exception as e:
            self.finished_signal.emit(f"Hardware mounting failed: {str(e)}")
            return

        silence_frames = 0
        speech_buffer = []
        min_silence_frames = int((1000 / CHUNK_MS) * self.min_speech_len_sec)
        max_speech_frames = int((1000 / CHUNK_MS) * self.max_segment_len_sec)

        while self.running:
            try:
                raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                np_data = np.frombuffer(raw_data, dtype=np.int16)

                if target_channels > 1:
                    np_data = np_data.reshape(-1, target_channels).mean(axis=1).astype(np.int16)
                vad_data = np_data.tobytes()

                self.waveform_signal.emit(np_data)

                is_speech = self.vad.is_speech(vad_data, SAMPLE_RATE)
                if not is_speech:
                    frame_rms = float(np.sqrt(np.mean(np_data.astype(np.float32) ** 2)))
                    if frame_rms >= self.energy_gate_rms:
                        is_speech = True

                if is_speech:
                    self.full_frames.append(vad_data)
                    speech_buffer.append(np_data)
                    silence_frames = 0
                else:
                    self.full_frames.append(vad_data)
                    silence_frames += 1

                reached_silence_boundary = len(speech_buffer) > 0 and silence_frames > min_silence_frames
                reached_max_segment = len(speech_buffer) >= max_speech_frames
                if reached_silence_boundary or reached_max_segment:
                    speech_buffer = self._flush_speech_buffer(speech_buffer)
                    silence_frames = 0
            except Exception as e:
                print(f"Audio loop error: {e}")
                break

        speech_buffer = self._flush_speech_buffer(speech_buffer)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not self.full_frames:
            self.finished_signal.emit("No audio recorded")
            return

        wav_path = self.filename + ".wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self.full_frames))
        self.finished_signal.emit(wav_path)
