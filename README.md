
# Project AURA: Ultimate Audio Assistant (全方位錄音助理)

![CI Status](https://img.shields.io/badge/Status-Prototype-informational?logo=github) ![Python Version](https://img.shields.io/badge/Python-3.12.3-blue?logo=python) ![ASR Engine](https://img.shields.io/badge/ASR-faster--whisper-orange) ![UI](https://img.shields.io/badge/UI-PyQt6-9cf) ![VAD](https://img.shields.io/badge/VAD-WebRTC_VAD-success)

## Executive Summary & Project Metadata

**Project AURA** is a dual-core desktop assistant that integrates:

1) **Real-time / file-based transcription** (ASR) with timestamped logs  
2) **Smart audio splitting** that finds natural pause points to avoid cutting speech mid-sentence

This project is designed for **Traditional Chinese meeting notes**, with prompt-guided punctuation and automatic transcript backup.

- **Project Name**: Project AURA (Ultimate Audio Assistant)
- **Academic Affiliation**: National Yang Ming Chiao Tung University (NYCU)
- **Project Lead**: Jason Chia-Sheng Lin (PhD. Student)
- **Release Version**: `2.0.1`
- **Release Date**: `2026-02-23`
- **Versioning Strategy**: Semantic Versioning (MAJOR / MINOR / PATCH)

---

## Feature Implementation Checklist

| Feature Category | Implementation Details |
| :--- | :--- |
| **Real-time Transcription** | Live microphone recording + streaming ASR via `faster-whisper` (GPU supported). |
| **File Transcription** | Import common audio/video formats and export a timestamped transcript. |
| **Traditional Chinese Prompting** | Uses an `initial_prompt` to encourage full-width punctuation for professional notes. |
| **Automatic Backup** | Appends transcript lines to `temp_transcript.txt` during processing. |
| **Voice Activity Detection (VAD)** | WebRTC VAD-based speech segmentation with configurable aggressiveness. |
| **Smart Splitting** | Uses silence detection to cut near natural pauses within target length windows. |
| **Modern Desktop UI** | PyQt6 tabs + waveform visualization via `pyqtgraph` + themed UI (`qt-material`). |
| **Linux Audio Robustness** | PulseAudio device probing + ALSA error handler suppression for cleaner logs. |

---

## 1. Project Introduction

Taking meeting notes from long recordings is expensive and error-prone. **Project AURA** turns raw audio into structured artifacts:

- **Transcription output**: timestamped text stream suitable for meeting minutes
- **Split output**: multiple smaller audio segments suitable for review, annotation, or downstream ASR

AURA provides two UI tabs:

- **📝 Recording & Transcription**: record mic audio, view live waveform, generate transcript, save `.txt`
- **✂️ Smart Splitter**: choose file + output folder, pick target length & tolerance, then export segmented chunks

---

## 2. Repository Structure

```text
project_aura/
├── audio_assistant_v1.2.py   # Main application (UI + ASR + VAD + splitter)
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

> Note: Large audio outputs and temporary transcripts are intentionally excluded from Git tracking.

---

## 3. Environment Requirements

### 3.1 Recommended Runtime

* **OS**: Ubuntu 22.04 / 24.04 (Desktop)
* **Python**: 3.10+ (recommended 3.11)
* **GPU (Optional but recommended)**: NVIDIA CUDA-capable GPU for faster ASR

### 3.2 Core Python Dependencies

This project uses:

* `faster-whisper` (ASR)
* `pyaudio` + PulseAudio / ALSA (audio input)
* `webrtcvad` (speech detection)
* `pydub` (audio loading/export)
* `pyqt6`, `pyqtgraph`, `qt-material` (UI)

Install via `requirements.txt` (see below).

---

## 4. Installation Steps

### 4.1 System Packages (Ubuntu)

```bash
sudo apt-get update

# Audio I/O (PyAudio)
sudo apt-get install -y portaudio19-dev python3-dev

# ffmpeg is required for pydub to read many formats (mp3/mp4/m4a, etc.)
sudo apt-get install -y ffmpeg
```

> If your mic devices behave oddly, confirm PulseAudio / PipeWire is working on your Ubuntu desktop.

### 4.2 Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Execution Commands

### 5.1 Run the App

```bash
python audio_assistant_v1.2.py
```

### 5.2 Expected UI Workflow

**Tab 1 — Recording & Transcription**

1. Wait for GPU model initialization (status label updates)
2. Click **🎙️ Start Recording**
3. Speak normally; waveform updates live
4. Click **🛑 Stop Recording**
5. Save transcript via **💾 Save transcript (.txt)**

**Tab 2 — Smart Splitter**

1. Select input file (mp3/wav/m4a/mp4/flac)
2. Select output folder
3. Set target length (minutes) + tolerance
4. Click **Start** to export `*_part01`, `*_part02`, ...

---

## 6. Configuration (Key Defaults)

* **Sample Rate**: `16000`
* **Chunk Size**: `30ms`
* **VAD Level**: `3` (aggressive)
* **Model**: `SoybeanMilk/faster-whisper-Breeze-ASR-25`
* **Device**: `cuda` (change to `cpu` if no GPU)
* **Compute Type**: `float16`

> If you do not have a GPU, set `DEVICE="cpu"` and consider `COMPUTE_TYPE="int8"` depending on your environment.

---

## 7. Output Artifacts

During runtime, AURA may create:

* `temp_transcript.txt` (auto-backup transcript stream)
* `*.wav` + normalized `*.mp3` for recordings
* split audio chunks: `{basename}_part01.{ext}`, `{basename}_part02.{ext}`, ...

---

## 8. Troubleshooting

### 8.1 “Invalid sample rate” / mic device issues (Linux)

* AURA tries to prioritize **PulseAudio** devices for automatic resampling
* If it fails, confirm your system microphone works in system settings and that PulseAudio/PipeWire is running

### 8.2 `pyaudio` install fails

Make sure you installed:

```bash
sudo apt-get install -y portaudio19-dev python3-dev
```

Then reinstall:

```bash
pip install pyaudio
```

### 8.3 Slow transcription on CPU

* Switch to GPU if available
* Reduce model size / adjust compute type

---

## 9. License

© 2026 Jason Chia-Sheng Lin (NYCU).
License TBD (add a LICENSE file if you plan to open-source it publicly).

---

***Project Lead**: Jason Chia-Sheng Lin | National Yang Ming Chiao Tung University (NYCU)*