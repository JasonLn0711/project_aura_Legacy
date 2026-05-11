# Project AURA Refactor

Project AURA is a desktop audio assistant for real-time recording, Whisper-based transcription, batch file transcription, and smart audio splitting.

This repository is a clean refactor of the working `audio_assistant_v1.5.0.py` script from `record_audio_ubuntu`. It intentionally does **not** copy the recording archive, `.record/` virtual environment, temporary transcripts, or generated media files.

## Why This Repo Exists

The original working folder had three roles mixed together:

- source code and README
- local Python runtime (`.record/`)
- generated recordings, transcripts, splits, and course/media artifacts

That made git history hard to reason about and hid new version scripts behind a broad `*` ignore rule. This repo separates the maintainable application source from runtime data.

## Current Refactor Scope

The previous monolithic script is now split by responsibility:

```text
src/aura/
├── app.py                    # QApplication entrypoint
├── config.py                 # Runtime constants
├── metadata.py               # Version and project metadata
├── asr/
│   └── threads.py            # Model loader, live ASR, batch file ASR
├── audio/
│   ├── capture.py            # PyAudio/PulseAudio recording thread
│   ├── denoise.py            # Safe noisereduce wrapper
│   └── splitter.py           # Smart audio splitting thread
├── system/
│   ├── cuda.py               # CUDA runtime preload and fallback detection
│   ├── native_audio.py       # ALSA/JACK stderr suppression helpers
│   └── update_checker.py     # Background GitHub release check
└── ui/
    ├── main_window.py
    ├── splitter_tab.py
    └── transcription_tab.py
```

The legacy one-file baseline is retained at:

```text
docs/legacy_audio_assistant_v1.5.0.py
```

## Fixed From v1.5.0 Baseline

- Short live denoise buffers now use adaptive `n_fft`, `win_length`, and `hop_length`.
- Native JACK/PortAudio probe noise is suppressed during audio device initialization.
- The application source is now importable and testable as a package.
- Runtime outputs are ignored without hiding source files.

## Default Prompt Behavior

The default file-transcription prompt is:

```text
這是一份專業的繁體中文會議紀錄，請務必根據語氣加上正確的全形標點符號。
```

It is loaded into the hidden Advanced Settings prompt field at startup and is passed to both batch file transcription and live recording when recording starts.

The lower-level ASR threads also have defaults now:

- File transcription uses the Traditional Mandarin meeting-record prompt when no prompt is supplied.
- Live transcription uses `The following is a professional meeting record.` when no live prompt is supplied.
- If a caller explicitly passes an empty string, the app respects that as "no prompt".

## Denoise Behavior

Live denoise is intentionally conservative:

- Silent and near-silent buffers are returned unchanged.
- Very tiny buffers are skipped because spectral reduction has too little context.
- Non-silent buffers use `noisereduce` in non-stationary mode with gentle reduction (`prop_decrease=0.35`).
- FFT and hop sizes are capped dynamically so short live buffers cannot trigger `noverlap must be less than nperseg`.

On the current workstation using the legacy `.record` environment, rough timings were:

| Buffer | Approx. audio length | Runtime |
| --- | ---: | ---: |
| 480 samples | 30 ms | ~11 ms |
| 8,000 samples | 0.5 s | ~12 ms |
| 16,000 samples | 1.0 s | ~13 ms |
| 128,000 samples | 8.0 s | ~33 ms |

A synthetic 2-second noisy tone check improved estimated SNR by about `+0.43 dB` without NaN/Inf output. This is only a smoke test, not a substitute for listening tests on real meeting audio.

## Install

Use a fresh virtual environment in this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Ubuntu system packages are still required for audio and media handling:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev ffmpeg
```

## Run

```bash
python -m aura
```

or, after editable install:

```bash
aura
```

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The current regression test uses the Python standard library and covers the short-buffer denoise failure reported from the live recorder path.

## Migration Notes

- Keep historical recordings in `record_audio_ubuntu` or another data folder.
- Use this repo for source refactoring, packaging, tests, and future releases.
- Copy only selected fixtures into `tests/fixtures/` if an audio regression needs a stable sample.
