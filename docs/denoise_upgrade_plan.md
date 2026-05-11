# Denoise Upgrade Plan

## Decision

Keep the current `noisereduce` spectral-gating path as the lightweight fallback, expose its presets clearly in the desktop UI, and evaluate model-based speech enhancement before adding heavyweight runtime dependencies.

The next backend candidates are:

| Backend | Primary use | Integration priority |
| --- | --- | --- |
| `noisereduce` | Lightweight fallback and quiet-room baseline | Current default fallback |
| DeepFilterNet3 | Real-time or near-real-time ASR preprocessing | First model-based candidate |
| ClearerVoice-Studio | Offline high-quality enhancement, separation, and super-resolution | Second candidate for import workflow |
| VoiceFixer | Heavily degraded speech restoration | Research / rescue workflow only |

## Why Not Replace Everything Immediately

ASR preprocessing is not the same as audio mastering. A stronger denoiser can make speech sound cleaner while removing consonants, breath endings, or rare domain terms that the ASR model needs. For Project AURA, every denoise backend must be judged by transcript quality first, not only by listening quality.

The safe rollout is:

1. Preserve `off` as the default.
2. Keep `noisereduce` `light` and `medium` available for low-dependency usage.
3. Add optional model-based backends behind explicit settings.
4. Compare each backend against a fixed local evaluation set before making it the recommended mode.

## Current Implementation

The current implementation lives in `src/aura/audio/denoise.py`.

- `off` returns the original audio unchanged.
- `light` uses non-stationary `noisereduce` with `prop_decrease=0.35`.
- `medium` uses non-stationary `noisereduce` with `prop_decrease=0.55`.
- Very short and near-silent buffers are bypassed to avoid unstable STFT settings.
- The desktop UI exposes these as a `Denoise Mode` combo box.

## Evaluation Set

Create a small private evaluation folder outside git because it may contain meeting audio:

```text
~/record_jn/aura_eval_audio/
├── quiet_room/
├── fan_or_ac_noise/
├── cafe_or_background_speech/
├── lecture_or_meeting/
└── rare_terms/
```

For each folder, keep:

- `input.wav`
- `reference.txt` when a trusted transcript exists
- `notes.md` with room, microphone, language, and expected hard terms

The minimum useful set is 10-20 short clips of 30-90 seconds each.

## Metrics

Use two classes of checks:

| Check | Purpose |
| --- | --- |
| WER / CER | Measures transcript accuracy against `reference.txt` |
| Rare-term hit rate | Measures whether domain vocabulary survives enhancement |
| ASR runtime | Ensures enhancement does not make the workflow too slow |
| Listening spot check | Catches artifacts that metrics miss |

For Project AURA, the recommended ranking rule is:

1. Prefer lower CER/WER.
2. If CER/WER is tied, prefer higher rare-term hit rate.
3. If transcript quality is tied, prefer lower latency and fewer dependencies.
4. Do not promote a backend if it sounds cleaner but harms ASR output.

## Proposed CLI Harness

Add a future script such as:

```bash
python scripts/evaluate_denoise_backends.py \
  --input-dir ~/record_jn/aura_eval_audio \
  --backends off,noisereduce-light,noisereduce-medium,deepfilternet3,clearervoice \
  --model SoybeanMilk/faster-whisper-Breeze-ASR-25 \
  --output reports/denoise_eval_YYYYMMDD.md
```

The report should include:

- backend name
- processed audio path
- transcript path
- CER/WER when reference exists
- rare-term hits and misses
- runtime
- recommendation per audio category

## Backend Integration Shape

The future code should keep one public entrypoint:

```python
enhance_audio(input_audio, mode="off", backend="noisereduce")
```

Recommended internal layout:

```text
src/aura/audio/
├── denoise.py                  # public policy and current noisereduce fallback
├── enhancement_backends.py     # backend registry
├── deepfilternet_backend.py    # optional dependency boundary
└── clearvoice_backend.py       # optional dependency boundary
```

Optional dependencies should be loaded inside backend modules, not at app startup. If DeepFilterNet3 or ClearerVoice is not installed, the UI should show the mode as unavailable instead of crashing.

## Recommended Rollout

### Phase 1: UI and Baseline

- Expose `off`, `light`, and `medium` in Advanced Settings.
- Keep `off` as the default.
- Verify live recording and imported-file paths both pass the selected preset.

### Phase 2: Evaluation Harness

- Add the private evaluation folder contract.
- Add a local benchmark script.
- Save generated reports under `reports/`, while keeping raw audio outside git.

### Phase 3: DeepFilterNet3

- Add an optional DeepFilterNet3 backend.
- Start with imported-file processing.
- Only enable live processing after measuring latency and stream stability.

### Phase 4: ClearerVoice-Studio

- Add ClearerVoice as an offline import-only backend.
- Use it for difficult audio, speaker overlap, and enhancement experiments.
- Do not use it as the default live recording path.

### Phase 5: Promote Defaults

Promote a new default only after the evaluation report shows that it improves ASR output across normal meeting audio and does not reduce rare-term recognition.
