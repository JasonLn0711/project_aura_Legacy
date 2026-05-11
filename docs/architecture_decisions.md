# Architecture Decisions

## First-Principles Ownership Split

Project AURA is a desktop audio application, but its core value is not the UI framework. Its core value is reliable audio capture, preparation, transcription, splitting, and export.

Therefore, each layer has one owner:

- `src/aura/settings.py` owns runtime defaults that should be easy to inspect, override, and test.
- `src/aura/ui/messages.py` owns user-facing text and dynamic UI message formatting.
- `src/aura/asr/` owns transcription behavior and ASR worker orchestration.
- `src/aura/audio/` owns audio capture, denoise, export, and splitting behavior.
- `src/aura/system/` owns platform/runtime concerns such as CUDA, native audio stderr, runtime paths, and update checks.
- `src/aura/ui/` owns widgets, signal wiring, and user interaction only.

The practical rule is: if a behavior can be tested without starting Qt, keep it outside `src/aura/ui/`.

## Current Refactor Direction

Keep extracting logic from UI classes into small service modules, then protect the service modules with fast synthetic-audio tests. This reduces the risk of changing the desktop UI while preserving behavior from the legacy one-file app.

The denoise policy is now explicit as presets: `off`, `light`, and `medium`. Advanced Settings exposes these as a `Denoise Mode` combo box while keeping `off` as the default.

The next high-value cleanup is to add the evaluation harness described in `docs/denoise_upgrade_plan.md`, then test DeepFilterNet3 and ClearerVoice-Studio as optional model-based backends before promoting any new default.
