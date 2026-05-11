# Refactor Plan

## Decision

Use this sibling repository as the new maintainable Python codebase. Keep `record_audio_ubuntu` as the legacy working/data folder.

## Boundaries

- Do not import recordings, transcripts, `.record/`, or generated split files into this repo.
- Keep the one-file legacy implementation only as an audit baseline.
- Prefer package modules under `src/aura/` for all new work.

## Refactor Phases

1. Package split
   - Move system helpers, ASR threads, audio workers, and UI tabs into separate modules.
   - Keep behavior equivalent to `audio_assistant_v1.5.0.py`.

2. Regression tests
   - Cover denoise short-buffer handling.
   - Add splitter tests using tiny synthetic audio fixtures.
   - Add import smoke tests for all modules.

3. Runtime hardening
   - Replace `terminate()` on file worker shutdown with cooperative cancellation.
   - Move temp files into a configurable runtime directory.
   - Add structured logging instead of `print()`.

4. UI cleanup
   - Move display strings into a localization layer.
   - Keep English and Traditional Mandarin variants in one codebase instead of duplicate scripts.

5. Packaging
   - Add release commands.
   - Add CI checks for compile, tests, and formatting.
