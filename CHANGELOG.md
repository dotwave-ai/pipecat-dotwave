# Changelog

All notable changes to **pipecat-dotwave** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

### Added

- `WaveSTTService`, a `WebsocketSTTService` for the .wave streaming
  transcription socket (`wss://api.dotwave.ai/v1/listen`). It pushes
  `InterimTranscriptionFrame` for interim results and `TranscriptionFrame` for
  finals, sends `Finalize` on `VADUserStoppedSpeakingFrame` and marks the
  answering final as finalized, and closes with `CloseStream`.
- `WaveSTTSettings` (`WaveSTTService.Settings`) with `model` (default
  `nemotron-asr-streaming`) and `language` (default automatic detection).
  Pipecat `Language` values map to the tags the socket serves; changing a
  setting at runtime reconnects.
- `linear16`, `mulaw` and `alaw` encodings, converting pipeline PCM to the rate
  the socket accepts for each.
- `utterance_end_ms` and `interim_results` connection options.
- Silence keepalive, since a session closes after 30 seconds without audio.
- Refused credentials and malformed requests (close codes 4401 and 4400) are
  reported as permanent errors and not retried.
- Foundational example `examples/transcription-dotwave.py`.
