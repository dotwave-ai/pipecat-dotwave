# pipecat-dotwave

[.wave](https://dotwave.ai) streaming speech-to-text for [Pipecat](https://github.com/pipecat-ai/pipecat).

`WaveSTTService` streams your pipeline's audio to the .wave `/v1/listen` socket, a
Deepgram-compatible streaming transcription endpoint, and pushes
`InterimTranscriptionFrame` and `TranscriptionFrame` frames as results arrive.

Built and maintained by .wave ([dotwave.ai](https://dotwave.ai)).

## Installation

```bash
uv add pipecat-dotwave
```

or with pip:

```bash
pip install pipecat-dotwave
```

## Prerequisites

Create an API key in the .wave console and export it:

```bash
export DOTWAVE_API_KEY=...
```

## Usage

```python
import os

from pipecat.transcriptions.language import Language

from pipecat_dotwave import WaveSTTService

stt = WaveSTTService(
    api_key=os.getenv("DOTWAVE_API_KEY"),
    settings=WaveSTTService.Settings(language=Language.EN_US),
)

pipeline = Pipeline(
    [
        transport.input(),
        vad,  # e.g. VADProcessor(vad_analyzer=SileroVADAnalyzer())
        stt,
        ...,
    ]
)
```

Put a VAD upstream of the service (a `VADProcessor`, or the user aggregator's
VAD in an agent pipeline). When the VAD reports the end of speech, the service
sends `Finalize`, and the segment it closes is pushed as a finalized
`TranscriptionFrame`.

### How results arrive

The socket sends a final result as soon as words are settled, often one word
at a time. The service collects those finals and pushes **one
`TranscriptionFrame` per segment**, holding the whole text of the segment. A
segment ends when:

- the socket answers `Finalize`, which the service sends when the VAD reports
  the end of speech. This frame is marked `finalized`;
- a final is marked `speech_final`, or an `UtteranceEnd` message arrives (with
  `utterance_end_ms` set);
- the service stops, is cancelled or reconnects with words pending, so nothing
  is lost.

While a segment is open, each interim result and each final pushes an
`InterimTranscriptionFrame` with the segment so far (settled words plus the
current guess), so turn strategies and RTVI clients show live text. Each
interim replaces the previous one. A final that arrives after a segment ended
starts the next segment.

Finals join with one space, except in Chinese and Japanese (a language tag
starting with `zh` or `ja`), where they join with no separator.

Pass `segment_finals=False` to get one frame per result instead: an
`InterimTranscriptionFrame` for each interim and a `TranscriptionFrame` for
each final, holding only that result's words. Successive `TranscriptionFrame`
texts then join with one space, which is what Pipecat's user aggregator does.

### Constructor parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | `str` | required | .wave API key, sent as `Authorization: Token <key>`. |
| `base_url` | `str` | `"api.dotwave.ai"` | Bare host, or a URL with a scheme (`"ws://localhost:8080"`). |
| `encoding` | `str` | `"linear16"` | `"linear16"`, `"mulaw"` or `"alaw"`. Pipeline audio is converted: `linear16` is sent at 16 or 24 kHz (other rates are resampled to 16 kHz), `mulaw` and `alaw` at 8 kHz. |
| `sample_rate` | `int \| None` | `None` | Sample rate of the incoming audio; the pipeline's input rate if `None`. |
| `utterance_end_ms` | `int \| None` | `None` | 1000 to 3200. Silence after which the socket ends a segment on its own (a final with `speech_final`, then `UtteranceEnd`). If `None`, segments end on `Finalize`. |
| `interim_results` | `bool` | `True` | Receive interim results. |
| `extra_headers` | `dict[str, str] \| None` | `None` | Extra handshake headers. |
| `settings` | `WaveSTTService.Settings \| None` | `None` | Runtime-updatable settings (below). |
| `segment_finals` | `bool` | `True` | Push one `TranscriptionFrame` per segment, with interims carrying the segment so far. `False` pushes one frame per result (see [How results arrive](#how-results-arrive)). |
| `ttfs_p99_latency` | `float \| None` | `None` | P99 speech-end-to-final latency, in seconds, used by turn strategies. .wave has not measured it for Pipecat, so Pipecat's default applies; measure your deployment with [stt-benchmark](https://github.com/pipecat-ai/stt-benchmark) and pass the result. |

### Settings

`WaveSTTService.Settings` fields can be changed at runtime with
`STTUpdateSettingsFrame`; a change reconnects the socket (after the user's turn
if they are speaking).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | `"nemotron-asr-streaming"` | Transcription model. |
| `language` | `Language \| str \| None` | `None` | Language to transcribe. `None` or `"multi"` selects automatic language detection. |
| `extra` | `dict` | `{}` | Additional query parameters, sent as given. |

A Pipecat `Language` is sent as the tag the socket expects: regional values pass
through (`Language.PT_BR` as `pt-BR`, `Language.EN_US` as `en-US`), `Language.EN`
is sent as `en` (English, United States), and a bare code with a single regional
variant on .wave is sent as that variant (`Language.DE` as `de-DE`). You can also
pass a tag string such as `"pt-BR"`. A language .wave does not serve is refused
when the socket opens, with an error naming the tags it does serve; see the
[socket reference](https://dotwave.ai/docs/deepgram/) for the list.

Results report the language as Deepgram spells it (`en-US`, `pt-BR`, `de`), and
the service sets it on each frame as a Pipecat `Language`. With automatic
detection, frames carry no language.

### Not supported

`diarize`, `multichannel`, `channels`, `keywords` and `endpointing` are not
supported by the socket. Passing them to `WaveSTTService` raises `TypeError`
instead of being ignored, and a parameter the socket refuses through `extra` is
reported as an `ErrorFrame`.

### Errors and reconnection

An `Error` message from the socket is pushed upstream as an `ErrorFrame`. A
refused key (close code 4401) or a malformed request (4400) is reported as a
permanent error and is not retried. Other closes, such as the service restarting
(1012) or being temporarily unable to take a new session (4429), reconnect
with backoff.

A session closes after 30 seconds without audio, so the service sends a short
burst of silence after 20 idle seconds.

### Event handlers

| Event | Arguments | Description |
| --- | --- | --- |
| `on_connected` | `(service)` | The socket connected. |
| `on_disconnected` | `(service)` | The socket disconnected. |
| `on_connection_error` | `(service, error: str)` | A connection error occurred. |

```python
@stt.event_handler("on_connected")
async def on_connected(service):
    logger.info("Connected to .wave")
```

## Running the example

[`examples/transcription-dotwave.py`](examples/transcription-dotwave.py)
transcribes what you say into your browser's microphone:

```bash
git clone https://github.com/dotwave-ai/pipecat-dotwave
cd pipecat-dotwave
uv sync
uv pip install "pipecat-ai[runner,webrtc,silero]"
export DOTWAVE_API_KEY=...
uv run examples/transcription-dotwave.py
```

Open http://localhost:7860, connect, and talk. Your words appear on the page
(the pipeline worker's RTVI support sends them through `transport.output()`)
and in the terminal, one `Transcription:` line per segment.

## Pricing

Streaming transcription is billed at $0.00045 per minute of audio. See
[pricing and limits](https://dotwave.ai/docs/pricing-and-limits/).

## Compatibility

| | Version |
| --- | --- |
| Pipecat | >= 1.12.0 |
| Python | >= 3.11 (Pipecat 1.12.0 requires it) |

Tested with Pipecat 1.12.0.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run ruff check && uv run ruff format --check
```

The tests run against a local fake of the socket and need neither a network
connection nor an API key.

## License

BSD 2-Clause. See [LICENSE](LICENSE).

## About

Built and maintained by .wave ([dotwave.ai](https://dotwave.ai)). Questions:
info@dotwave.ai.

## Releasing

Tag a version (`git tag v0.1.0 && git push --tags`). The `publish.yml`
workflow builds the wheel and sdist with `uv build` and uploads them to PyPI
through a trusted publisher configured for this repository. Without Actions,
`uv build && uv publish` does the same from a machine with a PyPI token.
