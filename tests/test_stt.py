#
# Copyright (c) 2026, Dollyglot, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WaveSTTService against a local fake of the /v1/listen socket."""

import asyncio
import json
import urllib.parse
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from websockets.asyncio.server import serve

from pipecat_dotwave import WaveSTTService, WaveSTTSettings
from pipecat_dotwave.stt import dotwave_language_to_language, language_to_dotwave_language

API_KEY = "test-key"


def _results(
    transcript: str,
    *,
    is_final: bool,
    speech_final: bool = False,
    from_finalize: bool = False,
    language: str | None = "en-US",
) -> str:
    alternative = {"transcript": transcript, "confidence": 1.0, "words": []}
    if language:
        alternative["languages"] = [language]
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": 0.16,
            "duration": 0.08,
            "is_final": is_final,
            "speech_final": speech_final,
            "from_finalize": from_finalize,
            "metadata": {
                "request_id": "sess_test",
                "model_info": {
                    "name": "nemotron-asr-streaming",
                    "arch": "nemotron-asr-streaming",
                    "version": "v1",
                },
                "model_uuid": "nemotron-asr-streaming",
            },
            "channel": {"alternatives": [alternative]},
        }
    )


@dataclass
class FakeListenServer:
    """A minimal stand-in for the /v1/listen socket.

    On the first audio message it sends SpeechStarted, an interim and a final
    ("Hello"); on the second an interim ("wor") and a final ("world."); it
    answers each Finalize with an empty from_finalize final, as the socket does
    when every word already went out in a final.

    A test can replace what follows the first audio message with ``script``,
    and the answer to Finalize with ``finalize_reply``.
    """

    language: str | None = "en-US"
    reject_with: int | None = None
    script: list[str] | None = None
    finalize_reply: str | None = None
    connections: list[dict] = field(default_factory=list)
    text_messages: list[dict] = field(default_factory=list)
    audio_bytes: int = 0

    async def handler(self, connection):
        request = connection.request
        url = urllib.parse.urlsplit(request.path)
        self.connections.append(
            {
                "path": url.path,
                "query": urllib.parse.parse_qs(url.query),
                "authorization": request.headers.get("Authorization"),
            }
        )
        if request.headers.get("Authorization") != f"Token {API_KEY}" or self.reject_with:
            await connection.send(json.dumps({"type": "Error", "message": "invalid credential"}))
            await connection.close(self.reject_with or 4401, "invalid credential")
            return

        await connection.send(
            json.dumps({"type": "Metadata", "request_id": "sess_test", "channels": 1})
        )
        audio_messages = 0
        async for message in connection:
            if isinstance(message, bytes):
                audio_messages += 1
                self.audio_bytes += len(message)
                if audio_messages == 1 and self.script is not None:
                    for scripted in self.script:
                        await connection.send(scripted)
                elif audio_messages == 1:
                    await connection.send(
                        json.dumps({"type": "SpeechStarted", "channel": [0], "timestamp": 0.0})
                    )
                    await connection.send(_results("hel", is_final=False, language=self.language))
                    await connection.send(_results("Hello", is_final=True, language=self.language))
                elif audio_messages == 2 and self.script is None:
                    await connection.send(_results("wor", is_final=False, language=self.language))
                    await connection.send(_results("world.", is_final=True, language=self.language))
                continue
            data = json.loads(message)
            self.text_messages.append(data)
            if data["type"] == "Finalize":
                reply = self.finalize_reply or _results(
                    "", is_final=True, from_finalize=True, language=self.language
                )
                await connection.send(reply)
            elif data["type"] == "CloseStream":
                await connection.close(1000)
                return


@pytest.fixture
async def fake_server():
    server_state = FakeListenServer()
    async with serve(server_state.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        server_state.base_url = f"ws://127.0.0.1:{port}"
        yield server_state


def _audio(chunk_ms: int = 100, sample_rate: int = 16000) -> InputAudioRawFrame:
    samples = sample_rate * chunk_ms // 1000
    return InputAudioRawFrame(audio=b"\x01\x00" * samples, sample_rate=sample_rate, num_channels=1)


def _transcriptions(frames):
    return [f for f in frames if isinstance(f, (InterimTranscriptionFrame, TranscriptionFrame))]


async def test_streams_audio_and_maps_results(fake_server):
    stt = WaveSTTService(
        api_key=API_KEY,
        base_url=fake_server.base_url,
        settings=WaveSTTService.Settings(language=Language.EN_US),
    )
    down, _ = await run_test(
        stt,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(),
            SleepFrame(sleep=0.2),
            _audio(),
            SleepFrame(sleep=0.2),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ],
    )

    frames = _transcriptions(down)
    # Interims carry the segment so far; the segment is pushed once, when the
    # socket answers Finalize, and marked finalized.
    assert [(type(f), f.text) for f in frames] == [
        (InterimTranscriptionFrame, "hel"),
        (InterimTranscriptionFrame, "Hello"),
        (InterimTranscriptionFrame, "Hello wor"),
        (InterimTranscriptionFrame, "Hello world."),
        (TranscriptionFrame, "Hello world."),
    ]
    assert all(f.language == Language.EN_US for f in frames)
    (final,) = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert final.finalized
    assert final.result["from_finalize"] is True

    (connection,) = fake_server.connections
    assert connection["path"] == "/v1/listen"
    assert connection["authorization"] == f"Token {API_KEY}"
    assert connection["query"] == {
        "encoding": ["linear16"],
        "sample_rate": ["16000"],
        "model": ["nemotron-asr-streaming"],
        "interim_results": ["true"],
        "language": ["en-US"],
    }
    # Finalize on the end of speech, CloseStream on the way out.
    assert [m["type"] for m in fake_server.text_messages] == ["Finalize", "CloseStream"]
    assert fake_server.audio_bytes == 2 * 3200


async def test_query_string_options(fake_server):
    stt = WaveSTTService(
        api_key=API_KEY,
        base_url=fake_server.base_url,
        utterance_end_ms=1000,
        interim_results=False,
        settings=WaveSTTService.Settings(language=Language.PT_BR, extra={"punctuate": True}),
    )
    await run_test(stt, frames_to_send=[SleepFrame(sleep=0.1)])

    (connection,) = fake_server.connections
    assert connection["query"] == {
        "encoding": ["linear16"],
        "sample_rate": ["16000"],
        "model": ["nemotron-asr-streaming"],
        "interim_results": ["false"],
        "language": ["pt-BR"],
        "utterance_end_ms": ["1000"],
        "punctuate": ["true"],
    }


async def test_automatic_language_detection_omits_language(fake_server):
    fake_server.language = None
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    down, _ = await run_test(
        stt,
        frames_to_send=[_audio(), SleepFrame(sleep=0.3)],
    )

    (connection,) = fake_server.connections
    assert "language" not in connection["query"]
    frames = _transcriptions(down)
    # The open segment is pushed when the pipeline ends.
    assert [f.text for f in frames] == ["hel", "Hello", "Hello"]
    assert all(f.language is None for f in frames)


async def test_unknown_result_language_does_not_raise(fake_server):
    fake_server.language = "xx-YY"
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(sleep=0.3)])

    frames = _transcriptions(down)
    assert [f.text for f in frames] == ["hel", "Hello", "Hello"]
    assert all(f.language is None for f in frames)


async def test_mulaw_converts_pipeline_audio(fake_server):
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url, encoding="mulaw")
    await run_test(
        stt,
        frames_to_send=[_audio(), _audio(), SleepFrame(sleep=0.3)],
    )

    (connection,) = fake_server.connections
    assert connection["query"]["encoding"] == ["mulaw"]
    assert connection["query"]["sample_rate"] == ["8000"]
    # 200 ms of 16 kHz PCM16 (6,400 bytes) is at most 1,600 one-byte samples at 8 kHz.
    assert 0 < fake_server.audio_bytes <= 1600


async def test_settings_change_reconnects(fake_server):
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    await run_test(
        stt,
        frames_to_send=[
            SleepFrame(sleep=0.1),
            STTUpdateSettingsFrame(delta=WaveSTTSettings(language=Language.DE)),
            SleepFrame(sleep=0.2),
        ],
    )

    assert len(fake_server.connections) == 2
    assert "language" not in fake_server.connections[0]["query"]
    assert fake_server.connections[1]["query"]["language"] == ["de-DE"]


async def test_bad_credential_is_reported_and_not_retried(fake_server):
    stt = WaveSTTService(api_key="wrong-key", base_url=fake_server.base_url)
    _, up = await run_test(stt, frames_to_send=[SleepFrame(sleep=0.3)])

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    messages = [e.error for e in errors]
    assert any("invalid credential" in m for m in messages)
    assert any("4401" in m for m in messages)
    assert not stt.is_usable
    # One attempt only: a refused credential is not retried.
    assert len(fake_server.connections) == 1


def _segment_frames(frames):
    return [(type(f).__name__, f.text) for f in _transcriptions(frames)]


async def test_speech_final_ends_segment_without_finalize(fake_server):
    fake_server.script = [
        _results("Good", is_final=True),
        _results("morning.", is_final=True, speech_final=True),
        _results("Bye", is_final=True),
    ]
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(sleep=0.3)])

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "Good"),
        ("TranscriptionFrame", "Good morning."),
        # A final after the boundary starts the next segment, pushed at the end.
        ("InterimTranscriptionFrame", "Bye"),
        ("TranscriptionFrame", "Bye"),
    ]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.finalized for f in finals] == [False, False]
    assert finals[0].result["speech_final"] is True
    assert finals[0].language == Language.EN_US


async def test_utterance_end_ends_segment(fake_server):
    utterance_end = json.dumps({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.2})
    fake_server.script = [
        _results("Hi", is_final=True),
        _results("there", is_final=False),
        _results("there", is_final=True),
        utterance_end,
        # A second boundary with nothing pending pushes nothing.
        utterance_end,
    ]
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url, utterance_end_ms=1000)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(sleep=0.3)])

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "Hi"),
        ("InterimTranscriptionFrame", "Hi there"),
        ("InterimTranscriptionFrame", "Hi there"),
        ("TranscriptionFrame", "Hi there"),
    ]
    (final,) = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert final.result["type"] == "UtteranceEnd"
    assert not final.finalized


async def test_finalize_answer_carries_its_own_word(fake_server):
    fake_server.script = [_results("Thank", is_final=True)]
    fake_server.finalize_reply = _results("you.", is_final=True, from_finalize=True)
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    down, _ = await run_test(
        stt,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(),
            SleepFrame(sleep=0.2),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ],
    )

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "Thank"),
        ("TranscriptionFrame", "Thank you."),
    ]
    (final,) = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert final.finalized


async def test_chinese_segment_joins_without_spaces(fake_server):
    fake_server.language = "zh-CN"
    fake_server.script = [
        _results("你", is_final=True, language="zh-CN"),
        _results("好", is_final=True, language="zh-CN"),
        _results("世", is_final=False, language="zh-CN"),
        _results("世", is_final=True, speech_final=True, language="zh-CN"),
    ]
    stt = WaveSTTService(
        api_key=API_KEY,
        base_url=fake_server.base_url,
        settings=WaveSTTService.Settings(language=Language.ZH),
    )
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(sleep=0.3)])

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "你"),
        ("InterimTranscriptionFrame", "你好"),
        ("InterimTranscriptionFrame", "你好世"),
        ("TranscriptionFrame", "你好世"),
    ]
    assert all(f.language == Language.ZH_CN for f in _transcriptions(down))


async def test_per_message_frames_when_segment_finals_off(fake_server):
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url, segment_finals=False)
    down, _ = await run_test(
        stt,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(),
            SleepFrame(sleep=0.2),
            _audio(),
            SleepFrame(sleep=0.2),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ],
    )

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "hel"),
        ("TranscriptionFrame", "Hello"),
        ("InterimTranscriptionFrame", "wor"),
        ("TranscriptionFrame", "world."),
    ]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    # The empty answer to Finalize pushes nothing, so no final is finalized.
    assert [f.finalized for f in finals] == [False, False]


async def test_pending_segment_is_pushed_on_end_frame(fake_server):
    # No boundary arrives: the run ends with an EndFrame while words are pending.
    fake_server.script = [_results("Still", is_final=True), _results("talking", is_final=True)]
    stt = WaveSTTService(api_key=API_KEY, base_url=fake_server.base_url)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(sleep=0.3)])

    assert _segment_frames(down) == [
        ("InterimTranscriptionFrame", "Still"),
        ("InterimTranscriptionFrame", "Still talking"),
        ("TranscriptionFrame", "Still talking"),
    ]
    (final,) = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert not final.finalized
    assert final.result["channel"]["alternatives"][0]["transcript"] == "talking"


def test_language_mapping():
    assert language_to_dotwave_language(Language.PT_BR) == "pt-BR"
    assert language_to_dotwave_language(Language.EN) == "en"
    assert language_to_dotwave_language(Language.EN_US) == "en-US"
    assert language_to_dotwave_language(Language.DE) == "de-DE"
    assert language_to_dotwave_language(Language.NO) == "nb-NO"
    assert language_to_dotwave_language(Language.AR) == "ar-AR"
    assert language_to_dotwave_language(Language.ES_US) == "es-US"
    # A language with no single regional variant passes through unchanged.
    assert language_to_dotwave_language(Language.PT) == "pt"

    assert dotwave_language_to_language("en-US") == Language.EN_US
    assert dotwave_language_to_language("pt-BR") == Language.PT_BR
    assert dotwave_language_to_language("de") == Language.DE
    assert dotwave_language_to_language("no") == Language.NO
    assert dotwave_language_to_language("ar") == Language.AR
    assert dotwave_language_to_language(None) is None
    assert dotwave_language_to_language("not-a-language") is None


def test_settings_language_conversion_at_init():
    stt = WaveSTTService(api_key=API_KEY, settings=WaveSTTSettings(language=Language.FR))
    assert stt._settings.language == "fr-FR"
    assert stt._settings.model == "nemotron-asr-streaming"

    stt = WaveSTTService(api_key=API_KEY, settings=WaveSTTSettings(language="multi"))
    assert stt._settings.language == "multi"

    stt = WaveSTTService(api_key=API_KEY)
    assert stt._settings.language is None


@pytest.mark.parametrize("option", ["diarize", "multichannel", "keywords", "endpointing"])
def test_unsupported_options_are_refused(option):
    with pytest.raises(TypeError, match=option):
        WaveSTTService(api_key=API_KEY, **{option: True})


def test_invalid_init_values_are_refused():
    with pytest.raises(ValueError, match="encoding"):
        WaveSTTService(api_key=API_KEY, encoding="opus")
    with pytest.raises(ValueError, match="utterance_end_ms"):
        WaveSTTService(api_key=API_KEY, utterance_end_ms=500)
    with pytest.raises(ValueError, match="utterance_end_ms"):
        WaveSTTService(api_key=API_KEY, utterance_end_ms=5000)


def test_base_url_forms():
    stt = WaveSTTService(api_key=API_KEY)
    assert (stt._scheme, stt._base_url) == ("wss", "api.dotwave.ai")
    stt = WaveSTTService(api_key=API_KEY, base_url="ws://localhost:8080")
    assert (stt._scheme, stt._base_url) == ("ws", "localhost:8080")
    stt = WaveSTTService(api_key=API_KEY, base_url="eu.example.test")
    assert (stt._scheme, stt._base_url) == ("wss", "eu.example.test")
    assert stt.can_generate_metrics()


async def test_no_network_needed():
    # Constructing the service opens nothing; only setup connects.
    stt = WaveSTTService(api_key=API_KEY)
    await asyncio.sleep(0)
    assert stt._websocket is None
