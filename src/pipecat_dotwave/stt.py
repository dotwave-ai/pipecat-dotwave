#
# Copyright (c) 2026, Dollyglot, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

""".wave speech-to-text service implementation.

This module provides a WebSocket-based STT service that streams audio to the
.wave ``/v1/listen`` socket, a Deepgram-compatible streaming transcription
endpoint, and turns its results into Pipecat transcription frames.
"""

import json
import urllib.parse
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.audio.utils import create_stream_resampler, pcm_to_alaw, pcm_to_ulaw
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

DEFAULT_BASE_URL = "api.dotwave.ai"
"""Host of the .wave API."""

DEFAULT_MODEL = "nemotron-asr-streaming"
"""Streaming transcription model used when no model is set."""

LISTEN_PATH = "/v1/listen"
"""Path of the streaming transcription socket."""

# The sample rates the socket accepts for each encoding. Linear PCM at any
# other pipeline rate is resampled to the first rate listed.
_WIRE_SAMPLE_RATES: dict[str, tuple[int, ...]] = {
    "linear16": (16000, 24000),
    "mulaw": (8000,),
    "alaw": (8000,),
}

# utterance_end_ms bounds the socket accepts.
_MIN_UTTERANCE_END_MS = 1000
_MAX_UTTERANCE_END_MS = 3200

# The socket closes a session after 30 seconds without audio. Silence is sent
# after this many idle seconds, checked every _KEEPALIVE_INTERVAL seconds.
_KEEPALIVE_TIMEOUT = 20.0
_KEEPALIVE_INTERVAL = 5.0

# Close codes after which reconnecting cannot help: the request or the key
# must change first.
_PERMANENT_CLOSE_CODES: dict[int, ErrorCategory] = {
    4400: ErrorCategory.INVALID_REQUEST,
    4401: ErrorCategory.AUTHENTICATION,
}

# Deepgram options the socket refuses by name. They are refused here too,
# at construction, rather than dropped without a word.
_UNSUPPORTED_OPTIONS: dict[str, str] = {
    "diarize": "speaker diarization is not supported",
    "multichannel": "multichannel audio is not supported; send one channel",
    "channels": "only single-channel audio is supported",
    "keywords": "keyword boosting is not supported",
    "endpointing": (
        "endpointing is not supported; segments end on Finalize (sent when the "
        "VAD reports the end of speech) or on utterance_end_ms"
    ),
}

# Languages written without spaces between words. The socket's words there are
# single characters, so the finals of a segment join with no separator.
_UNSPACED_LANGUAGE_PREFIXES = ("zh", "ja")

# Bare language codes mapped to the regional tag the socket serves for them.
# Tags already in this form (Language.PT_BR, "de-DE", ...) pass through as is,
# and "en" and "multi" are accepted by the socket directly.
_LANGUAGE_TAGS: dict[Language, str] = {
    Language.AR: "ar-AR",
    Language.BG: "bg-BG",
    Language.CS: "cs-CZ",
    Language.DA: "da-DK",
    Language.DE: "de-DE",
    Language.ES: "es-ES",
    Language.ET: "et-EE",
    Language.FI: "fi-FI",
    Language.FR: "fr-FR",
    Language.HI: "hi-IN",
    Language.HR: "hr-HR",
    Language.HU: "hu-HU",
    Language.IT: "it-IT",
    Language.JA: "ja-JP",
    Language.KO: "ko-KR",
    Language.NB: "nb-NO",
    Language.NL: "nl-NL",
    Language.NO: "nb-NO",
    Language.PL: "pl-PL",
    Language.RO: "ro-RO",
    Language.RU: "ru-RU",
    Language.SK: "sk-SK",
    Language.SV: "sv-SE",
    Language.TR: "tr-TR",
    Language.UK: "uk-UA",
    Language.VI: "vi-VN",
    Language.ZH: "zh-CN",
}


def language_to_dotwave_language(language: Language) -> str:
    """Convert a Pipecat language to the tag the .wave socket expects.

    Bare codes with a single regional variant on .wave map to that variant
    (``Language.DE`` to ``"de-DE"``). Everything else is passed through as its
    value (``Language.EN`` to ``"en"``, ``Language.PT_BR`` to ``"pt-BR"``), and
    the socket refuses a tag it does not serve with an error naming the tags
    it does.

    Args:
        language: The language to convert.

    Returns:
        The language tag for the ``language`` query parameter.
    """
    return _LANGUAGE_TAGS.get(language, language.value)


def dotwave_language_to_language(tag: str | None) -> Language | None:
    """Convert a language reported in a result to a Pipecat language.

    Results report the language as Deepgram spells it (``"en-US"``,
    ``"pt-BR"``, ``"de"``); the field is absent with automatic language
    detection.

    Args:
        tag: The reported language, or None.

    Returns:
        The matching language, or None when the tag is absent or unknown.
    """
    if not tag:
        return None
    try:
        return Language(tag)
    except ValueError:
        logger.debug(f"Unknown language tag in a .wave result: {tag!r}")
        return None


@dataclass
class WaveSTTSettings(STTSettings):
    """Settings for WaveSTTService.

    Parameters:
        model: Transcription model. Defaults to ``"nemotron-asr-streaming"``.
        language: Language to transcribe, as a Pipecat ``Language`` or a tag
            string such as ``"pt-BR"``. ``None`` (the default) or ``"multi"``
            selects automatic language detection. Changing the model or the
            language at runtime reconnects the socket.
        extra: Additional query parameters sent on the connection URL as given
            (booleans as ``true``/``false``). Parameters the socket refuses
            produce an error rather than being ignored.
    """


class WaveSTTService(WebsocketSTTService):
    """Speech-to-text service using the .wave streaming transcription socket.

    Streams pipeline audio to ``wss://api.dotwave.ai/v1/listen``, a
    Deepgram-compatible socket, and turns its results into transcription
    frames.

    The socket sends a final ``Results`` message as soon as words are settled,
    often one word at a time. By default (``segment_finals=True``) the service
    collects those finals and pushes one ``TranscriptionFrame`` per segment,
    with the whole text of the segment. A segment ends when:

    - the socket answers ``Finalize`` (a final marked ``from_finalize``). The
      service sends ``Finalize`` when the VAD reports the end of speech
      (``VADUserStoppedSpeakingFrame``), and this frame is marked finalized;
    - a final is marked ``speech_final``, or an ``UtteranceEnd`` message
      arrives (with ``utterance_end_ms`` set);
    - the service stops, is cancelled or reconnects with words pending.

    While a segment is open, every interim result and every final pushes an
    ``InterimTranscriptionFrame`` carrying the segment so far, so turn
    strategies and clients see live text. A final that arrives after a segment
    ended starts the next segment.

    The finals of a segment join with one space, except in Chinese and
    Japanese (a segment language starting with ``zh`` or ``ja``), where they
    join with no separator.

    With ``segment_finals=False`` the service pushes one frame per ``Results``
    message instead: an ``InterimTranscriptionFrame`` for each interim and a
    ``TranscriptionFrame`` for each final, holding only that message's words.

    The socket closes a session after 30 seconds without audio, so the
    service sends a short burst of silence after 20 idle seconds.

    See https://dotwave.ai/docs/deepgram/ for the socket reference.

    Event handlers:
        on_connected: Called when the socket is connected.
        on_disconnected: Called when the socket is disconnected.
        on_connection_error: Called when a connection error occurs.

    Example::

        stt = WaveSTTService(
            api_key=os.getenv("DOTWAVE_API_KEY"),
            settings=WaveSTTService.Settings(language=Language.EN_US),
        )
    """

    Settings = WaveSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        encoding: str = "linear16",
        sample_rate: int | None = None,
        utterance_end_ms: int | None = None,
        interim_results: bool = True,
        extra_headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = None,
        segment_finals: bool = True,
        **kwargs,
    ):
        """Initialize the .wave STT service.

        Args:
            api_key: .wave API key, sent as ``Authorization: Token <key>``.
            base_url: API endpoint, either a bare host (``"api.dotwave.ai"``) or a
                URL with a scheme (``"ws://localhost:8080"``). Defaults to
                ``"api.dotwave.ai"`` over ``wss``.
            encoding: Encoding sent on the socket: ``"linear16"`` (default),
                ``"mulaw"`` or ``"alaw"``. Pipeline audio is 16-bit PCM and is
                converted as needed: ``linear16`` is sent at 16 or 24 kHz
                (other rates are resampled to 16 kHz), ``mulaw`` and ``alaw``
                at 8 kHz.
            sample_rate: Sample rate of the audio this service receives. If
                None, uses the pipeline input sample rate.
            utterance_end_ms: Silence, in milliseconds (1000 to 3200), after
                which the socket ends a segment on its own, sending a final
                with ``speech_final`` and an ``UtteranceEnd`` message. If None
                (default), segments end on ``Finalize``, which the service sends
                when the VAD reports the end of speech.
            interim_results: Whether to receive interim results. Defaults to True.
            extra_headers: Additional headers to send on the websocket handshake.
                Merged after the default headers, so it can override them.
            settings: Runtime-updatable settings. See ``WaveSTTService.Settings``.
            ttfs_p99_latency: P99 latency from speech end to final transcript, in
                seconds. .wave has not measured this value for Pipecat; if None,
                Pipecat's default applies. Measure your deployment with
                https://github.com/pipecat-ai/stt-benchmark and pass the result.
            segment_finals: Whether to push one ``TranscriptionFrame`` per
                segment (True, the default), joining the finals the socket sends
                until ``Finalize`` is answered, a ``speech_final`` final or an
                ``UtteranceEnd`` message; interim frames then carry the segment
                so far. If False, every final ``Results`` message is pushed as
                its own ``TranscriptionFrame`` and every interim as its own
                ``InterimTranscriptionFrame``.
            **kwargs: Additional arguments passed to ``WebsocketSTTService``.

        Raises:
            ValueError: If ``encoding`` or ``utterance_end_ms`` is not accepted
                by the socket.
            TypeError: If a Deepgram option the socket does not support (such as
                ``diarize``) is passed.
        """
        unsupported = sorted(set(kwargs) & set(_UNSUPPORTED_OPTIONS))
        if unsupported:
            reasons = "; ".join(f"{name}: {_UNSUPPORTED_OPTIONS[name]}" for name in unsupported)
            raise TypeError(f"WaveSTTService does not support {', '.join(unsupported)} ({reasons})")
        if encoding not in _WIRE_SAMPLE_RATES:
            raise ValueError(
                f"encoding must be one of {', '.join(_WIRE_SAMPLE_RATES)}, got {encoding!r}"
            )
        if utterance_end_ms is not None and not (
            _MIN_UTTERANCE_END_MS <= utterance_end_ms <= _MAX_UTTERANCE_END_MS
        ):
            raise ValueError(
                f"utterance_end_ms must be between {_MIN_UTTERANCE_END_MS} and "
                f"{_MAX_UTTERANCE_END_MS}, got {utterance_end_ms}"
            )

        default_settings = self.Settings(model=DEFAULT_MODEL, language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        kwargs.setdefault("keepalive_timeout", _KEEPALIVE_TIMEOUT)
        kwargs.setdefault("keepalive_interval", _KEEPALIVE_INTERVAL)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        # Accept either a bare host ("api.dotwave.ai") or a full URL carrying a
        # scheme ("ws://localhost:8080"); urlsplit only finds a netloc after "//".
        url_parts = urllib.parse.urlsplit(base_url if "//" in base_url else f"//{base_url}")
        self._base_url = url_parts.netloc or DEFAULT_BASE_URL
        self._scheme = url_parts.scheme or "wss"
        self._extra_headers = dict(extra_headers) if extra_headers else {}
        self._receive_task = None

        # Init-only connection config (not runtime-updatable).
        self._encoding = encoding
        self._utterance_end_ms = utterance_end_ms
        self._interim_results = interim_results
        self._wire_sample_rate = 0
        self._resampler = create_stream_resampler()
        self._request_id: str | None = None

        # Segment-level finals: the finals since the last segment boundary, the
        # language reported for the segment and the last final of the segment.
        self._segment_finals = segment_finals
        self._segment: list[str] = []
        self._segment_language: str | None = None
        self._segment_result: dict[str, Any] | None = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Pipecat language to the tag the .wave socket expects.

        Args:
            language: The language to convert.

        Returns:
            The language tag for the ``language`` query parameter.
        """
        return language_to_dotwave_language(language)

    @property
    def request_id(self) -> str | None:
        """Identifier of the current socket session, from its ``Metadata`` message.

        Returns:
            The session identifier, or None before the socket has reported one.
        """
        return self._request_id

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        rates = _WIRE_SAMPLE_RATES[self._encoding]
        self._wire_sample_rate = self.sample_rate if self.sample_rate in rates else rates[0]
        await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, sending ``Finalize`` when speech ends.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                # Mark that a from_finalize final is expected, as Deepgram does.
                self.request_finalize()
                try:
                    await self._websocket.send(json.dumps({"type": "Finalize"}))
                except Exception as e:
                    logger.warning(f"{self}: failed to send Finalize: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the socket for transcription.

        Args:
            audio: Raw 16-bit PCM audio bytes.

        Yields:
            None: transcription results arrive on the socket.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()

        if self._websocket is None:
            logger.warning(f"{self}: websocket unavailable after reconnect, dropping audio")
            yield None
            return

        try:
            payload = await self._encode_audio(audio)
            if payload:
                await self._websocket.send(payload)
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Send silence in the socket's encoding to keep the session open.

        Args:
            silence: Silent 16-bit mono PCM audio bytes at the input sample rate.
        """
        if self._websocket is None:
            return
        payload = await self._encode_audio(silence)
        if payload:
            await self._websocket.send(payload)
        # Keepalive silence is audio the socket transcribes, so it counts
        # toward usage.
        self._record_stt_audio_usage(silence)

    async def _encode_audio(self, audio: bytes) -> bytes:
        """Convert pipeline PCM to the encoding and rate sent on the socket.

        Args:
            audio: Raw 16-bit PCM audio bytes at the input sample rate.

        Returns:
            The bytes to send, possibly empty while the resampler fills.
        """
        if self._encoding == "mulaw":
            return await pcm_to_ulaw(audio, self.sample_rate, 8000, self._resampler)
        if self._encoding == "alaw":
            return await pcm_to_alaw(audio, self.sample_rate, 8000, self._resampler)
        if self.sample_rate == self._wire_sample_rate:
            return audio
        return await self._resampler.resample(audio, self.sample_rate, self._wire_sample_rate)

    async def _connect(self):
        await self._connect_websocket()

        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting when anything changed.

        Model, language and extra parameters are set when the socket opens, so
        a change takes effect on a new connection.

        Args:
            delta: A :class:`STTSettings` (or ``WaveSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        await self._request_reconnect()

        return changed

    def _build_url(self) -> str:
        """Build the socket URL from the current settings.

        Returns:
            The ``/v1/listen`` URL with its query string.
        """
        params: list[tuple[str, str]] = [
            ("encoding", self._encoding),
            ("sample_rate", str(self._wire_sample_rate)),
            ("model", str(self._settings.model or DEFAULT_MODEL)),
            ("interim_results", "true" if self._interim_results else "false"),
        ]
        if self._settings.language:
            params.append(("language", str(self._settings.language)))
        if self._utterance_end_ms is not None:
            params.append(("utterance_end_ms", str(self._utterance_end_ms)))
        for key, value in self._settings.extra.items():
            if value is None:
                continue
            if isinstance(value, bool):
                params.append((key, "true" if value else "false"))
            elif isinstance(value, list):
                params.extend((key, str(item)) for item in value)
            else:
                params.append((key, str(value)))
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        return urllib.parse.urlunsplit((self._scheme, self._base_url, LISTEN_PATH, query, ""))

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to .wave STT")

            headers = {
                "Authorization": f"Token {self._api_key}",
                **self._extra_headers,
            }
            self._websocket = await self._websocket_connect(
                self._build_url(), additional_headers=headers
            )
            logger.info(f"{self}: Connected to .wave STT")
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            # A refused handshake (401, 400, ...) is classified from the
            # exception's HTTP status.
            await self.push_error(error_msg=f"Unable to connect to .wave: {e}", exception=e)

    async def _disconnect_websocket(self):
        # Push the words of an unfinished segment before the session ends, on
        # stop, cancel and reconnect alike, so none is lost.
        await self._flush_segment()
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug(f"{self}: Disconnecting from .wave STT")
                await ws.send(json.dumps({"type": "CloseStream"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            # Only clear if no concurrent _connect has already replaced it.
            if self._websocket is ws:
                self._websocket = None
            self._request_id = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Process incoming socket messages until the socket closes."""
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: received non-JSON message: {message}")
                    continue
                try:
                    await self._process_response(data)
                except Exception as e:
                    logger.error(f"{self}: error processing message: {e}")
        except ConnectionClosedError as e:
            close = e.rcvd
            category = _PERMANENT_CLOSE_CODES.get(close.code) if close else None
            if close is None or category is None:
                # Reconnect with backoff (the base class decides).
                raise
            # Reconnecting would only be refused the same way, so report it as
            # permanent; the base receive loop then stops trying.
            reason = f": {close.reason}" if close.reason else ""
            await self.push_error(
                error_msg=f".wave closed the connection ({close.code}{reason})",
                exception=e,
                category=category,
            )

    async def _process_response(self, data: dict[str, Any]):
        message_type = data.get("type")
        if message_type == "Results":
            await self._on_results(data)
        elif message_type == "Metadata":
            self._request_id = data.get("request_id")
            logger.debug(f"{self}: session {self._request_id}")
        elif message_type == "SpeechStarted":
            logger.trace(f"{self}: speech started at {data.get('timestamp')}")
        elif message_type == "UtteranceEnd":
            logger.trace(f"{self}: utterance ended at {data.get('last_word_end')}")
            if self._segment_finals:
                await self._end_segment(data)
        elif message_type == "Error":
            error_msg = data.get("message") or data.get("description") or "Unknown error"
            parameter = data.get("parameter")
            if parameter:
                error_msg = f"{error_msg} (parameter: {parameter})"
            await self.push_error(error_msg=f".wave error: {error_msg}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_results(self, data: dict[str, Any]):
        alternatives = (data.get("channel") or {}).get("alternatives") or []
        if not alternatives:
            return
        alternative = alternatives[0]
        transcript = alternative.get("transcript") or ""
        languages = alternative.get("languages") or []
        tag = languages[0] if languages else None

        if not self._segment_finals:
            await self._on_results_per_message(data, transcript, tag)
            return

        if tag:
            self._segment_language = tag

        if not data.get("is_final"):
            if transcript:
                await self._push_interim([*self._segment, transcript], data)
            return

        if transcript:
            self._segment.append(transcript)
            self._segment_result = data

        from_finalize = bool(data.get("from_finalize"))
        if from_finalize or data.get("speech_final"):
            await self._end_segment(data, from_finalize=from_finalize)
        elif transcript:
            # Not a boundary: show the segment so far, and wait for the end of
            # the segment to push the TranscriptionFrame.
            await self._push_interim(self._segment, data)

    def _join(self, parts: list[str]) -> str:
        """Join transcripts of one segment as the segment's language writes them.

        Args:
            parts: Transcripts in arrival order.

        Returns:
            The joined text.
        """
        # Results carry no language with automatic detection; the configured
        # language then decides.
        tag = str(self._segment_language or self._settings.language or "").lower()
        separator = "" if tag.startswith(_UNSPACED_LANGUAGE_PREFIXES) else " "
        return separator.join(part.strip() for part in parts if part.strip())

    async def _push_interim(self, parts: list[str], result: dict[str, Any]):
        text = self._join(parts)
        if not text:
            return
        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                dotwave_language_to_language(self._segment_language),
                result=result,
            )
        )

    async def _end_segment(self, result: dict[str, Any] | None, *, from_finalize: bool = False):
        """Push the open segment as one TranscriptionFrame and start a new one.

        Args:
            result: The message that ended the segment, set as the frame's
                ``result``.
            from_finalize: Whether the segment ended on the answer to Finalize.
        """
        text = self._join(self._segment)
        language = dotwave_language_to_language(self._segment_language)
        # Reset before pushing, so a message handled meanwhile starts the next
        # segment instead of joining this one.
        self._segment = []
        self._segment_language = None
        self._segment_result = None

        if not text:
            if from_finalize:
                # The words already went out in earlier segments, so the empty
                # answer to Finalize closes the measurement now instead of
                # waiting for the timeout.
                await self.stop_ttfb_metrics()
                await self._cancel_ttfb_timeout()
            return

        # Only mark as finalized when a finalize was requested and the socket
        # confirms it.
        if from_finalize:
            self.confirm_finalize()
        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
        )
        await self._handle_transcription(text, True, language)

    async def _flush_segment(self):
        """Push an unfinished segment, if any, as a TranscriptionFrame."""
        if self._segment_finals and self._segment:
            await self._end_segment(self._segment_result)

    async def _on_results_per_message(self, data: dict[str, Any], transcript: str, tag: str | None):
        """Push one frame per Results message (``segment_finals=False``)."""
        language = dotwave_language_to_language(tag)
        is_final = bool(data.get("is_final"))

        if is_final:
            from_finalize = bool(data.get("from_finalize"))
            if transcript:
                # Only mark as finalized when a finalize was requested and the
                # socket confirms it.
                if from_finalize:
                    self.confirm_finalize()
                # Report usage before the transcription frame so tracing can
                # attach it to the STT span the frame closes.
                await self.emit_stt_usage_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                    )
                )
                await self._handle_transcription(transcript, is_final, language)
            elif from_finalize:
                # The words already went out in earlier finals, so the empty
                # answer to Finalize closes the measurement now instead of
                # waiting for the timeout.
                await self.stop_ttfb_metrics()
                await self._cancel_ttfb_timeout()
        elif transcript:
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=data,
                )
            )
