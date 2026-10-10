#
# Copyright (c) 2026, Dollyglot, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcribe your microphone with .wave in a Pipecat pipeline.

Run it, open http://localhost:7860 in a browser, connect and talk. What you
say appears on the page and in the terminal, one line per sentence or pause;
interim results are logged at debug level.

    export DOTWAVE_API_KEY=...
    uv run examples/transcription-dotwave.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_dotwave import WaveSTTService

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    """Prints each final transcription; interim results go to the debug log."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text} (language: {frame.language})")
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"Interim: {frame.text}")

        await self.push_frame(frame, direction)


# The development runner picks the transport; WebRTC opens a page on localhost.
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting transcription with .wave")

    # The VAD tells the service when speech ends, so it can ask for the final.
    # Half a second of silence ends a segment, so short pauses do not split a
    # sentence.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)))

    # No language set: .wave detects the language being spoken. Pin one when
    # you know it, which is more accurate: settings=WaveSTTService.Settings(
    # language="fr-FR"). A session pinned to the wrong language drops
    # most of what is said.
    stt = WaveSTTService(api_key=os.getenv("DOTWAVE_API_KEY"))

    # The pipeline worker adds Pipecat's RTVI processor and observer by default.
    # They turn transcription frames into RTVI messages, which reach the
    # browser page through transport.output().
    # Turn frames tell the browser page when a turn ends, so each turn shows as
    # its own entry. A turn starts when the VAD hears speech and ends after
    # 0.8 s without it; a conversational pipeline would use the LLM user
    # aggregator's turn handling instead. Interruptions stay off: there is no
    # bot to interrupt, and an interruption discards the frames queued at every
    # processor, which would drop a transcription still on its way to the page
    # when the speaker starts the next turn.
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=False)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8)],
        )
    )

    pipeline = Pipeline(
        [transport.input(), vad, stt, turns, TranscriptionLogger(), transport.output()]
    )

    worker = PipelineWorker(pipeline, idle_timeout_secs=runner_args.pipeline_idle_timeout_secs)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point, called by the Pipecat development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
