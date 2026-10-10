#
# Copyright (c) 2026, Dollyglot, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcribe your microphone with .wave in a Pipecat pipeline.

Run it, open http://localhost:7860 in a browser, connect and talk: interim
results and finals are printed in the terminal.

    export DOTWAVE_API_KEY=...
    uv run examples/transcription-dotwave.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_dotwave import WaveSTTService

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    """Prints interim results and finals as they arrive."""

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
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    stt = WaveSTTService(
        api_key=os.getenv("DOTWAVE_API_KEY"),
        settings=WaveSTTService.Settings(language=Language.EN_US),
    )

    pipeline = Pipeline([transport.input(), vad, stt, TranscriptionLogger()])

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
