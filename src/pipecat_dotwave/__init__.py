#
# Copyright (c) 2026, Dollyglot, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

""".wave speech-to-text for Pipecat.

Provides :class:`WaveSTTService`, a streaming STT service for the .wave
``/v1/listen`` socket, and its settings class :class:`WaveSTTSettings`.
"""

from pipecat_dotwave.stt import WaveSTTService, WaveSTTSettings

__all__ = ["WaveSTTService", "WaveSTTSettings"]
__version__ = "0.1.0"
