# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
#
# OmniVoice-only build: only prompt_utils survives so the multi-TTS
# dispatcher in entrypoints/openai/serving_speech.py can still import it
# at module load. The actual Ming model classes are not bundled.
