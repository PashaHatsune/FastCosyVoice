# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 FastCosyVoice Implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FastCosyVoice3 - Parallel Pipeline Implementation

This module provides a parallelized version of CosyVoice3 with separate CUDA streams
for LLM, Flow, and Hift stages to maximize GPU utilization and reduce latency.

Supports TensorRT acceleration:
- Flow decoder TensorRT: ~2.5x speedup (load_trt=True)
- LLM TensorRT-LLM: ~3x speedup (load_trt_llm=True)

Usage:
    from fastcosyvoice import FastCosyVoice3
    
    # With both TensorRT accelerations
    model = FastCosyVoice3(
        "pretrained_models/Fun-CosyVoice3-0.5B",
        fp16=True,
        load_trt=True,      # Flow decoder TensorRT
        load_trt_llm=True,  # LLM TensorRT-LLM (~3x faster)
    )
    
    # Inference
    for chunk in model.inference_zero_shot_stream(
        "Hello world",
        "Reference text",
        "reference.wav"
    ):
        audio = chunk['tts_speech']
"""

from .cosyvoice import FastCosyVoice3
from .model import FastCosyVoice3Model

__all__ = [
    'FastCosyVoice3',
    'FastCosyVoice3Model',
]

