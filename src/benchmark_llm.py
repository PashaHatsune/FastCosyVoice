#!/usr/bin/env python3
"""
Benchmark script for CosyVoice3 LLM model (Qwen2-based).
Tests raw token generation speed without TTS overhead.

Usage:
    python benchmark_llm.py  # SDPA vs torch.compile(default)

TF32 (TensorFloat-32):
- Only helps FP32 operations, not FP16
- Enabled automatically, but since we use FP16, it won't help here
- For FP32 workloads: ~2-3x speedup on Ampere GPUs (RTX 30xx, A100)

Results on RTX 3090 (FP16):
- SDPA: ~50 tok/s
- torch.compile(default): ~95-105 tok/s (~2x)
- Flash Attention 2: ~35 tok/s (worse than SDPA for KV-cache inference)
"""

import time
import torch
from transformers import Qwen2ForCausalLM, Qwen2Config
import os

# Configuration
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
LLM_PT_PATH = os.path.join(MODEL_DIR, "llm.pt")
DEVICE = "cuda"
DTYPE = torch.float16

# Model params from config
LLM_INPUT_SIZE = 896
LLM_OUTPUT_SIZE = 896
SPEECH_TOKEN_SIZE = 6561

# Qwen2 config from the model
QWEN2_CONFIG = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "use_cache": True,
    "tie_word_embeddings": False,
}


def create_qwen2_model(attn_implementation="sdpa"):
    """Create a Qwen2 model from config."""
    config = Qwen2Config(**QWEN2_CONFIG, attn_implementation=attn_implementation)
    model = Qwen2ForCausalLM(config)
    return model


def run_generation_benchmark(model, name, seq_lengths=[100, 200, 500, 1000], warmup=True, num_runs=3):
    """Run generation benchmark on a model.
    
    Args:
        num_runs: Number of times to run each benchmark (takes best result)
    """
    
    if warmup:
        print("\nWarming up...")
        with torch.inference_mode():
            prompt_embeds = torch.randn(1, 20, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
            past_key_values = None
            current_embeds = prompt_embeds
            for _ in range(30):
                cache_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
                attention_mask = torch.ones((1, cache_len + current_embeds.shape[1]), device=DEVICE, dtype=torch.bool)
                outputs = model(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                current_embeds = torch.randn(1, 1, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
        torch.cuda.synchronize()
    
    print(f"\nBenchmarking {name} ({num_runs} runs each, taking best)...")
    results = []
    
    for seq_len in seq_lengths:
        best_tps = 0
        best_elapsed = float('inf')
        
        for run in range(num_runs):
            prompt_len = 20
            prompt_embeds = torch.randn(1, prompt_len, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            past_key_values = None
            current_embeds = prompt_embeds
            
            with torch.inference_mode():
                for _ in range(seq_len):
                    cache_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
                    total_len = cache_len + current_embeds.shape[1]
                    attention_mask = torch.ones((1, total_len), device=DEVICE, dtype=torch.bool)
                    
                    outputs = model(
                        inputs_embeds=current_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    current_embeds = torch.randn(1, 1, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = seq_len / elapsed
            
            if tokens_per_sec > best_tps:
                best_tps = tokens_per_sec
                best_elapsed = elapsed
        
        results.append((seq_len, best_elapsed, best_tps))
        print(f"  Generated {seq_len} tokens: best = {best_tps:.1f} tok/s ({best_elapsed:.3f}s)")
    
    return results


def benchmark_sdpa():
    """Benchmark with SDPA (PyTorch native scaled dot product attention)."""
    print("=" * 60)
    print("Benchmark 1: SDPA (baseline)")
    print("=" * 60)
    
    model = create_qwen2_model(attn_implementation="sdpa")
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M parameters, dtype: {DTYPE}")
    
    results = run_generation_benchmark(model, "SDPA")
    
    del model
    torch.cuda.empty_cache()
    return results


def benchmark_torch_compile(mode="default", fullgraph=False):
    """Benchmark with torch.compile - the fastest option!
    
    Args:
        mode: Compilation mode - "default", "reduce-overhead", or "max-autotune"
        fullgraph: If True, compile entire model as single graph (can be faster but less compatible)
    """
    print("\n" + "=" * 60)
    print(f"Benchmark: SDPA + torch.compile (mode={mode}, fullgraph={fullgraph})")
    print("=" * 60)
    
    # Enable TF32 for FP32 operations (won't help for FP16 but doesn't hurt)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    model = create_qwen2_model(attn_implementation="sdpa")
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    
    print(f"Compiling model with mode={mode}, fullgraph={fullgraph}...")
    model = torch.compile(model, mode=mode, fullgraph=fullgraph)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M parameters, dtype: {DTYPE}")
    
    # Extended warmup - generate 1200 tokens to cover all benchmark lengths
    print("\nWarming up compiled model (generating 1200 tokens to trigger all compilations)...")
    with torch.inference_mode():
        prompt_embeds = torch.randn(1, 20, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
        past_key_values = None
        current_embeds = prompt_embeds
        for i in range(1200):
            cache_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
            attention_mask = torch.ones((1, cache_len + current_embeds.shape[1]), device=DEVICE, dtype=torch.bool)
            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            current_embeds = torch.randn(1, 1, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
            if (i + 1) % 300 == 0:
                print(f"  Warmup progress: {i + 1}/1200 tokens...")
    torch.cuda.synchronize()
    
    # Second warmup run to ensure everything is compiled
    print("Running second warmup pass...")
    with torch.inference_mode():
        prompt_embeds = torch.randn(1, 20, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
        past_key_values = None
        current_embeds = prompt_embeds
        for _ in range(500):
            cache_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
            attention_mask = torch.ones((1, cache_len + current_embeds.shape[1]), device=DEVICE, dtype=torch.bool)
            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            current_embeds = torch.randn(1, 1, LLM_INPUT_SIZE, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    print("Compilation done!")
    
    results = run_generation_benchmark(model, "torch.compile", warmup=False)
    
    del model
    torch.cuda.empty_cache()
    return results


def benchmark_cosyvoice3_llm():
    """Benchmark full CosyVoice3LM with real weights."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Full CosyVoice3LM (real weights)")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '.')
    from cosyvoice.llm.llm import CosyVoice3LM
    from cosyvoice.utils.common import ras_sampling
    from functools import partial
    
    config = Qwen2Config(**QWEN2_CONFIG)
    
    class Qwen2EncoderLocal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Qwen2ForCausalLM(config)
        
        def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
            from cosyvoice.utils.mask import make_pad_mask
            T = xs.size(1)
            masks = ~make_pad_mask(xs_lens, T)
            outs = self.model(
                inputs_embeds=xs,
                attention_mask=masks,
                output_hidden_states=True,
                return_dict=True,
            )
            return outs.hidden_states[-1], masks.unsqueeze(1)
        
        def forward_one_step(self, xs, masks, cache=None):
            if masks.dim() == 3:
                attention_mask = masks[:, -1, :]
            else:
                attention_mask = masks
            
            outs = self.model.model(
                inputs_embeds=xs,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                past_key_values=cache,
            )
            return outs.last_hidden_state, outs.past_key_values
    
    qwen2_encoder = Qwen2EncoderLocal()
    sampling_fn = partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
    
    llm = CosyVoice3LM(
        llm_input_size=LLM_INPUT_SIZE,
        llm_output_size=LLM_OUTPUT_SIZE,
        speech_token_size=SPEECH_TOKEN_SIZE,
        llm=qwen2_encoder,
        sampling=sampling_fn,
        length_normalized_loss=True,
        lsm_weight=0,
        mix_ratio=[5, 15],
    )
    
    print(f"Loading weights from {LLM_PT_PATH}...")
    state_dict = torch.load(LLM_PT_PATH, map_location='cpu')
    llm.load_state_dict(state_dict, strict=True)
    
    llm.to(DEVICE)
    if DTYPE == torch.float16:
        llm.half()
    llm.eval()
    
    print(f"CosyVoice3LM loaded, dtype: {next(llm.parameters()).dtype}")
    
    # Warmup
    print("\nWarming up...")
    with torch.inference_mode():
        dummy_text = torch.randint(0, 1000, (1, 20), device=DEVICE)
        dummy_prompt_text = torch.randint(0, 1000, (1, 10), device=DEVICE)
        dummy_prompt_speech = torch.randint(0, SPEECH_TOKEN_SIZE, (1, 50), device=DEVICE)
        dummy_embedding = torch.zeros(0, 192, device=DEVICE, dtype=DTYPE)
        
        gen = llm.inference(
            text=dummy_text,
            text_len=torch.tensor([20], dtype=torch.int32, device=DEVICE),
            prompt_text=dummy_prompt_text,
            prompt_text_len=torch.tensor([10], dtype=torch.int32, device=DEVICE),
            prompt_speech_token=dummy_prompt_speech,
            prompt_speech_token_len=torch.tensor([50], dtype=torch.int32, device=DEVICE),
            embedding=dummy_embedding,
            sampling=25,
            max_token_text_ratio=5,
            min_token_text_ratio=2,
        )
        warmup_tokens = list(gen)
    torch.cuda.synchronize()
    print(f"Warmup: generated {len(warmup_tokens)} tokens")
    
    # Benchmark
    print("\nBenchmarking CosyVoice3LM inference...")
    
    for text_len in [20, 50, 100]:
        dummy_text = torch.randint(0, 1000, (1, text_len), device=DEVICE)
        dummy_prompt_text = torch.randint(0, 1000, (1, 10), device=DEVICE)
        dummy_prompt_speech = torch.randint(0, SPEECH_TOKEN_SIZE, (1, 50), device=DEVICE)
        dummy_embedding = torch.zeros(0, 192, device=DEVICE, dtype=DTYPE)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            gen = llm.inference(
                text=dummy_text,
                text_len=torch.tensor([text_len], dtype=torch.int32, device=DEVICE),
                prompt_text=dummy_prompt_text,
                prompt_text_len=torch.tensor([10], dtype=torch.int32, device=DEVICE),
                prompt_speech_token=dummy_prompt_speech,
                prompt_speech_token_len=torch.tensor([50], dtype=torch.int32, device=DEVICE),
                embedding=dummy_embedding,
                sampling=25,
                max_token_text_ratio=10,
                min_token_text_ratio=2,
            )
            tokens = list(gen)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        num_tokens = len(tokens)
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
        
        print(f"  text_len={text_len}: generated {num_tokens} tokens in {elapsed:.3f}s = {tokens_per_sec:.1f} tok/s")
    
    del llm
    torch.cuda.empty_cache()


def check_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_mem:.1f} GB")
        print(f"SDPA available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
    else:
        print("CUDA not available!")


def print_multi_summary(results_dict):
    """Print comparison summary for multiple configurations."""
    print("\n" + "=" * 70)
    print("SUMMARY: All Configurations")
    print("=" * 70)
    
    # Header
    configs = list(results_dict.keys())
    header = f"{'Tokens':<10}"
    for cfg in configs:
        header += f" {cfg:<15}"
    print(header)
    print("-" * (10 + 16 * len(configs)))
    
    # Get sequence lengths from first config
    first_results = list(results_dict.values())[0]
    seq_lengths = [r[0] for r in first_results]
    
    for i, seq_len in enumerate(seq_lengths):
        row = f"{seq_len:<10}"
        for cfg in configs:
            tps = results_dict[cfg][i][2]
            row += f" {tps:<15.1f}"
        print(row)
    
    # Averages
    print("-" * (10 + 16 * len(configs)))
    row = f"{'Average':<10}"
    avgs = {}
    for cfg in configs:
        avg = sum(r[2] for r in results_dict[cfg]) / len(results_dict[cfg])
        avgs[cfg] = avg
        row += f" {avg:<15.1f}"
    print(row)
    
    # Speedups vs SDPA
    if "SDPA" in avgs:
        print("\nSpeedup vs SDPA:")
        baseline = avgs["SDPA"]
        for cfg, avg in avgs.items():
            if cfg != "SDPA":
                speedup = avg / baseline
                print(f"  {cfg}: {speedup:.2f}x")


if __name__ == "__main__":
    check_gpu_info()
    print()
    
    # Enable TF32 globally (helps if using FP32 anywhere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32 enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")
    print(f"Note: TF32 only helps FP32 ops, you're using {DTYPE}")
    print()
    
    all_results = {}
    
    # 1. SDPA baseline
    all_results["SDPA"] = benchmark_sdpa()
    
    # 2. torch.compile with default mode
    all_results["compile"] = benchmark_torch_compile(mode="default")
    
    # Optional: Full CosyVoice3LM benchmark
    try:
        benchmark_cosyvoice3_llm()
    except Exception as e:
        print(f"CosyVoice3LM benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print_multi_summary(all_results)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION: torch.compile(model, mode='default') for ~2x speedup")
    print("Note: TF32 only helps FP32; for FP16 use torch.autocast or model.half()")
    print("=" * 70)
