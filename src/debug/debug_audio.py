#!/usr/bin/env python3
"""
Диагностика проблемы с пустым аудио в TRT-LLM.

Сравнивает:
1. PyTorch LLM -> Flow -> Hift (оригинал)
2. TRT-LLM -> Flow -> Hift (наша реализация)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party/Matcha-TTS'))

import torch
import torchaudio
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/audio5.wav'
TEST_TEXT = "Привет мир!"
INSTRUCTION = "You are a helpful assistant."


def analyze_audio(audio: torch.Tensor, name: str):
    """Анализирует аудио тензор."""
    print(f"\n{'='*60}")
    print(f"📊 Audio Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {audio.shape}")
    print(f"  Dtype: {audio.dtype}")
    print(f"  Device: {audio.device}")
    print(f"  Min: {audio.min().item():.6f}")
    print(f"  Max: {audio.max().item():.6f}")
    print(f"  Mean: {audio.mean().item():.6f}")
    print(f"  Std: {audio.std().item():.6f}")
    print(f"  Abs Max: {audio.abs().max().item():.6f}")
    
    # Check if mostly zeros
    zeros_ratio = (audio.abs() < 1e-6).float().mean().item()
    print(f"  Zero ratio: {zeros_ratio*100:.2f}%")
    
    # RMS
    rms = torch.sqrt((audio ** 2).mean()).item()
    print(f"  RMS: {rms:.6f}")
    
    if rms < 0.001:
        print("  ⚠️ WARNING: Audio is essentially silent!")
    elif rms < 0.01:
        print("  ⚠️ WARNING: Audio is very quiet!")
    else:
        print("  ✅ Audio has reasonable amplitude")
    
    return {
        'min': audio.min().item(),
        'max': audio.max().item(),
        'mean': audio.mean().item(),
        'std': audio.std().item(),
        'rms': rms,
        'zeros_ratio': zeros_ratio
    }


def analyze_mel(mel: torch.Tensor, name: str):
    """Анализирует mel-spectrogram."""
    print(f"\n{'='*60}")
    print(f"📊 Mel Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {mel.shape}")
    print(f"  Dtype: {mel.dtype}")
    print(f"  Min: {mel.min().item():.6f}")
    print(f"  Max: {mel.max().item():.6f}")
    print(f"  Mean: {mel.mean().item():.6f}")
    print(f"  Std: {mel.std().item():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(mel).any():
        print("  ⚠️ WARNING: Contains NaN!")
    if torch.isinf(mel).any():
        print("  ⚠️ WARNING: Contains Inf!")


def analyze_tokens(tokens: list, name: str):
    """Анализирует список токенов."""
    print(f"\n{'='*60}")
    print(f"📊 Token Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Count: {len(tokens)}")
    print(f"  Unique: {len(set(tokens))}")
    print(f"  Min: {min(tokens) if tokens else 'N/A'}")
    print(f"  Max: {max(tokens) if tokens else 'N/A'}")
    print(f"  First 20: {tokens[:20]}")
    print(f"  Last 20: {tokens[-20:]}")
    
    # Distribution
    from collections import Counter
    counter = Counter(tokens)
    most_common = counter.most_common(10)
    print(f"  Most common: {most_common}")


def test_pytorch_inference():
    """Тест с PyTorch LLM (без TRT-LLM)."""
    print("\n" + "="*70)
    print("🔬 TEST 1: PyTorch LLM (no TRT-LLM)")
    print("="*70)
    
    from fastcosyvoice import FastCosyVoice3
    
    model = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=True,
        load_trt_llm=False,  # PyTorch LLM
    )
    
    # Load prompt
    with open(REFERENCE_AUDIO.replace('.wav', '.txt'), 'r') as f:
        prompt_text = f"{INSTRUCTION}<|endofprompt|>{f.read().strip()}"
    
    model.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, 'test')
    
    print(f"\nSynthesizing: {TEST_TEXT}")
    
    chunks = []
    for chunk in model.inference_zero_shot_stream(
        TEST_TEXT, prompt_text, REFERENCE_AUDIO, 'test'
    ):
        chunks.append(chunk['tts_speech'])
    
    if chunks:
        audio = torch.cat(chunks, dim=1)
        analyze_audio(audio, "PyTorch LLM Output")
        torchaudio.save('debug_pytorch.wav', audio.cpu(), model.sample_rate)
        print(f"\n💾 Saved: debug_pytorch.wav")
        return audio
    else:
        print("❌ No audio generated!")
        return None


def test_trt_llm_inference():
    """Тест с TRT-LLM."""
    print("\n" + "="*70)
    print("🔬 TEST 2: TRT-LLM")
    print("="*70)
    
    from fastcosyvoice import FastCosyVoice3
    
    model = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=True,
        load_trt_llm=True,
    )
    
    # Load prompt
    with open(REFERENCE_AUDIO.replace('.wav', '.txt'), 'r') as f:
        prompt_text = f"{INSTRUCTION}<|endofprompt|>{f.read().strip()}"
    
    model.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, 'test')
    
    print(f"\nSynthesizing: {TEST_TEXT}")
    
    chunks = []
    for chunk in model.inference_zero_shot_stream(
        TEST_TEXT, prompt_text, REFERENCE_AUDIO, 'test'
    ):
        chunks.append(chunk['tts_speech'])
    
    if chunks:
        audio = torch.cat(chunks, dim=1)
        analyze_audio(audio, "TRT-LLM Output")
        torchaudio.save('debug_trt_llm.wav', audio.cpu(), model.sample_rate)
        print(f"\n💾 Saved: debug_trt_llm.wav")
        return audio
    else:
        print("❌ No audio generated!")
        return None


def test_flow_directly():
    """Тестирует Flow напрямую с известными токенами."""
    print("\n" + "="*70)
    print("🔬 TEST 3: Flow Direct Test")
    print("="*70)
    
    from fastcosyvoice import FastCosyVoice3
    
    model = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=True,
        load_trt_llm=True,
    )
    
    # Load prompt
    with open(REFERENCE_AUDIO.replace('.wav', '.txt'), 'r') as f:
        prompt_text = f"{INSTRUCTION}<|endofprompt|>{f.read().strip()}"
    
    # Get frontend data
    model_input = model.frontend.frontend_zero_shot(
        TEST_TEXT, prompt_text, REFERENCE_AUDIO, model.sample_rate, ''
    )
    
    print("\n📦 Model Input Keys:")
    for k, v in model_input.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    
    # Extract prompt speech tokens for TRT-LLM
    prompt_speech_tokens = model_input['llm_prompt_speech_token'].squeeze(0).tolist()
    analyze_tokens(prompt_speech_tokens, "Prompt Speech Tokens")
    
    # Run TRT-LLM to get generated tokens
    text_chunk = list(model.frontend.text_normalize(TEST_TEXT, split=True, text_frontend=True))[0]
    prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    
    print(f"\nRunning TRT-LLM inference...")
    all_tokens = list(model._run_trt_llm_inference(
        text=text_chunk,
        prompt_text=prompt_text_norm,
        prompt_speech_tokens=prompt_speech_tokens,
    ))
    
    analyze_tokens(all_tokens, "Generated Speech Tokens")
    
    # Now test Flow directly with these tokens
    print("\n🔧 Testing Flow with generated tokens...")
    
    dtype = torch.float16 if model.fp16 else torch.float32
    device = model.model.device
    
    flow_prompt_token = model_input['flow_prompt_speech_token'].to(device)
    flow_prompt_token_len = torch.tensor([flow_prompt_token.shape[1]], dtype=torch.int32).to(device)
    prompt_feat = model_input['prompt_speech_feat'].to(device, dtype=dtype)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(device)
    flow_embedding = model_input['flow_embedding'].to(device, dtype=dtype)
    
    print(f"\nFlow inputs:")
    print(f"  flow_prompt_token: {flow_prompt_token.shape}")
    print(f"  prompt_feat: {prompt_feat.shape}")
    print(f"  flow_embedding: {flow_embedding.shape}")
    print(f"  generated tokens: {len(all_tokens)}")
    
    # Create token tensor
    token_tensor = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0).to(device)
    token_len = torch.tensor([len(all_tokens)], dtype=torch.int32).to(device)
    
    print(f"  token_tensor: {token_tensor.shape}, min={token_tensor.min()}, max={token_tensor.max()}")
    
    # Run Flow
    with torch.no_grad():
        tts_mel, _ = model.model.flow.inference(
            token=token_tensor,
            token_len=token_len,
            prompt_token=flow_prompt_token,
            prompt_token_len=flow_prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=flow_embedding
        )
    
    analyze_mel(tts_mel, "Flow Output Mel")
    
    # Run Hift
    print("\n🔧 Testing Hift...")
    with torch.no_grad():
        tts_speech, _ = model.model.hift.inference(
            speech_feat=tts_mel.float(),
            finalize=True
        )
    
    analyze_audio(tts_speech, "Hift Output Audio")
    torchaudio.save('debug_flow_direct.wav', tts_speech.cpu(), model.sample_rate)
    print(f"\n💾 Saved: debug_flow_direct.wav")
    
    return tts_speech


def main():
    print("="*70)
    print("🔬 Audio Debugging Script")
    print("="*70)
    
    # Test 1: PyTorch LLM (reference)
    try:
        pytorch_audio = test_pytorch_inference()
    except Exception as e:
        logger.error(f"PyTorch test failed: {e}", exc_info=True)
        pytorch_audio = None
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test 2: TRT-LLM 
    try:
        trt_audio = test_trt_llm_inference()
    except Exception as e:
        logger.error(f"TRT-LLM test failed: {e}", exc_info=True)
        trt_audio = None
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test 3: Flow direct
    try:
        flow_audio = test_flow_directly()
    except Exception as e:
        logger.error(f"Flow direct test failed: {e}", exc_info=True)
        flow_audio = None
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if pytorch_audio is not None:
        rms = torch.sqrt((pytorch_audio ** 2).mean()).item()
        print(f"PyTorch LLM: RMS={rms:.6f} {'✅' if rms > 0.01 else '❌'}")
    
    if trt_audio is not None:
        rms = torch.sqrt((trt_audio ** 2).mean()).item()
        print(f"TRT-LLM: RMS={rms:.6f} {'✅' if rms > 0.01 else '❌'}")
    
    if flow_audio is not None:
        rms = torch.sqrt((flow_audio ** 2).mean()).item()
        print(f"Flow Direct: RMS={rms:.6f} {'✅' if rms > 0.01 else '❌'}")
    
    print("\n🎧 Check the generated .wav files!")


if __name__ == '__main__':
    main()

