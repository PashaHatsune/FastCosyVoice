#!/usr/bin/env python3
"""
Debug script to test TensorRT flow estimator independently.
This helps identify where the problem is in the TRT pipeline.
"""

import os
import sys
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def test_pytorch_vs_onnx_vs_trt(model_dir: str):
    """
    Test PyTorch, ONNX Runtime, and TensorRT outputs for the flow estimator.
    """
    print("=" * 60)
    print("DEBUG: Testing Flow Estimator (PyTorch vs ONNX vs TRT)")
    print("=" * 60)
    
    # Load model without TRT first
    print("\n[1] Loading model WITHOUT TRT...")
    model_pytorch = CosyVoice3(model_dir, load_trt=False, fp16=False)
    
    # Get the PyTorch estimator
    estimator_pytorch = model_pytorch.model.flow.decoder.estimator
    estimator_pytorch.eval()
    device = model_pytorch.model.device
    
    print(f"Device: {device}")
    print(f"Estimator type: {type(estimator_pytorch).__name__}")
    print(f"out_channels: {estimator_pytorch.out_channels}")
    
    # Create test inputs
    batch_size = 2  # CFG uses batch=2
    seq_len = 100   # Typical length
    out_channels = 80
    
    print(f"\n[2] Creating test inputs: batch={batch_size}, seq_len={seq_len}, channels={out_channels}")
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
    mask = torch.ones(batch_size, 1, seq_len, device=device, dtype=torch.float32)
    mu = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    spks = torch.randn(batch_size, out_channels, device=device, dtype=torch.float32)
    cond = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
    
    # Test PyTorch
    print("\n[3] Testing PyTorch inference (streaming=True)...")
    with torch.no_grad():
        output_pytorch = estimator_pytorch(x, mask, mu, t, spks, cond, streaming=True)
    
    print(f"PyTorch output shape: {output_pytorch.shape}")
    print(f"PyTorch output stats: min={output_pytorch.min():.4f}, max={output_pytorch.max():.4f}, mean={output_pytorch.mean():.4f}, std={output_pytorch.std():.4f}")
    print(f"PyTorch has NaN: {torch.isnan(output_pytorch).any()}")
    print(f"PyTorch has Inf: {torch.isinf(output_pytorch).any()}")
    
    # Test ONNX Runtime
    onnx_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
    if os.path.exists(onnx_path):
        print(f"\n[4] Testing ONNX Runtime inference...")
        print(f"ONNX path: {onnx_path}")
        
        import onnxruntime
        
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=providers)
        
        print(f"ONNX providers: {ort_session.get_providers()}")
        print(f"ONNX inputs: {[i.name for i in ort_session.get_inputs()]}")
        print(f"ONNX outputs: {[o.name for o in ort_session.get_outputs()]}")
        
        ort_inputs = {
            'x': x.cpu().numpy(),
            'mask': mask.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            't': t.cpu().numpy(),
            'spks': spks.cpu().numpy(),
            'cond': cond.cpu().numpy()
        }
        
        output_onnx = ort_session.run(None, ort_inputs)[0]
        output_onnx_tensor = torch.from_numpy(output_onnx).to(device)
        
        print(f"ONNX output shape: {output_onnx_tensor.shape}")
        print(f"ONNX output stats: min={output_onnx_tensor.min():.4f}, max={output_onnx_tensor.max():.4f}, mean={output_onnx_tensor.mean():.4f}, std={output_onnx_tensor.std():.4f}")
        print(f"ONNX has NaN: {torch.isnan(output_onnx_tensor).any()}")
        print(f"ONNX has Inf: {torch.isinf(output_onnx_tensor).any()}")
        
        # Compare PyTorch vs ONNX
        diff_onnx = (output_pytorch - output_onnx_tensor).abs()
        print(f"\nPyTorch vs ONNX diff: max={diff_onnx.max():.6f}, mean={diff_onnx.mean():.6f}")
    else:
        print(f"\n[4] ONNX not found: {onnx_path}")
        print("Run export first: python cosyvoice/bin/export_onnx_optimized.py --model_dir ...")
    
    # Test TensorRT
    trt_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.mygpu.plan')
    if os.path.exists(trt_path):
        print(f"\n[5] Testing TensorRT inference...")
        print(f"TRT path: {trt_path}")
        
        # Load model with TRT
        del model_pytorch
        torch.cuda.empty_cache()
        
        model_trt = CosyVoice3(model_dir, load_trt=True, fp16=False)
        
        # Get the TRT estimator wrapper
        estimator_trt = model_trt.model.flow.decoder.estimator
        print(f"TRT estimator type: {type(estimator_trt).__name__}")
        
        # Recreate inputs (same seed for reproducibility)
        torch.manual_seed(42)
        x = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        mask = torch.ones(batch_size, 1, seq_len, device=device, dtype=torch.float32)
        mu = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        t = torch.rand(batch_size, device=device, dtype=torch.float32)
        spks = torch.randn(batch_size, out_channels, device=device, dtype=torch.float32)
        cond = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        
        # Test TRT via forward_estimator
        print("Running TRT inference via forward_estimator...")
        
        # Make copies because TRT might modify in-place
        x_trt = x.clone()
        
        output_trt = model_trt.model.flow.decoder.forward_estimator(
            x_trt, mask, mu, t, spks, cond, streaming=True
        )
        
        print(f"TRT output shape: {output_trt.shape}")
        print(f"TRT output stats: min={output_trt.min():.4f}, max={output_trt.max():.4f}, mean={output_trt.mean():.4f}, std={output_trt.std():.4f}")
        print(f"TRT has NaN: {torch.isnan(output_trt).any()}")
        print(f"TRT has Inf: {torch.isinf(output_trt).any()}")
        
        # Check if output is same as input (which would be wrong!)
        if torch.allclose(output_trt, x, atol=1e-3):
            print("\n⚠️  WARNING: TRT output is SAME as input! TRT engine might not be running properly.")
        
        # Compare with PyTorch
        # Reload PyTorch model for comparison
        model_pytorch2 = CosyVoice3(model_dir, load_trt=False, fp16=False)
        estimator_pytorch2 = model_pytorch2.model.flow.decoder.estimator
        
        torch.manual_seed(42)
        x2 = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        mask2 = torch.ones(batch_size, 1, seq_len, device=device, dtype=torch.float32)
        mu2 = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        t2 = torch.rand(batch_size, device=device, dtype=torch.float32)
        spks2 = torch.randn(batch_size, out_channels, device=device, dtype=torch.float32)
        cond2 = torch.randn(batch_size, out_channels, seq_len, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output_pytorch2 = estimator_pytorch2(x2, mask2, mu2, t2, spks2, cond2, streaming=True)
        
        diff_trt = (output_pytorch2 - output_trt).abs()
        print(f"\nPyTorch vs TRT diff: max={diff_trt.max():.6f}, mean={diff_trt.mean():.6f}")
        
        if diff_trt.max() > 1.0:
            print("\n❌ LARGE DIFFERENCE between PyTorch and TRT!")
            print("This indicates the TRT engine is not producing correct outputs.")
        elif diff_trt.max() > 0.1:
            print("\n⚠️  Moderate difference between PyTorch and TRT (may be acceptable for fp32)")
        else:
            print("\n✅ TRT output matches PyTorch closely!")
            
    else:
        print(f"\n[5] TRT engine not found: {trt_path}")
        print("It will be created automatically when running with load_trt=True")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


def test_full_tts_pipeline(
    model_dir: str,
    use_trt: bool = False,
    prompt_wav: str | None = None,
    prompt_text: str | None = None,
    tts_text: str = "Привет, это тест.",
    stream: bool = False,
):
    """
    Test the full TTS pipeline and check intermediate outputs.
    """
    print("\n" + "=" * 60)
    print(f"DEBUG: Full TTS Pipeline (TRT={use_trt})")
    print("=" * 60)
    
    model = CosyVoice3(model_dir, load_trt=use_trt, fp16=True)

    # Resolve prompt wav path (prefer explicit CLI, then refs/, then asset/)
    prompt_wav_path = prompt_wav
    if prompt_wav_path is None:
        candidate = os.path.join("refs", "audio5.wav")
        if os.path.exists(candidate):
            prompt_wav_path = candidate
    if prompt_wav_path is None:
        asset_dir = os.path.join(model_dir, "asset")
        if os.path.exists(asset_dir):
            for f in os.listdir(asset_dir):
                if f.endswith(".wav"):
                    prompt_wav_path = os.path.join(asset_dir, f)
                    break

    if prompt_wav_path is None or not os.path.exists(prompt_wav_path):
        print("No prompt wav found. Provide --prompt_wav (e.g. refs/audio5.wav).")
        return

    # Resolve prompt text (prefer explicit, then sidecar .txt, else fallback)
    if prompt_text is None:
        sidecar = prompt_wav_path.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        else:
            prompt_text = "Пример голоса."

    print(f"Using prompt wav: {prompt_wav_path}")
    print(f"Prompt text: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
    print(f"TTS text: {tts_text}")
    print(f"stream={stream}, TRT={use_trt}")

    print("\nRunning TTS...")

    import torchaudio

    chunks = []
    for i, result in enumerate(model.inference_zero_shot(
        tts_text=tts_text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav_path,
        stream=stream,
    )):
        tts_speech = result["tts_speech"]
        chunks.append(tts_speech)
        print(f"Chunk {i}: shape={tts_speech.shape}")
        print(f"  stats: min={tts_speech.min():.4f}, max={tts_speech.max():.4f}, mean={tts_speech.mean():.4f}")
        print(f"  has NaN: {torch.isnan(tts_speech).any()}")
        print(f"  has Inf: {torch.isinf(tts_speech).any()}")
        print(f"  all zeros: {(tts_speech == 0).all()}")
        print(f"  near zero: {(tts_speech.abs() < 1e-6).float().mean():.2%}")

    if not chunks:
        print("❌ No audio chunks produced.")
        return

    full = torch.cat(chunks, dim=1)
    output_path = f"debug_output_trt_{use_trt}_stream{stream}.wav"
    torchaudio.save(output_path, full, model.sample_rate)
    print(f"\nSaved combined audio: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug TRT flow estimator')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--test_estimator', action='store_true', help='Test estimator directly')
    parser.add_argument('--test_tts', action='store_true', help='Test full TTS pipeline')
    parser.add_argument('--use_trt', action='store_true', help='Use TRT for TTS test')
    parser.add_argument('--build_trt', action='store_true',
                        help='Build TensorRT engine (flow.decoder.estimator.*.plan) before running tests')
    parser.add_argument('--prompt_wav', type=str, default=None,
                        help='Path to reference prompt wav (default: refs/audio5.wav if exists)')
    parser.add_argument('--prompt_text', type=str, default=None,
                        help='Prompt transcript text (default: reads sidecar .txt if exists)')
    parser.add_argument('--tts_text', type=str, default="Привет, это тест.",
                        help='Text to synthesize')
    parser.add_argument('--stream', action='store_true',
                        help='Run TTS in streaming mode (stream=True)')
    args = parser.parse_args()
    
    if args.build_trt:
        # This will export ONNX if missing and build the TRT engine if missing.
        print("\n[0] Building TRT engine via CosyVoice3(load_trt=True)...")
        _ = CosyVoice3(args.model_dir, load_trt=True, fp16=False)
        print("[0] TRT engine build step complete.")

    if args.test_estimator or (not args.test_tts):
        test_pytorch_vs_onnx_vs_trt(args.model_dir)
    
    if args.test_tts:
        test_full_tts_pipeline(
            args.model_dir,
            use_trt=args.use_trt,
            prompt_wav=args.prompt_wav,
            prompt_text=args.prompt_text,
            tts_text=args.tts_text,
            stream=args.stream,
        )

