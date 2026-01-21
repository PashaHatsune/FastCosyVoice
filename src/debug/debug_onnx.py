#!/usr/bin/env python3
"""
Проверка ONNX моделей после обновления TensorRT/ONNX.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party/Matcha-TTS'))

import torch
import torchaudio
import numpy as np
import onnxruntime as ort

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/audio5.wav'

def check_onnx_session(path, name):
    """Проверяет ONNX сессию."""
    print(f"\n{'='*60}")
    print(f"🔍 Checking: {name}")
    print(f"{'='*60}")
    print(f"  Path: {path}")
    
    if not os.path.exists(path):
        print(f"  ❌ File not found!")
        return None
    
    file_size = os.path.getsize(path) / (1024*1024)
    print(f"  Size: {file_size:.2f} MB")
    
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(path, providers=providers)
        print(f"  ✅ Session created successfully")
        print(f"  Providers: {session.get_providers()}")
        
        # Show inputs/outputs
        print(f"  Inputs:")
        for inp in session.get_inputs():
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")
        print(f"  Outputs:")
        for out in session.get_outputs():
            print(f"    - {out.name}: {out.shape} ({out.type})")
        
        return session
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def test_speech_tokenizer(session, audio_path):
    """Тестирует speech_tokenizer на реальном аудио."""
    print(f"\n{'='*60}")
    print(f"🎤 Testing Speech Tokenizer")
    print(f"{'='*60}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    print(f"  Audio: {waveform.shape}, sr={sr}")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        print(f"  Resampled to 16kHz: {waveform.shape}")
    
    # Prepare input
    audio_np = waveform.numpy()
    
    try:
        outputs = session.run(None, {'wav': audio_np})
        tokens = outputs[0]
        print(f"  ✅ Output shape: {tokens.shape}")
        print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
        print(f"  First 20 tokens: {tokens.flatten()[:20].tolist()}")
        return tokens
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def test_campplus(session, audio_path):
    """Тестирует campplus на реальном аудио."""
    print(f"\n{'='*60}")
    print(f"🎤 Testing CamPPlus (Speaker Embedding)")
    print(f"{'='*60}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    print(f"  Audio: {waveform.shape}, sr={sr}")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        print(f"  Resampled to 16kHz: {waveform.shape}")
    
    # CamPPlus expects specific format
    # Usually (batch, time)
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.squeeze(0)  # Remove channel dim
    
    # Add batch dim
    audio_np = audio_np.reshape(1, -1).astype(np.float32)
    
    try:
        outputs = session.run(None, {'fbank': audio_np})
        embedding = outputs[0]
        print(f"  ✅ Output shape: {embedding.shape}")
        print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"  Embedding mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
        
        if np.isnan(embedding).any():
            print(f"  ❌ Contains NaN!")
        elif np.isinf(embedding).any():
            print(f"  ❌ Contains Inf!")
        else:
            print(f"  ✅ No NaN/Inf")
        
        return embedding
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_flow_onnx():
    """Проверяет Flow ONNX."""
    print(f"\n{'='*60}")
    print(f"🔍 Checking Flow ONNX")
    print(f"{'='*60}")
    
    onnx_path = os.path.join(MODEL_DIR, 'flow.decoder.estimator.fp32.onnx')
    
    if not os.path.exists(onnx_path):
        print(f"  ❌ File not found: {onnx_path}")
        return
    
    file_size = os.path.getsize(onnx_path) / (1024*1024)
    print(f"  Path: {onnx_path}")
    print(f"  Size: {file_size:.2f} MB")
    
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✅ ONNX model is valid")
    except Exception as e:
        print(f"  ❌ ONNX validation error: {e}")


def test_pytorch_flow():
    """Тестирует PyTorch Flow напрямую."""
    print(f"\n{'='*60}")
    print(f"🔧 Testing PyTorch Flow (no TensorRT)")
    print(f"{'='*60}")
    
    from hyperpyyaml import load_hyperpyyaml
    
    hyper_yaml_path = os.path.join(MODEL_DIR, 'cosyvoice3.yaml')
    hf_llm_dir = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
    
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})
    
    # Load only Flow
    flow = configs['flow']
    flow.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'flow.pt'), map_location='cpu'), strict=True)
    flow.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow.to(device)
    
    print(f"  Flow loaded on {device}")
    
    # Check weights
    for name, param in list(flow.named_parameters())[:5]:
        print(f"  {name}: shape={param.shape}, nan={torch.isnan(param).any()}, inf={torch.isinf(param).any()}")
    
    # Test with dummy input
    print(f"\n  Testing with dummy input...")
    
    batch_size = 1
    num_tokens = 50
    num_mel_frames = 100
    
    token = torch.randint(0, 6561, (batch_size, num_tokens), dtype=torch.int32).to(device)
    token_len = torch.tensor([num_tokens], dtype=torch.int32).to(device)
    prompt_token = torch.randint(0, 6561, (batch_size, 100), dtype=torch.int32).to(device)
    prompt_token_len = torch.tensor([100], dtype=torch.int32).to(device)
    prompt_feat = torch.randn(batch_size, num_mel_frames, 80).to(device)
    prompt_feat_len = torch.tensor([num_mel_frames], dtype=torch.int32).to(device)
    embedding = torch.randn(batch_size, 192).to(device)
    
    with torch.no_grad():
        try:
            mel, _ = flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
            )
            print(f"  ✅ Output shape: {mel.shape}")
            print(f"  Output range: [{mel.min().item():.4f}, {mel.max().item():.4f}]")
            
            if torch.isnan(mel).any():
                print(f"  ❌ Contains NaN!")
            elif torch.isinf(mel).any():
                print(f"  ❌ Contains Inf!")
            else:
                print(f"  ✅ No NaN/Inf")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("="*70)
    print("🔬 ONNX/TensorRT Debug")
    print("="*70)
    
    # Print versions
    print(f"\n📦 Versions:")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Check ONNX files
    speech_tokenizer_path = os.path.join(MODEL_DIR, 'speech_tokenizer_v3.onnx')
    campplus_path = os.path.join(MODEL_DIR, 'campplus.onnx')
    
    speech_session = check_onnx_session(speech_tokenizer_path, "Speech Tokenizer")
    campplus_session = check_onnx_session(campplus_path, "CamPPlus")
    
    # Test with real audio
    if speech_session and os.path.exists(REFERENCE_AUDIO):
        test_speech_tokenizer(speech_session, REFERENCE_AUDIO)
    
    if campplus_session and os.path.exists(REFERENCE_AUDIO):
        test_campplus(campplus_session, REFERENCE_AUDIO)
    
    # Check Flow ONNX
    test_flow_onnx()
    
    # Test PyTorch Flow
    test_pytorch_flow()
    
    print("\n" + "="*70)
    print("✅ Debug complete!")
    print("="*70)


if __name__ == '__main__':
    main()

