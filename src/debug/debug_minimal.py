#!/usr/bin/env python3
"""
Минимальный тест для отладки NaN проблемы.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party/Matcha-TTS'))

import torch
import torchaudio
import numpy as np

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/audio5.wav'

def check_tensor(t, name):
    """Проверяет тензор на NaN/Inf."""
    if torch.isnan(t).any():
        print(f"  ❌ {name}: Contains NaN!")
        return False
    if torch.isinf(t).any():
        print(f"  ❌ {name}: Contains Inf!")
        return False
    print(f"  ✅ {name}: OK (min={t.min().item():.4f}, max={t.max().item():.4f})")
    return True

def main():
    print("="*70)
    print("🔬 Minimal Debug Test")
    print("="*70)
    
    # 1. Загрузка оригинальной модели CosyVoice (без FastCosyVoice3)
    print("\n📦 Loading original CosyVoice3 model...")
    
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.model import CosyVoice3Model
    
    hyper_yaml_path = os.path.join(MODEL_DIR, 'cosyvoice3.yaml')
    hf_llm_dir = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
    
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})
    
    model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'])
    model.load(
        os.path.join(MODEL_DIR, 'llm.pt'),
        os.path.join(MODEL_DIR, 'flow.pt'),
        os.path.join(MODEL_DIR, 'hift.pt')
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✅ Model loaded on {device}")
    
    # 2. Проверяем веса модели
    print("\n📊 Checking model weights...")
    
    # LLM
    print("\nLLM weights:")
    check_tensor(model.llm.speech_embedding.weight, "speech_embedding")
    check_tensor(model.llm.llm_decoder.weight, "llm_decoder")
    
    # Flow
    print("\nFlow weights:")
    # Проверяем decoder (DiT)
    for name, param in list(model.flow.decoder.named_parameters())[:3]:
        check_tensor(param, f"flow.decoder.{name}")
    
    # 3. Загрузка frontend
    print("\n📦 Loading frontend...")
    from fastcosyvoice.frontend import CosyVoiceFrontEnd
    
    frontend = CosyVoiceFrontEnd(
        configs['get_tokenizer'],
        configs['feat_extractor'],
        os.path.join(MODEL_DIR, 'campplus.onnx'),
        os.path.join(MODEL_DIR, 'speech_tokenizer_v3.onnx'),
        os.path.join(MODEL_DIR, 'spk2info.pt'),
        configs['allowed_special']
    )
    
    # 4. Подготовка данных
    print("\n📦 Preparing data...")
    
    with open(REFERENCE_AUDIO.replace('.wav', '.txt'), 'r') as f:
        prompt_text = f"You are a helpful assistant.<|endofprompt|>{f.read().strip()}"
    
    prompt_text = frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    test_text = "Привет!"
    test_text = list(frontend.text_normalize(test_text, split=True, text_frontend=True))[0]
    
    model_input = frontend.frontend_zero_shot(
        test_text, prompt_text, REFERENCE_AUDIO, 24000, ''
    )
    
    print("\nModel input:")
    for k, v in model_input.items():
        if isinstance(v, torch.Tensor):
            ok = check_tensor(v, k)
    
    # 5. Тест LLM
    print("\n🔧 Testing LLM...")
    
    text = model_input['text'].to(device)
    text_len = torch.tensor([text.shape[1]], dtype=torch.int32).to(device)
    prompt_text_t = model_input['prompt_text'].to(device)
    prompt_text_len = torch.tensor([prompt_text_t.shape[1]], dtype=torch.int32).to(device)
    llm_prompt_speech_token = model_input['llm_prompt_speech_token'].to(device)
    llm_prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(device)
    llm_embedding = model_input['llm_embedding'].to(device)
    
    tokens = []
    for token in model.llm.inference(
        text=text,
        text_len=text_len,
        prompt_text=prompt_text_t,
        prompt_text_len=prompt_text_len,
        prompt_speech_token=llm_prompt_speech_token,
        prompt_speech_token_len=llm_prompt_speech_token_len,
        embedding=llm_embedding,
    ):
        tokens.append(token)
        if len(tokens) >= 50:  # Limit for testing
            break
    
    print(f"  Generated {len(tokens)} tokens")
    print(f"  First 10: {tokens[:10]}")
    print(f"  Last 10: {tokens[-10:]}")
    print(f"  Min: {min(tokens)}, Max: {max(tokens)}")
    
    if max(tokens) >= model.llm.speech_token_size:
        print(f"  ❌ WARNING: Token {max(tokens)} >= speech_token_size {model.llm.speech_token_size}!")
    else:
        print(f"  ✅ All tokens in valid range [0, {model.llm.speech_token_size})")
    
    # 6. Тест Flow
    print("\n🔧 Testing Flow...")
    
    token_tensor = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0).to(device)
    token_len = torch.tensor([len(tokens)], dtype=torch.int32).to(device)
    
    flow_prompt_token = model_input['flow_prompt_speech_token'].to(device)
    flow_prompt_token_len = torch.tensor([flow_prompt_token.shape[1]], dtype=torch.int32).to(device)
    prompt_feat = model_input['prompt_speech_feat'].to(device)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(device)
    flow_embedding = model_input['flow_embedding'].to(device)
    
    print(f"  token_tensor: shape={token_tensor.shape}, min={token_tensor.min()}, max={token_tensor.max()}")
    print(f"  flow_prompt_token: shape={flow_prompt_token.shape}")
    print(f"  prompt_feat: shape={prompt_feat.shape}")
    
    with torch.no_grad():
        tts_mel, _ = model.flow.inference(
            token=token_tensor,
            token_len=token_len,
            prompt_token=flow_prompt_token,
            prompt_token_len=flow_prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=flow_embedding,
            streaming=False,
            finalize=True
        )
    
    print(f"\nFlow output mel:")
    check_tensor(tts_mel, "tts_mel")
    print(f"  Shape: {tts_mel.shape}")
    
    # 7. Тест Hift
    print("\n🔧 Testing Hift...")
    
    with torch.no_grad():
        tts_speech, _ = model.hift.inference(speech_feat=tts_mel)
    
    print(f"\nHift output audio:")
    check_tensor(tts_speech, "tts_speech")
    print(f"  Shape: {tts_speech.shape}")
    
    if not torch.isnan(tts_speech).any():
        torchaudio.save('debug_minimal.wav', tts_speech.cpu(), 24000)
        print(f"\n💾 Saved: debug_minimal.wav")
        
        rms = torch.sqrt((tts_speech ** 2).mean()).item()
        print(f"  RMS: {rms:.6f}")
        if rms > 0.01:
            print("  ✅ Audio has sound!")
        else:
            print("  ⚠️ Audio is very quiet")
    else:
        print("\n❌ Cannot save - audio contains NaN")
    
    print("\n" + "="*70)
    print("✅ Debug complete!")
    print("="*70)


if __name__ == '__main__':
    main()

