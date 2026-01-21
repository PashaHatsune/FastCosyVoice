#!/usr/bin/env python3
"""
Тест TRT-LLM интеграции для CosyVoice3.

Этот скрипт:
1. Конвертирует CosyVoice3 в merged HuggingFace модель
2. Собирает TRT-LLM engine
3. Синтезирует тестовую речь
4. Сохраняет результат для прослушивания

Usage:
    python test_trt.py
"""
import os
import sys
import time
import logging

# Добавляем пути
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party/Matcha-TTS'))

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/audio5.wav'
OUTPUT_DIR = 'output_trt_test'
DTYPE = 'bfloat16'

# Тексты для синтеза
TEST_TEXTS = [
    "Привет! Это тест TensorRT-LLM интеграции для CosyVoice3.",
    "Если вы слышите этот текст, значит всё работает корректно!",
    "Теперь модель использует полное TRT-LLM ускорение с правильной архитектурой.",
]

INSTRUCTION = "You are a helpful assistant."


def get_gpu_memory():
    """Возвращает использование GPU памяти."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved


def load_prompt_text(audio_path: str) -> str:
    """Загружает транскрипцию из txt файла."""
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    return f"{INSTRUCTION}<|endofprompt|>{transcription}"


def test_trt_llm():
    """Основной тест TRT-LLM интеграции."""
    print("=" * 70)
    print("🚀 CosyVoice3 TRT-LLM Integration Test")
    print("=" * 70)
    
    # Проверяем файлы
    if not os.path.exists(MODEL_DIR):
        logger.error(f"Model not found: {MODEL_DIR}")
        return False
    
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}")
        return False
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Загружаем prompt
    prompt_text = load_prompt_text(REFERENCE_AUDIO)
    print(f"\n📝 Reference: {REFERENCE_AUDIO}")
    
    # =========================================================================
    # Шаг 1: Загрузка модели с TRT-LLM
    # =========================================================================
    print("\n" + "=" * 70)
    print("📦 Step 1: Loading FastCosyVoice3 with TRT-LLM")
    print("=" * 70)
    
    alloc, res = get_gpu_memory()
    print(f"GPU Memory [before]: {alloc:.2f} GB allocated, {res:.2f} GB reserved")
    
    load_start = time.time()
    
    from fastcosyvoice import FastCosyVoice3
    
    cosyvoice = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_vllm=False,
        load_trt=True,           # TensorRT for Flow
        load_trt_llm=True,       # TensorRT-LLM for LLM
        trt_llm_dtype=DTYPE,
    )
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    alloc, res = get_gpu_memory()
    print(f"GPU Memory [after load]: {alloc:.2f} GB allocated, {res:.2f} GB reserved")
    
    # Проверяем статус TRT-LLM
    if cosyvoice.trt_llm_loaded:
        print("✅ TRT-LLM: LOADED")
        print(f"   Speech token offset: {cosyvoice.speech_token_offset}")
        print(f"   Speech token size: {cosyvoice.speech_token_size}")
    else:
        print("❌ TRT-LLM: NOT LOADED (using PyTorch)")
        logger.warning("TRT-LLM not loaded! Check logs above for errors.")
    
    # =========================================================================
    # Шаг 2: Подготовка спикера
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎤 Step 2: Preparing speaker embeddings")
    print("=" * 70)
    
    spk_id = "test_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Speaker embeddings prepared in {embed_time:.3f}s")
    
    # =========================================================================
    # Шаг 3: Прогрев
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔥 Step 3: Warmup")
    print("=" * 70)
    
    warmup_start = time.time()
    # Use a longer warmup text to better match real usage
    warmup_text = "Это тест прогрева модели для синтеза речи."
    
    for chunk in cosyvoice.inference_zero_shot_stream(
        tts_text=warmup_text,
        prompt_text=prompt_text,
        prompt_wav=REFERENCE_AUDIO,
        zero_shot_spk_id=spk_id,
    ):
        pass  # Просто прогреваем
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Clear VRAM after warmup
        torch.cuda.empty_cache()
    
    warmup_time = time.time() - warmup_start
    alloc, res = get_gpu_memory()
    print(f"✅ Warmup complete in {warmup_time:.2f}s")
    print(f"GPU Memory [after warmup+cleanup]: {alloc:.2f} GB allocated, {res:.2f} GB reserved")
    
    # =========================================================================
    # Шаг 4: Синтез тестовых текстов
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎵 Step 4: Synthesizing test audio")
    print("=" * 70)
    
    all_results = []
    
    for idx, text in enumerate(TEST_TEXTS, 1):
        print(f"\n--- Text {idx}/{len(TEST_TEXTS)} ---")
        print(f"📝 {text[:60]}{'...' if len(text) > 60 else ''}")
        
        audio_chunks = []
        chunk_count = 0
        
        synth_start = time.time()
        first_chunk_time = None
        
        for chunk in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
        ):
            chunk_count += 1
            if first_chunk_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                first_chunk_time = time.time() - synth_start
            
            audio_chunks.append(chunk['tts_speech'])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - synth_start
        
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=1)
            audio_duration = full_audio.shape[1] / cosyvoice.sample_rate
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
            
            # Сохраняем
            output_path = os.path.join(OUTPUT_DIR, f'trt_test_{idx:02d}.wav')
            torchaudio.save(output_path, full_audio, cosyvoice.sample_rate)
            
            print(f"💾 Saved: {output_path}")
            print(f"   TTFB: {first_chunk_time:.3f}s")
            print(f"   Duration: {audio_duration:.2f}s")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   RTF: {rtf:.3f}")
            print(f"   Chunks: {chunk_count}")
            
            if rtf < 1.0:
                print(f"   ✅ {1/rtf:.1f}x faster than realtime")
            else:
                print(f"   ⚠️ {rtf:.1f}x slower than realtime")
            
            all_results.append({
                'text': text,
                'ttfb': first_chunk_time,
                'duration': audio_duration,
                'total_time': total_time,
                'rtf': rtf,
                'chunks': chunk_count,
                'output': output_path,
            })
        else:
            print("   ❌ No audio generated!")
            all_results.append({
                'text': text,
                'error': 'No audio generated',
            })
    
    # =========================================================================
    # Итоги
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    successful = [r for r in all_results if 'error' not in r]
    
    if successful:
        avg_ttfb = sum(r['ttfb'] for r in successful) / len(successful)
        avg_rtf = sum(r['rtf'] for r in successful) / len(successful)
        total_duration = sum(r['duration'] for r in successful)
        total_time = sum(r['total_time'] for r in successful)
        
        print(f"TRT-LLM Status: {'✅ ENABLED' if cosyvoice.trt_llm_loaded else '❌ DISABLED'}")
        print(f"Tests passed: {len(successful)}/{len(all_results)}")
        print(f"Average TTFB: {avg_ttfb:.3f}s")
        print(f"Average RTF: {avg_rtf:.3f}")
        print(f"Total audio: {total_duration:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        
        if avg_rtf < 1.0:
            print(f"\n✅ Overall: {1/avg_rtf:.1f}x faster than realtime!")
        
        print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        for r in successful:
            print(f"  - {r['output']}")
        
        return True
    else:
        print("❌ All tests failed!")
        return False


def main():
    try:
        success = test_trt_llm()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ TEST PASSED!")
            print("=" * 70)
            print("\n🎧 Послушай аудио файлы в папке output_trt_test/")
        else:
            print("❌ TEST FAILED!")
            print("=" * 70)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print("\n" + "=" * 70)
        print("❌ TEST FAILED WITH EXCEPTION!")
        print("=" * 70)
        raise


if __name__ == '__main__':
    main()

