# Some notes

# Basic export
python cosyvoice/bin/export_onnx_optimized.py \
    --model_dir pretrained_models/Fun-CosyVoice3-0.5B

# Optimized export - no benefits from flag `--optimize` after tests
python cosyvoice/bin/export_onnx_optimized.py \
    --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
    --optimize --fp16 --trt

# Debug
```
python debug_trt.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B --test_tts --use_trt --stream --prompt_wav refs/audio.wav --tts_text "Проверка звука TRT стриминг"
```