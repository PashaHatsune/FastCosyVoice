from loguru import logger
from faster_whisper import WhisperModel

model_size = "turbo"
_asr_model = None


class ASR:
    def __init__(
            self
    ) -> None:
        global _asr_model
        if _asr_model is None:
            logger.info(f"Загружается ASR модель: {model_size}")
    
            self._asr_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="auto"
            )
            logger.success("ASR загружен!")

            return _asr_model

    async def transcribe(
            self,
            audio
    ) -> str:
        try:

            segments, info = self._asr_model.transcribe(
                audio,
                beam_size=5

            )

            logger.info(f"Detected language {info.language}")
            text = " ".join(segment.text for segment in segments)
            text = " ".join(text.split())
            logger.success(f"Транскрибированный текст: {text}")

            if text:
                return text
            else: raise("Нет текста.")
    
        except Exception as e:
            logger.exception(e)

# import asyncio
# asr = ASR()
# logger.debug(asyncio.run(asr.transcribe("prompt.wav")))