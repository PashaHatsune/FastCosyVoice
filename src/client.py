# tts_zero_shot_client.py
import time
import requests
from pathlib import Path
import uuid

BASE_URL = "http://127.0.0.1:8000"  # поменяй если нужно
SPK_ID = "kind_mita_neutral"

def tts_zero_shot(
    text: str,
    prompt_text: str = "",
    prompt_wav: str | None = None,
    speed: float = 1.0,
    out_dir: str = "outputs"
):
    files = {}
    if prompt_wav:
        files["prompt_wav"] = open(prompt_wav, "rb")

    data = {
        "text": text,
        "mode": "zero_shot",
        "spk_id": SPK_ID,
        "speed": speed, 
        "seed": 9494
    }

    r = requests.post(f"{BASE_URL}/api/tts/async", data=data, files=files)
    r.raise_for_status()
    task_id = r.json()["task_id"]

    print("task_id:", task_id)

    while True:
        time.sleep(0.5)
        s = requests.get(f"{BASE_URL}/api/task/{task_id}")
        s.raise_for_status()
        status = s.json()

        print(status)

        if status["status"] == "completed":
            filename = status["output_file"]
            break
        if status["status"] == "failed":
            raise RuntimeError(status.get("error"))

    Path(out_dir).mkdir(exist_ok=True)
    wav = requests.get(f"{BASE_URL}/api/download/{task_id}")
    wav.raise_for_status()


    out_path = Path(out_dir) / f"{uuid.uuid4()}.wav"
    out_path.write_bytes(wav.content)
    print("saved:", out_path)


if __name__ == "__main__":
    tts_zero_shot(
        text="Я тебя люблю! А ты, меня любишь?",
        speed=1.0,
    )


# # create_voice_client.py
# import requests
# from pathlib import Path

# URL = "http://127.0.0.1:8000/v1/voices/create"

# audio_path = Path("prompt.mp3")  # путь к wav
# name = "CrazyMita"
# text = "Именно поэтому у меня всё выходит. Опа! У меня больше снежков. Догоняй меня. Ха! Один-один. Ты прирождённый пингвин. Я так рада. Твоя взяла. Ох, ты нашёл её. Круто. Ура! Теперь у тебя достаточно денег, чтобы купить мне телевизор."  # можешь оставить пустым, тогда ASR сам распознает

# with audio_path.open("rb") as f:
#     files = {
#         "audio": ("test.wav", f, "audio/wav")
#     }
#     data = {
#         "name": name,
#         "text": text
#     }

#     r = requests.post(URL, files=files, data=data)
#     r.raise_for_status()
#     print(r.json())
