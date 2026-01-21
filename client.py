# tts_zero_shot_client.py
import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"  # поменяй если нужно
SPK_ID = "d3311f8f9ffe"

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
    wav = requests.get(f"{BASE_URL}/api/download/{filename}")
    wav.raise_for_status()

    out_path = Path(out_dir) / filename
    out_path.write_bytes(wav.content)
    print("saved:", out_path)


if __name__ == "__main__":
    tts_zero_shot(
        text="Люди живут по циркадным ритмам — внутренним биологическим часам. Они управляют циклом сна и бодрствования, регулируют выработку гормонов, температуру тела, аппетит и настроение.",
        speed=1.0,
    )
