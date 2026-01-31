
    # def list_available_spks(self):
    #     spks = list(self.frontend.spk2info.keys())
    #     return spks

    # def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
    #     assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
    #     model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_wav, self.sample_rate, '')
    #     del model_input['text']
    #     del model_input['text_len']
    #     self.frontend.spk2info[zero_shot_spk_id] = model_input
    #     return True

    # def save_spkinfo(self):
    #     torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))


import torch
import io
import soundfile as sf
from pathlib import Path
from cosyvoice.cli.cosyvoice import CosyVoice3

AUDIO_PTH = "prompt.mp3"
TEXT = '''Именно поэтому у меня всё выходит. Опа! У меня больше снежков. Догоняй меня. Один-один. Ты прирождённый пингвин. Я так рада. Твоя взяла. О-о, Ты нашёл её. Круто. Ура! Теперь у тебя достаточно денег, чтобы купить мне телевизор.'''

model = CosyVoice3(
    model_dir="pretrained_models/Fun-CosyVoice3-0.5B"
)

def add_speaker():

    model.add_zero_shot_spk(
        prompt_wav=AUDIO_PTH,
        prompt_text=TEXT,
        zero_shot_spk_id="mita_1"
    )
    model.save_spkinfo()
    print("DONE")

    print(model.list_available_spks())


# add_speaker()


def inference():

    output = model.inference_zero_shot(
        tts_text="Привет! Как дела?",
        prompt_text="",
        prompt_wav="",
        zero_shot_spk_id="mita_1"
    )


    speeches = [chunk['tts_speech'] for chunk in output]
    full_speech = torch.cat(speeches, dim=1)

    # -> numpy
    audio = full_speech.squeeze(0).cpu().numpy()

    # -> байты wav
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=model.sample_rate, format="WAV")
    wav_bytes = buf.getvalue()


    NAME_AUDIO_OUT = Path("1.wav")
    NAME_AUDIO_OUT.write_bytes(wav_bytes)

inference()