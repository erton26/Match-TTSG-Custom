from match_ttsg.utils.audio import mel_spectrogram
import torchaudio as ta
from match_ttsg.utils.model import fix_len_compatibility, normalize, denormalize
from speechbrain.inference.vocoders import HIFIGAN

def get_mel(filepath):
    audio, sr = ta.load(filepath)
    assert sr == 16000
    print(audio.shape)
    mel = mel_spectrogram(audio, 1024, 80, 16000, 256,
                            1024, 0, 8000, center=False).squeeze()
    mel = normalize(mel, -5.217583656311035, 1.5713294744491577)
    mel = denormalize(mel, -5.217583656311035, 1.5713294744491577)
    return mel

mel = get_mel("./data122_host_0_3.060_5.600.wav")
print(mel.shape)

vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="pretrained_models/tts-hifigan-libritts-16kHz")
out_audio = vocoder.decode_batch(mel,256)
print(out_audio.shape)