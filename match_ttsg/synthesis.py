import torch
import datetime as dt

# MatchTTSG imports
from match_ttsg.models.match_ttsg_custom_deca import MatchTTSGCustomDECA
from match_ttsg.text import sequence_to_text, text_to_sequence

from speechbrain.inference.vocoders import HIFIGAN

class Synthetizer():
    def __init__(
        self,
        checkpoint_path,
        n_timesteps,
        temperature = 0.667,
        hop_len = 256,
        sample_rate = 16000,
        hifigan_source = "speechbrain/tts-hifigan-libritts-16kHz",
        hifigan_savedir = "pretrained_models/tts-hifigan-libritts-16kHz",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={self.device}")

        self.checkpoint_path = checkpoint_path
        self.model = MatchTTSGCustomDECA.load_from_checkpoint(checkpoint_path, map_location=self.device)
        self.vocoder = HIFIGAN.from_hparams(source=hifigan_source, savedir=hifigan_savedir)

        self.n_timesteps = n_timesteps
        self.temperature = temperature
        self.hop_len = hop_len
        self.sample_rate = sample_rate

    def process_text(self, text: str):
        x = torch.tensor(text_to_sequence(text, ['japanese_accent_cleaners']),dtype=torch.long, device=self.device)[None]
        x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=self.device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())
        return {
            'x_orig': text,
            'x': x,
            'x_lengths': x_lengths,
            'x_phones': x_phones
        }
    
    def to_waveform(self, mel):
        audio = self.vocoder.decode_batch(spectrogram=mel, hop_len=self.hop_len)
        return audio.cpu().squeeze()
    
    def synthesize(self, text: str, spks=None, length_scale = 1.0):
        text_processed = self.process_text(text)
        start_t = dt.datetime.now()

        output = self.model.synthesise(
            text_processed['x'], 
            text_processed['x_lengths'],
            n_timesteps=self.n_timesteps,
            temperature=self.temperature,
            spks=spks,
            length_scale=length_scale
        )

        output['waveform'] = self.to_waveform(output['mel'])
        t = (dt.datetime.now() - start_t).total_seconds()

        rtf_w_vocoder = t * self.sample_rate / (output['waveform'].shape[-1])
        dur = output['waveform'].shape[-1] / self.sample_rate

        output['rtf_w_vocoder'] = rtf_w_vocoder
        output['waveform_duration'] = dur

        return output