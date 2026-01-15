import datetime as dt
import warnings
from pathlib import Path

import ffmpeg
import IPython.display as ipd
import joblib as jl
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from match_ttsg.hifigan.config import v1
from match_ttsg.hifigan.denoiser import Denoiser
from match_ttsg.hifigan.env import AttrDict
from match_ttsg.hifigan.models import Generator as HiFiGAN
# MatchTTSG imports
from match_ttsg.models.match_ttsg_custom_deca_noprior import MatchTTSGCustomDECA
from match_ttsg.text import sequence_to_text, text_to_sequence
from match_ttsg.utils.model import denormalize
from match_ttsg.utils.utils import get_user_data_dir, intersperse

#from match_ttsg.models.match_ttsg_custom_audioonly import MatchTTSGCustomDECA

print(torch.__version__)

import speechbrain
from speechbrain.inference.vocoders import HIFIGAN
import pandas as pd
import pickle

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='abi_single_noprior.log', encoding='utf-8', level=logging.DEBUG)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(speechbrain.__version__)
#base+finetune 600 step "logs/train/multimodal_finetune_verbatim_baseline/runs/2025-10-03_16-59-11/checkpoints/last.ckpt"
#finetune 100 step "logs/train/multimodal_single100k_verbatim_baseline/runs/2025-10-05_02-59-05/checkpoints/last.ckpt"
#base 500 step "logs/train/multimodal_base_verbatim_baseline/runs/2025-09-11_14-47-30/checkpoints/last.ckpt"
#kon single MATCHTTSGDECA_CHECKPOINT = "logs/train/multimodal_single_verbatim_baseline/runs/2025-05-25_02-52-53/checkpoints/last.ckpt"
MATCHTTSGDECA_CHECKPOINT = "logs/train/multimodal_single100k_verbatim_baseline/runs/abi_scratch_decanew_noprior/checkpoints/last.ckpt"
HIFIGAN_CHECKPOINT = get_user_data_dir() / "generator_v1"#"hifigan_T2_v1"


def load_model(checkpoint_path):
    model = MatchTTSGCustomDECA.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model
count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

model = load_model(MATCHTTSGDECA_CHECKPOINT)

def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

vocoder = load_vocoder(HIFIGAN_CHECKPOINT)

n_timestepss = [50, 500]
for n_timesteps in n_timestepss:
    ## Number of ODE Solver steps
    #n_timesteps = 50

    OUTPUT_FOLDER = f"output/abi_single100k_noprior_audio_{n_timesteps}"

    ## Changes to the speaking rate
    length_scale=1.0

    ## Sampling temperature
    temperature = 0.667

    @torch.inference_mode()
    def process_text(text: str):
        #x = torch.tensor(intersperse(text_to_sequence(text, ['japanese_accent_cleaners']), 0),dtype=torch.long, device=device)[None]
        x = torch.tensor(text_to_sequence(text, ['japanese_accent_cleaners']),dtype=torch.long, device=device)[None]
        x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())
        return {
            'x_orig': text,
            'x': x,
            'x_lengths': x_lengths,
            'x_phones': x_phones
        }


    @torch.inference_mode()
    def synthesise(text, spks=None):
        text_processed = process_text(text)
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed['x'], 
            text_processed['x_lengths'],
            n_timesteps=n_timesteps,
            temperature=temperature,
            spks=spks,
            length_scale=length_scale
        )
        # merge everything to one dict    
        output.update({'start_t': start_t, **text_processed})
        return output

    @torch.inference_mode()
    def to_waveform(mel, vocoder):
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="pretrained_models/tts-hifigan-libritts-16kHz")
        audio = hifi_gan.decode_batch(mel)
        return audio.cpu().squeeze()

    def save_to_folder(filename: str, output: dict, folder: str):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        np.save(folder / f'{filename}', output['mel'].cpu().numpy())
        sf.write(folder / f'{filename}.wav', output['waveform'], 16000, 'PCM_24')
        #with open(folder / f'{filename}.bvh', 'w') as f:
            #bvh_writer.write(output['bvh'], f)

    def save_to_movement_csv(filename: str, output: dict, folder:str):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        expression_df = pd.DataFrame(output['expression'].cpu().squeeze().T.numpy())
        pose_df = pd.DataFrame(output['pose'].cpu().squeeze().T.numpy())
        with open(f"./{folder}/{filename}.decaexp.pkl", 'wb') as f:
            pickle.dump(expression_df, f)
        with open(f"./{folder}/{filename}.decapose.pkl", 'wb') as f:
            pickle.dump(pose_df, f)

    test_data_list = []
    test_data_path = "../data/multimodal_single_abi_verbatim_DECA_newfilter/test.txt"
    with open(test_data_path, 'r') as f:
        for line in f:
            dict = {
                "filename": line.strip().split("|")[0].split("/")[-1][:-4],
                "text": line.strip().split("|")[1]
            }
            
            test_data_list.append(dict)


    outputs, rtfs = [], []
    rtfs_w = []
    for data in (tqdm(test_data_list)):
        output = synthesise(data["text"]) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        #print(output['decoder_outputs_mel'].shape)
        output['waveform'] = to_waveform(output['mel'], vocoder)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 16000 / (output['waveform'].shape[-1])
        dur = output['waveform'].shape[-1] / 16000

        ## Pretty print
        """
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output['x_orig'])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output['x_phones'])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']*16000/22050:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        print(f"Audio duration:\t{dur:.6f}")
        """
        rtfs.append(output['rtf'])
        rtfs_w.append(rtf_w)
        outputs.append(output)

        ## Display the synthesised waveform
        ipd.display(ipd.Audio(output['waveform'], rate=16000))

        ## Save the generated waveform
        save_to_folder(data["filename"], output, OUTPUT_FOLDER)

        ## face
        save_to_movement_csv(data["filename"], output, f"output/abi_single100k_noprior_movement_{n_timesteps}")

    #print(f"Number of ODE steps: {n_timesteps}")
    #print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    #print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
    logger.info(f"Number of ODE steps: {n_timesteps}")
    logger.info(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    logger.info(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
