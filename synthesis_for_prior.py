from match_ttsg.synthesis import Synthetizer
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import csv

test_data_list = []
test_data_path = "../data/multimodal_single_abi_verbatim_DECA_newfilter/test.txt"
with open(test_data_path, 'r') as f:
    for line in f:
        dict = {
            "filename": line.strip().split("|")[0].split("/")[-1][:-4],
            "text": line.strip().split("|")[1]
        }
        
        test_data_list.append(dict)


synthetizer_prior = Synthetizer(checkpoint_path = "logs/train/multimodal_single100k_verbatim_baseline/runs/abi_scratch_decanew2/checkpoints/last.ckpt", n_timesteps = 50)
synthetizer_no_prior = Synthetizer(checkpoint_path = "logs/train/multimodal_single100k_verbatim_baseline/runs/abi_scratch_decanew_noprior/checkpoints/last.ckpt", n_timesteps = 50)

prior_list = []
no_prior_list = []
for data in (tqdm(test_data_list)):
    synthesis_result_prior = synthetizer_prior.synthesize(data["text"])
    synthesis_result_no_prior = synthetizer_no_prior.synthesize(data["text"])
    #print(synthesis_result.keys())
    #print(synthesis_result["encoder_outputs"].shape)
    #print(synthesis_result["decoder_outputs"].shape)
    diff_prior = synthesis_result_prior["encoder_outputs"] - synthesis_result_prior["decoder_outputs"]
    diff_no_prior = synthesis_result_no_prior["encoder_outputs"] - synthesis_result_no_prior["decoder_outputs"]

    diff_prior_list = np.array(diff_prior.cpu()).flatten().tolist()
    diff_no_prior_list = np.array(diff_no_prior.cpu()).flatten().tolist()

    prior_list = prior_list + diff_prior_list
    no_prior_list = no_prior_list + diff_no_prior_list

with open("abi_prior_diff.csv", mode='w', newline='') as file:
    writer = csv.writer(file)

    # Wrap each item in its own list to write it as a single-column row
    for item in prior_list:
        writer.writerow([item])

with open("abi_noprior_diff.csv", mode='w', newline='') as file:
    writer = csv.writer(file)

    # Wrap each item in its own list to write it as a single-column row
    for item in no_prior_list:
        writer.writerow([item])
