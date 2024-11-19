r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

#from match_ttsg.data.text_mel_datamodule import TextMelDataModule
from match_ttsg.data.text_mel_blendshape_rotation_datamodule import TextMelBlendshapeRotationDataModule
from match_ttsg.utils.logging_utils import pylogger

log = pylogger.get_pylogger(__name__)


def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int, out_channels_blendshapes: int, out_channels_rotations: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    total_blendshapes_sum = 0
    total_blendshapes_sq_sum = 0
    total_rotations_sum = 0
    total_rotations_sq_sum = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        blendshapes = batch["y_blendshape"]
        rotations = batch["y_rotation"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

        total_blendshapes_sum += torch.sum(blendshapes)
        total_blendshapes_sq_sum += torch.sum(torch.pow(blendshapes, 2))
        total_rotations_sum += torch.sum(rotations)
        total_rotations_sq_sum += torch.sum(torch.pow(rotations, 2))

    mel_mean = total_mel_sum / (total_mel_len * out_channels)
    mel_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(mel_mean, 2))

    blendshape_mean = total_blendshapes_sum / (total_mel_len * out_channels_blendshapes)
    blendshape_std = torch.sqrt((total_blendshapes_sq_sum / (total_mel_len * out_channels_blendshapes)) - torch.pow(blendshape_mean, 2))
    rotation_mean = total_rotations_sum / (total_mel_len * out_channels_rotations)
    rotation_std = torch.sqrt((total_rotations_sq_sum / (total_mel_len * out_channels_rotations)) - torch.pow(rotation_mean, 2))

    return {"mel_mean": mel_mean.item(), "mel_std": mel_std.item(), "blendshape_mean": blendshape_mean.item(), "blendshape_std": blendshape_std.item(), "rotation_mean": rotation_mean.item(), "rotation_std": rotation_std.item()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        default="vctk.yaml",
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="256",
        help="Can have increased batch size for faster computation",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    args = parser.parse_args()
    output_file = Path(args.input_config).with_suffix(".json")

    if os.path.exists(output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["data_statistics"] = None
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))

    text_mel_datamodule = TextMelBlendshapeRotationDataModule(**cfg)
    text_mel_datamodule.setup()
    data_loader = text_mel_datamodule.train_dataloader()
    log.info("Dataloader loaded! Now computing stats...")
    params = compute_data_statistics(data_loader, cfg["n_feats"], cfg["n_blendshapes"], cfg["n_rotations"])
    print(params)
    json.dump(
        params,
        open(output_file, "w"),
    )


if __name__ == "__main__":
    main()
