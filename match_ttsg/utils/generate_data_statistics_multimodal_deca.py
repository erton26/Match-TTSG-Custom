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
from match_ttsg.data.text_mel_deca_datamodule import TextMelDECADataModule
from match_ttsg.utils.logging_utils import pylogger

log = pylogger.get_pylogger(__name__)


def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int, out_channels_expressions: int, out_channels_poses: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    total_expressions_sum = 0
    total_expressions_sq_sum = 0
    total_poses_sum = 0
    total_poses_sq_sum = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        expressions = batch["y_expression"]
        poses = batch["y_pose"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

        total_expressions_sum += torch.sum(expressions)
        total_expressions_sq_sum += torch.sum(torch.pow(expressions, 2))
        total_poses_sum += torch.sum(poses)
        total_poses_sq_sum += torch.sum(torch.pow(poses, 2))

    mel_mean = total_mel_sum / (total_mel_len * out_channels)
    mel_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(mel_mean, 2))

    expression_mean = total_expressions_sum / (total_mel_len * out_channels_expressions)
    expression_std = torch.sqrt((total_expressions_sq_sum / (total_mel_len * out_channels_expressions)) - torch.pow(expression_mean, 2))
    pose_mean = total_poses_sum / (total_mel_len * out_channels_poses)
    pose_std = torch.sqrt((total_poses_sq_sum / (total_mel_len * out_channels_poses)) - torch.pow(pose_mean, 2))

    return {"mel_mean": mel_mean.item(), "mel_std": mel_std.item(), "expression_mean": expression_mean.item(), "expression_std": expression_std.item(), "pose_mean": pose_mean.item(), "pose_std": pose_std.item()}


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

    text_mel_datamodule = TextMelDECADataModule(**cfg)
    text_mel_datamodule.setup()
    data_loader = text_mel_datamodule.train_dataloader()
    log.info("Dataloader loaded! Now computing stats...")
    params = compute_data_statistics(data_loader, cfg["n_feats"], cfg["n_expressions"], cfg["n_poses"])
    print(params)
    json.dump(
        params,
        open(output_file, "w"),
    )


if __name__ == "__main__":
    main()
