"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from match_ttsg import utils
from match_ttsg.utils.utils import plot_tensor

import wandb

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                'mel_mean': 0.0,
                'mel_std': 1.0,
                #'motion_mean': 0.0,
                #'motion_std': 1.0,
                'blendshape_mean': 0.0,
                'blendshape_std': 1.0,
                'rotation_mean': 0.0,
                'rotation_std': 1.0,
            }

        self.register_buffer('mel_mean', torch.tensor(data_statistics['mel_mean']))
        self.register_buffer('mel_std', torch.tensor(data_statistics['mel_std']))
        #self.register_buffer('motion_mean', torch.tensor(data_statistics['motion_mean']))
        #self.register_buffer('motion_std', torch.tensor(data_statistics['motion_std']))
        self.register_buffer('blendshape_mean', torch.tensor(data_statistics['blendshape_mean']))
        self.register_buffer('blendshape_std', torch.tensor(data_statistics['blendshape_std']))
        self.register_buffer('rotation_mean', torch.tensor(data_statistics['rotation_mean']))
        self.register_buffer('rotation_std', torch.tensor(data_statistics['rotation_std']))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        #y_motion = batch["y_motion"]
        y_blendshape = batch["y_blendshape"]
        y_rotation = batch["y_rotation"]


        dur_loss, prior_loss, diff_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            #y_motion=y_motion,
            y_blendshape=y_blendshape,
            y_rotation=y_rotation,
            spks=spks,
            out_size=self.out_size,
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    """
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
                    """
                    self.logger.experiment.log({
                    f"original/{i}": wandb.Image(
                        plot_tensor(y.squeeze().cpu()),
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs_mel"], output["decoder_outputs_mel"]
                y_all_enc, y_all_dec = output["encoder_outputs"], output["decoder_outputs"], 
                #y_motion_enc, y_motion_dec, attn = output['encoder_outputs_motion'], output['decoder_outputs_motion'], output['attn']
                y_blendshape_enc, y_blendshape_dec, attn = output['encoder_outputs_blendshape'], output['decoder_outputs_blendshape'], output['attn']
                y_rotation_enc, y_rotation_dec, attn = output['encoder_outputs_rotation'], output['decoder_outputs_rotation'], output['attn']
                attn = output["attn"]
                """
                self.logger.experiment.add_image(
                    f"generated_enc/mel_{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/mel_{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                #self.logger.experiment.add_image(f'generated_enc/motion_{i}', plot_tensor(y_motion_enc.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                #self.logger.experiment.add_image(f'generated_dec/motion_{i}', plot_tensor(y_motion_dec.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(f'generated_enc/blendshape_{i}', plot_tensor(y_blendshape_enc.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(f'generated_dec/blendshape_{i}', plot_tensor(y_blendshape_dec.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(f'generated_enc/rotation_{i}', plot_tensor(y_rotation_enc.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(f'generated_dec/rotation_{i}', plot_tensor(y_rotation_dec.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                """

                self.logger.experiment.log({
                    f"generated_enc/mel_{i}": wandb.Image(
                        plot_tensor(y_enc.squeeze().cpu()), 
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })
                self.logger.experiment.log({
                    f"generated_enc/all_{i}": wandb.Image(
                        plot_tensor(y_all_enc.squeeze().cpu()), 
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })
                self.logger.experiment.log({
                    f"generated_dec/mel_{i}": wandb.Image(
                        plot_tensor(y_dec.squeeze().cpu()), 
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })
                self.logger.experiment.log({
                    f"generated_dec/all_{i}": wandb.Image(
                        plot_tensor(y_all_dec.squeeze().cpu()), 
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })
                self.logger.experiment.log({
                    f"alignment/{i}": wandb.Image(
                        plot_tensor(attn.squeeze().cpu()), 
                        caption=f"Epoch {self.current_epoch}"
                    ),
                    "epoch": self.current_epoch
                })
                #self.logger.experiment.add_image(f'generated_enc/motion_{i}', plot_tensor(y_motion_enc.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                #self.logger.experiment.add_image(f'generated_dec/motion_{i}', plot_tensor(y_motion_dec.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.log({f"generated_enc/blendshape_{i}": wandb.Image(plot_tensor(y_blendshape_enc.squeeze().cpu()), caption=f"Epoch {self.current_epoch}"), "epoch": self.current_epoch})
                self.logger.experiment.log({f"generated_dec/blendshape_{i}": wandb.Image(plot_tensor(y_blendshape_dec.squeeze().cpu()), caption=f"Epoch {self.current_epoch}"), "epoch": self.current_epoch})
                self.logger.experiment.log({f"generated_enc/rotation_{i}": wandb.Image(plot_tensor(y_rotation_enc.squeeze().cpu()), caption=f"Epoch {self.current_epoch}"), "epoch": self.current_epoch})
                self.logger.experiment.log({f"generated_dec/rotation_{i}": wandb.Image(plot_tensor(y_rotation_dec.squeeze().cpu()), caption=f"Epoch {self.current_epoch}"), "epoch": self.current_epoch})

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
