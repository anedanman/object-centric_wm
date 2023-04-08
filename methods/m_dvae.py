import os
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from configs.dvae.dvae_base import DVAEBaseConfig
from configs.dvae.utils import get_dvae_config
from datasets import get_dataset
from methods.utills import register_method
from modules.slots.dVAE import dVAE
from utils.slotformer_utils import get_slotformer

from utils.savi_utils import to_rgb_from_tensor
from utils.cossine_lr import CosineAnnealingWarmupRestarts
from utils.optim import get_optimizer, filter_wd_parameters
from utils.steve_utils import cosine_anneal


@register_method('dvae')
class DVAEMethod(LightningModule):
    def __init__(self, config_name: str):
        super().__init__()
        self.config: DVAEBaseConfig = get_dvae_config(config_name)
        self.video_logged = False
        self.model = dVAE(self.config.vocab_size)

    @staticmethod
    def _make_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(torch.stack([video, pred_video],
                                             dim=1))  # [T, 2, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 2*W]
        return save_video

    def on_train_batch_start(self):

        cur_steps = self.global_step
        total_steps = self.config.max_epochs * len(self.train_loader())
        decay_steps = self.config.tau_decay_pct * total_steps

        self.model.tau = cosine_anneal(
            cur_steps,
            start_value=self.config.init_tau,
            final_value=self.config.final_tau,
            start_step=0,
            final_step=decay_steps,
        )

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results = []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            all_recons, bs = [], 100  # a hack to avoid OOM
            for batch_idx in range(0, video.shape[0], bs):
                data_dict = {
                    'img': video[batch_idx:batch_idx + bs],
                    'tau': 1.,
                    'hard': True,
                }
                recon = model(data_dict)['recon']
                all_recons.append(recon)
                torch.cuda.empty_cache()
            recon_video = torch.cat(all_recons, dim=0)
            save_video = self._make_video(video, recon_video)
            results.append(save_video)

        wandb.log({'val/video': self._convert_video(results)}, step=self.it)
        torch.cuda.empty_cache()

    @property
    def vis_fps(self):
        # PHYRE
        if 'phyre' in self.config.dataset.lower():
            return 4
        # OBJ3D, CLEVRER, Physion
        else:
            return 8

    def log_step(self, data_dict, prefix=''):
        if prefix:
            data_dict = {f'{prefix}/{key}': val for key, val in data_dict.items()}
        self.log_dict(data_dict)

    def training_step(self, data_batch):
        model_out = self.model(data_batch)
        loss_out = self.model.calc_train_loss(data_batch, model_out)
        loss_out['total_loss'] = self.resolve_loss(loss_out)
        loss_out['tau_gumbell'] = self.model.tau
        self._log_step(loss_out, prefix='train')
        return loss_out['total_loss']

    def validation_step(self, data_batch, k):
        model_out = self.model(data_batch)
        loss_out = self.model.calc_train_loss(data_batch, model_out)
        loss = self.resolve_loss(loss_out)
        self._log_step(loss_out, prefix='val')
        if not self.video_logged:
            self.video_logged = True
            self._sample_video()
        return loss

    @staticmethod
    def _pad_frame(video, target_T):
        """Pad the video to a target length at the end"""
        if video.shape[0] >= target_T:
            return video
        dup_video = torch.stack(
            [video[-1]] * (target_T - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)

    @staticmethod
    def _pause_frame(video, N=4):
        """Pause the video on the first frame by duplicating it"""
        dup_video = torch.stack([video[0]] * N, dim=0)
        return torch.cat([dup_video, video], dim=0)

    def _convert_video(self, video, caption=None):
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).numpy().astype(np.uint8)
        return wandb.Video(video, fps=self.vis_fps, caption=caption)

    @staticmethod
    def _get_sample_idx(N, dst):
        """Load videos uniformly from the dataset."""
        dst_len = len(dst.files)  # treat each video as a sample
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx

    def on_validation_end(self):
        self.video_logged = False

    def resolve_loss(self, loss_dict: Dict[str, torch.Tensor]):
        loss = 0
        losses_weights = self.config.losses_weights
        for loss, loss_val in loss_dict.items():
            if loss in losses_weights:
                loss = loss + losses_weights[loss] * loss_val
            else:
                loss = loss + loss_val
        return loss

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_getter = get_dataset(self.config.dataset)
        self.train_dataset, self.val_dataset = dataset_getter(self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.val_batch_size)

    def configure_optimizers(self):

        params = self.model.parameters()
        if self.config.weight_decay > 0:
            params = filter_wd_parameters(self.model)

        optimizer = get_optimizer(self.config.optimizer)(params,
                                                         weight_decay=self.config.weight_decay,
                                                         lr=self.config.lr)

        total_steps = len(self.train_dataloader()) * self.config.max_epochs
        warmup_steps = self.config.warmup_steps_pct * total_steps
        lr = self.config.lr
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
