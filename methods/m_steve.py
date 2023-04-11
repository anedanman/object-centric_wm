from typing import Optional, Dict
import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import wandb
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from configs.steve.steve_base import STEVEBaseConfig
from configs.steve.utils import get_steve_config
from datasets import get_dataset
from methods.utills import register_method
from modules.slots.steve import STEVE
from utils.cossine_lr import CosineAnnealingWarmupRestarts
from utils.optim import filter_wd_parameters, get_optimizer
from utils.savi_utils import to_rgb_from_tensor
from utils.steve_utils import gumbel_softmax, make_one_hot


@register_method("steve")
class STEVEMethod(LightningModule):
    def __init__(self, config_name: str):
        super().__init__()
        self.config: STEVEBaseConfig = get_steve_config(config_name)
        self.video_logged = False
        self.model = STEVE(**self.config.get_model_config())

    @property
    def vis_fps(self):
        # PHYRE
        if 'phyre' in self.config.dataset.lower():
            return 4
        # OBJ3D, CLEVRER, Physion
        else:
            return 8

    def _log_step(self, data_dict, prefix=''):
        if prefix:
            data_dict = {f'{prefix}/{key}': val for key, val in data_dict.items()}
        self.log_dict(data_dict)

    def training_step(self, data_batch, k):
        model_out = self.model(data_batch)
        loss_out = self.model.calc_train_loss(data_batch, model_out)
        loss_out['total_loss'] = self.resolve_loss(loss_out)
        self._log_step(loss_out, prefix='train')
        return loss_out['total_loss']

    def validation_step(self, data_batch, k):
        model_out = self.model(data_batch)
        loss_out = self.model.calc_train_loss(data_batch, model_out)
        loss_out['total_loss'] = self.resolve_loss(loss_out)
        self._log_step(loss_out, prefix='val')
        return loss_out['total_loss']

    @staticmethod
    def _make_video(video, soft_video, hard_video, history_len=None):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    video.cpu(),  # original video
                    soft_video.cpu(),  # dVAE gumbel softmax reconstruction
                    hard_video.cpu(),  # argmax token reconstruction
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    @staticmethod
    def _make_slots_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video,  # [T, num_slots, 3, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video

    @staticmethod
    def _make_masks_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video.unsqueeze(2).expand(-1, -1, 3, -1, -1),  # [T, num_slots, 1, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video

    def on_train_epoch_end(self, *args, **kwarg):
        checkpoints_path = os.path.join(os.path.dirname(__file__), '..',
                                        f'checkpoint/{self.config.run_name}/epochs/model_{self.current_epoch}.pt')
        torch.save(self.model.state_dict(), checkpoints_path)

    @torch.no_grad()
    def _sample_video(self):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        self.model.eval()
        self.model.testing = True  # we only want the slots
        dst = self.val_dataset
        sampled_idx = self._get_sample_idx(self.config.n_samples, dst)
        num_patches = self.model.num_patches
        n = int(num_patches ** 0.5)
        results, recon_results, masks_result = [], [], []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            data_dict = {'img': video[None]}
            out_dict = self.model(data_dict)
            masks = out_dict['masks'][0]  # [T, num_slots, H, W]
            masked_video = video.unsqueeze(1) * masks.unsqueeze(2)
            # [T, num_slots, C, H, W]
            save_video = self._make_slots_video(video, masked_video)
            masks_video = self._make_masks_video(video, masks)
            results.append(save_video)
            masks_result.append(masks_video)
            if not self.config.recon_video:
                continue

            # reconstruct the video by autoregressively generating patch tokens
            # using Transformer decoder conditioned on slots
            slots = out_dict['slots'][0]  # [T, num_slots, slot_size]
            all_soft_video, all_hard_video, bs = [], [], 16  # to avoid OOM
            for batch_idx in range(0, slots.shape[0], bs):
                _, logits = self.model.trans_decoder.generate(
                    slots[batch_idx:batch_idx + bs],
                    steps=num_patches,
                    sample=False,
                )
                # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
                logits = logits.transpose(2, 1).unflatten(
                    -1, (n, n)).contiguous().cuda()
                # 1. use logits after gumbel softmax to reconstruct the video
                z_logits = F.log_softmax(logits, dim=1)
                z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
                recon_video = self.model.dvae.detokenize(z)
                all_soft_video.append(recon_video.cpu())
                del z_logits, z, recon_video
                torch.cuda.empty_cache()
                # 2. SLATE directly use ont-hot token (argmax) as input
                z_hard = make_one_hot(logits, dim=1)
                recon_video_hard = self.model.dvae.detokenize(z_hard)
                all_hard_video.append(recon_video_hard.cpu())
                del logits, z_hard, recon_video_hard
                torch.cuda.empty_cache()

            recon_video = torch.cat(all_soft_video, dim=0)
            recon_video_hard = torch.cat(all_hard_video, dim=0)
            save_video = self._make_video(video, recon_video, recon_video_hard)
            recon_results.append(save_video)
            torch.cuda.empty_cache()

        log_dict = {'val/video': self._convert_video(results),
                    'val/masks_video': self._convert_video(masks_result)}
        if self.config.recon_video:
            log_dict['val/recon_video'] = self._convert_video(recon_results)
        self.logger.experiment.log(log_dict, step=self.global_step)
        torch.cuda.empty_cache()
        self.model.testing = False

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

    def on_validation_epoch_end(self, *args, **kwargs):
        self._sample_video()

    def resolve_loss(self, loss_dict: Dict[str, torch.Tensor]):
        loss = 0
        losses_weights = self.config.losses_weights
        for loss_name, loss_val in loss_dict.items():
            loss = loss + losses_weights.get(loss_name, 1) * loss_val
        return loss

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_getter = get_dataset(self.config.dataset)
        self.train_dataset, self.val_dataset = dataset_getter(self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.train_batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.val_batch_size,
                          pin_memory=True,
                          num_workers=self.config.num_workers)

    def configure_optimizers(self):
        # STEVE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': self.config.dec_lr,
            },
        ]

        optimizer = get_optimizer(self.config.optimizer)(params,
                                                         weight_decay=self.config.weight_decay,
                                                         lr=self.config.lr)

        total_steps = len(self.train_dataloader()) * self.config.max_epochs
        warmup_steps = self.config.warmup_steps_pct * total_steps
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(self.config.lr, self.config.dec_lr),
            min_lr=(0., 0.),
            warmup_steps=warmup_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]