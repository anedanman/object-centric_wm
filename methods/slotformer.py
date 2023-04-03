import os
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from configs.slotformer import get_slotformer_config
from configs.slotformer.slotformer_base import SlotFormerBaseConfig
from datasets import get_dataset
from methods import register_method
from utils.slotformer_utils import get_slotformer

from utils.savi_utils import to_rgb_from_tensor
from utils.cossine_lr import CosineAnnealingWarmupRestarts
from utils.optim import get_optimizer


@register_method('slotformer')
class SlotFormerMethod(LightningModule):
    def __init__(self, config_name: str):
        super().__init__()
        self.config: SlotFormerBaseConfig = get_slotformer_config(config_name)
        self.video_logged = False
        self.model = get_slotformer(self.config)

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # pause the video on the 1st frame in PHYRE
        if 'phyre' in self.config.dataset.lower():
            imgs, recon_combined, recons, masks = [
                self._pause_frame(x)
                for x in [imgs, recon_combined, recons, masks]
            ]
        # in PHYRE if the background is black, we scale the mask differently
        scale = 0. if self.config.__dict__.get('reverse_color', False) else 1.
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1. - masks) * scale,  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                padding=3,
                pad_value=1. - scale,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    def on_train_batch_start(self):
        if not hasattr(self.config, 'use_loss_decay'):
            return

            # decay the temporal weighting linearly
        if not self.config.use_loss_decay:
            self.model.module.loss_decay_factor = 1.
            return

        cur_steps = self.global_step
        total_steps = self.config.max_epochs * len(self.train_loader)
        decay_steps = self.config.loss_decay_pct * total_steps

        if cur_steps >= decay_steps:
            self.model.loss_decay_factor = 1.
            return

        # increase tau linearly from 0.01 to 1
        self.model.loss_decay_factor = \
            0.01 + cur_steps / decay_steps * 0.99

    def _compare_videos(self, img, recon_combined, rollout_combined):
        """Stack 3 videos to compare them."""
        # pause the 1st frame if on PHYRE
        if 'phyre' in self.config.dataset.lower():
            img, recon_combined, rollout_combined = [
                self._pause_frame(x)
                for x in [img, recon_combined, rollout_combined]
            ]
        # pad to the length of rollout video
        T = rollout_combined.shape[0]
        img = self._pad_frame(img, T)
        recon_combined = self._pad_frame(recon_combined, T)
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    img,  # original images
                    recon_combined,  # reconstructions
                    rollout_combined,  # rollouts
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                # pad white if using black background
                padding=3,
                pad_value=1 if self.config.__dict__.get('reverse_color', False) else 0,
            ) for i in range(img.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    def _read_video_and_slots(self, dst, idx):
        """Read the video and slots from the dataset."""
        # PHYRE
        if 'phyre' in self.config.dataset.lower():
            # read video
            data_dict = dst.get_video(idx, video_len=self.config.video_len)
            video = data_dict['video']
            # read slots
            slots = dst._read_slots(
                data_dict['data_idx'],
                video_len=self.config.video_len,
            )['slots']  # [T, N, C]
            slots = torch.from_numpy(slots).float().to(self.device)
        # OBJ3D, CLEVRER, Physion
        else:
            # read video
            video = dst.get_video(idx)['video']
            # read slots
            video_path = dst.files[idx]
            slots = dst.video_slots[os.path.basename(video_path)]  # [T, N, C]
            if self.config.frame_offset > 1:
                slots = np.ascontiguousarray(slots[::self.config.frame_offset])
            slots = torch.from_numpy(slots).float().to(self.device)
        T = min(video.shape[0], slots.shape[0])
        # video: [T, 3, H, W], slots: [T, N, C]
        return video[:T], slots[:T]

    @torch.no_grad()
    def _sample_video(self):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        dst = self.val_dataloader().dataset
        sampled_idx = self._get_sample_idx(self.config.n_samples, dst)
        results, rollout_results, compare_results = [], [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            actions = None
            if self.config.rollout_dict['action_conditioning']:
                actions = dst.read_actions(i.item())

            T = video.shape[0]
            # reconstruct gt_slots as sanity-check
            # i.e. if the pre-trained weights are loaded correctly
            recon_combined, recons, masks, _ = self.model.decode(slots)
            img = video.type_as(recon_combined)
            save_video = self._make_video_grid(img, recon_combined, recons,
                                               masks)
            results.append(save_video)
            # rollout
            past_steps = self.config.input_frames
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            out_dict = self.model.rollout(
                past_slots, T - past_steps, decode=True, with_gt=True, actions=actions)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            rollout_combined, recons, masks = out_dict['recon_combined'], \
                out_dict['recons'], out_dict['masks']
            img = video.type_as(rollout_combined)
            pred_video = self._make_video_grid(img, rollout_combined, recons,
                                               masks)
            rollout_results.append(pred_video)  # per-slot rollout results
            # stack (gt video, gt slots recon video, slot_0 rollout video)
            # horizontally to better compare the 3 videos
            compare_video = self._compare_videos(img, recon_combined,
                                                 rollout_combined)
            compare_results.append(compare_video)

        log_dict = {
            'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
            'val/compare_video': self._convert_video(compare_results),
        }
        self.log_dict(log_dict)
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
        loss_out = self.model.calc_train_loss(data_batch)
        loss_out['total_loss'] = self.resolve_loss(loss_out)
        self._log_step(loss_out, prefix='train')
        return loss_out['total_loss']

    def validation_step(self, data_batch):
        loss_out = self.model.calc_train_loss(data_batch)
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
        optimizer = get_optimizer(self.config.optimizer)
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
