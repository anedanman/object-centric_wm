import os
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from configs.slotformer.slotformer_base import SlotFormerBaseConfig
from configs.slotformer.utils import get_slotformer_config
from datasets import get_dataset
from methods.m_steve import STEVEMethod
from methods.utills import register_method
from utils.slotformer_utils import get_slotformer

from utils.savi_utils import to_rgb_from_tensor
from utils.cossine_lr import CosineAnnealingWarmupRestarts
from utils.optim import get_optimizer, filter_wd_parameters


@register_method('slotformer')
class SlotFormerMethod(LightningModule):
    def __init__(self, config_name: str, viz_only: bool = False):
        super().__init__()
        self.config: SlotFormerBaseConfig = get_slotformer_config(config_name)
        self.model = get_slotformer(self.config)
        
    @staticmethod
    def _make_video(video, soft_video, hard_video):
        """Compare the 3 videos."""
        return STEVEMethod._make_video(video, soft_video, hard_video)

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # pause the video on the 1st frame in PHYRE
        if 'phyre' in self.config.dataset.lower():
            imgs, recon_combined, recons, masks = [
                self._pause_frame(x)
                for x in [imgs, recon_combined, recons, masks]
            ]
        # in PHYRE if the background is black, we scale the mask differently
        scale = 0. if getattr(self.config, 'reverse_color', False) else 1.
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

    # def on_train_batch_start(self, *args, **kwargs):

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
                padding=5,
                pad_value=1 if getattr(self.config, 'reverse_color', False) else 0,
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

    def _sample_video(self):
        if self.config.slots_encoder.upper() == 'SAVI':
            return self._sample_video_savi()
        else:
            return self._sample_video_steve()

    @torch.no_grad()
    def _sample_video_savi(self):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        dst = self.val_dataset
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
        self.logger.experiment.log(log_dict, step=self.global_step)
        torch.cuda.empty_cache()

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

    def training_step(self, data_batch):
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

    def on_validation_epoch_end(self, *args, **kwargs):
        self._sample_video()

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
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.val_batch_size)

    def _slots2video(self, slots):
        """Decode slots to videos."""
        T = slots.shape[0]
        all_soft_recon, all_hard_recon, bs = [], [], 16  # to avoid OOM
        for idx in range(0, T, bs):
            soft_recon, hard_recon = self.model.decode(slots[idx:idx + bs])
            all_soft_recon.append(soft_recon.cpu())
            all_hard_recon.append(hard_recon.cpu())
            del soft_recon, hard_recon
            torch.cuda.empty_cache()
        soft_recon = torch.cat(all_soft_recon, dim=0)
        hard_recon = torch.cat(all_hard_recon, dim=0)
        return soft_recon, hard_recon

    @torch.no_grad()
    def _sample_video_steve(self):
        self.model.eval()
        dst = self.val_dataset
        sampled_idx = self._get_sample_idx(self.config.n_samples, dst)
        results, rollout_results = [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            actions = None
            if self.config.rollout_dict['action_conditioning']:
                actions = dst.get_all_actions(i.item())
                actions = actions.unsqueeze(0).to(self.device)
            T = video.shape[0]
            # # recon as sanity-check
            # soft_recon, hard_recon = self._slots2video(slots)
            # save_video = self._make_video(video, soft_recon, hard_recon)
            # results.append(save_video)
            # rollout
            past_steps = self.config.input_frames
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            pred_slots = self.model.rollout(past_slots, T - past_steps, actions=actions)[0]
            slots = torch.cat([slots[:past_steps], pred_slots], dim=0)
            soft_recon, hard_recon = self._slots2video(slots)
            save_video = self._make_video(
                video, soft_recon, hard_recon)
            rollout_results.append(save_video)
            del soft_recon, hard_recon
            torch.cuda.empty_cache()

        log_dict = {
            # 'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
        }
        self.logger.experiment.log(log_dict, step=self.global_step)
        torch.cuda.empty_cache()

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
