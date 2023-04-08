import numpy as np

import torch
from torchvision.ops import masks_to_boxes
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import pycocotools.mask as mask_utils

from registry import Registry


def compact(l):
    return list(filter(None, l))

def register_dataset(name: str):
    def decorator(dataset_getter):
        registry = Registry()
        registry.add_to_registry(f'datasets/{name}', dataset_getter)
        return dataset_getter

    return decorator


class BaseTransforms(object):
    """Data pre-processing steps."""

    def __init__(self, resolution, mean=(0.5,), std=(0.5,)):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Normalize(mean, std),  # [-1, 1]
            transforms.Resize(resolution),
        ])
        self.resolution = resolution

    def process_mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)[0]
        else:
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)
        return mask

    def __call__(self, input):
        return self.transforms(input)


def anno2mask(anno):
    """`anno` corresponds to `anno['frames'][i]` of a CLEVRER annotation."""
    masks = []
    for obj in anno['objects']:
        mask = mask_utils.decode(obj['mask'])
        masks.append(mask)
    masks = np.stack(masks, axis=0).astype(np.int32)  # [N, H, W]
    # put background mask at the first
    bg_mask = np.logical_not(np.any(masks, axis=0))[None]
    masks = np.concatenate([bg_mask, masks], axis=0)  # [N+1, H, W]
    return masks


def masks_to_boxes_pad(masks, num):
    """Extract bbox from mask, then pad to `num`.

    Args:
        masks: [N, H, W], masks of foreground objects
        num: int

    Returns:
        bboxes: [num, 4]
        pres_mask: [num], True means object, False means padded
    """
    masks = masks.clone()
    masks = masks[masks.sum([-1, -2]) > 0]
    bboxes = masks_to_boxes(masks).float()  # [N, 4]
    pad_bboxes = torch.zeros((num, 4)).type_as(bboxes)  # [num, 4]
    pad_bboxes[:bboxes.shape[0]] = bboxes
    pres_mask = torch.zeros(num).bool()  # [num]
    pres_mask[:bboxes.shape[0]] = True
    return pad_bboxes, pres_mask


class ContrastTransform:
    """Rotate by one of the given angles."""

    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, x):
        return F.adjust_contrast(x, self.contrast)


class ContrastTransforms(BaseTransforms):

    def __init__(self, resolution, mean=(0.5,), std=(0.5,), contrast=3):
        super().__init__(resolution, mean, std)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            ContrastTransform(contrast=contrast),
            transforms.Normalize(mean, std),  # [-1, 1]
            transforms.Resize(resolution),
        ])