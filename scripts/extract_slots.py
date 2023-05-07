import os
import sys
import pdb
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data._utils.collate import default_collate

from modules.slots.steve import STEVE
from datasets import get_dataset
from utils.checkpoint import delete_model_from_state_dict
from utils.io import mkdir_or_exist, dump_obj
from configs.steve.utils import get_steve_config



@torch.no_grad()
def extract_video_slots(model, dataset, config):
    """Returns slots extracted from each video of the dataset."""
    model.eval()
    slot_key = 'post_slots' if config.model == 'StoSAVi' else 'slots'
    torch.cuda.empty_cache()
    # videos are long, so we use 1 video per GPU as 1 batch
    bs = torch.cuda.device_count()
    all_slots = []
    range_idx = range(0, dataset.num_videos, bs)
    for start_idx in tqdm(range_idx):
        end_idx = min(start_idx + bs, dataset.num_videos)
        data_dict = default_collate(
            [dataset.get_video(i) for i in range(start_idx, end_idx)])
        in_dict = {'img': data_dict['video'].float().cuda()}
        out_dict = model(in_dict)
        slots = out_dict[slot_key].detach().cpu().numpy()  # [B, T, n, c]
        all_slots += [slot for slot in slots]
        torch.cuda.empty_cache()
    all_slots = np.stack(all_slots, axis=0)  # [N, T, n, c]
    return all_slots

def resolve_model_and_config(model_name, config_name):
    if model_name == 'steve':
        config = get_steve_config(config_name)
        model = STEVE(**config.get_model_config())
        return model, config
    else:
        raise NotImplementedError(f'Configs for model {model} are not implemented')

def process_video(model, config, args):
    """Extract slot_embs using video SlotAttn model"""
    train_set, val_set = get_dataset(config.dataset)(config)

    # forward through train/val dataset
    print(f'Processing {config.dataset} video val set...')
    val_slots = extract_video_slots(model, val_set, args)
    print(f'Processing {config.dataset} video train set...')
    train_slots = extract_video_slots(model, train_set, args)

    # # also extract test_set for CLEVRER
    # if config.dataset == 'clevrer':
    #     test_set = build_clevrer_dataset(config, test_set=True)
    #     print(f'Processing {params.dataset} video test set...')
    #     test_slots = extract_video_slots(model, test_set)

    try:
        train_slots = {
            os.path.basename(train_set.files[i]): train_slots[i]
            for i in range(len(train_slots))
        }  # each embedding is of shape [T, N, C]
        val_slots = {
            os.path.basename(val_set.files[i]): val_slots[i]
            for i in range(len(val_slots))
        }
        slots = {'train': train_slots, 'val': val_slots}

        # if params.dataset == 'clevrer':
        #     test_slots = {
        #         os.path.basename(test_set.files[i]): test_slots[i]
        #         for i in range(len(test_slots))
        #     }
        #     slots['test'] = test_slots

        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {config.dataset} video dataset, '
              f'train: {len(train_slots)}/{train_set.num_videos}, '
              f'val: {len(val_slots)}/{val_set.num_videos}')

        # if params.dataset == 'clevrer':
        #     print(f'test: {len(test_slots)}/{test_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to the weight dir
    # if 'physion' in args.params:
    #     ln_path = os.path.join(
    #         os.path.dirname(args.weight), f'{args.subset}_slots.pkl')
    # else:
    ln_path = os.path.join(
            os.path.dirname(args.weight), 'slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def process_test_video(model, config):
    """Extract slot_embs using video SlotAttn model"""
    test_set = get_dataset(config.dataset)(config)

    # forward through test dataset
    print(f'Processing {config.dataset} video test set...')
    test_slots = extract_video_slots(model, test_set, args)

    try:
        test_slots = {
            os.path.basename(test_set.files[i]): test_slots[i]
            for i in range(len(test_slots))
        }  # each embedding is of shape [T, N, C]
        slots = {'test': test_slots}
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {config.dataset} video dataset, '
              f'test: {len(test_slots)}/{test_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to the weight dir
    ln_path = os.path.join(
            os.path.dirname(args.weight), 'test_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def main(args):
    model, config = resolve_model_and_config(args.model, args.config)
    state = delete_model_from_state_dict(torch.load(args.weight, map_location='cpu')['state_dict'])
    model.load_state_dict(state)
    model.testing = True  # we just want slots
    model = torch.nn.DataParallel(model).cuda().eval()
    if 'test' in config.dataset:
        process_test_video(model, config, args)
    else:
        process_video(model, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract slots from videos')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--subset', type=str, default='training')  # Physion
    parser.add_argument(
        '--weight', type=str, required=True, help='pretrained model weight')
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,  # './data/CLEVRER/slots.pkl'
        help='path to save slots',
    )
    args = parser.parse_args()
    
    

    torch.backends.cudnn.benchmark = True
    main(args)
