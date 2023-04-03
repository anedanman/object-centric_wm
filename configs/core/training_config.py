from dataclasses import dataclass
from typing import Any, Dict

from attr import define


@define(kw_only=True)
class TrainingConfig:
    project: str = "Object centric WM"
    run_name: str = "Run"

    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = 0

    # training params
    accelerator: str = 'auto'
    devices = None
    max_epochs = 100
    san_check_val_step = 2  # to verify code correctness
    print_iter = 50
    save_interval = 1.0  # save every (num_epoch_iters * save_interval) iters
    eval_interval = 1  # should be int, number of epochs between each eval
    save_epoch_end = False  # save ckp at the end of every epoch

    # data settings
    data_root = ''
    train_batch_size = 64
    val_batch_size = train_batch_size * 2
    num_workers = 8
    grad_accum_steps = 1


    losses_weights: Dict[str, int] = {}



