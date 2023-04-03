import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from tap import Tap

from configs.core.training_config import TrainingConfig
from methods import get_method

parser = argparse.ArgumentParser()


class TrainArguments(Tap):
    method: str
    config: str
    ddp: bool = False
    fp16: bool = False
    checkpoint: Optional[str] = None


if __name__ == "__main__":
    args = TrainArguments().parse_args()
    method_class = get_method(args.method)
    method = method_class(args.config)
    config: TrainingConfig = method.config
    strategy = "ddp" if args.ddp else None
    precision = 16 if args.fp16 else None

    wandb_logger = WandbLogger(project=config.project, name=config.run_name)
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoint/{config.run_name}/epochs")
    lr_callback = LearningRateMonitor(logging_interval='step')
    tqdm_callback = TQDMProgressBar(refresh_rate=5)

    trainer = pl.Trainer(
        default_root_dir=f"checkpoint/{config.run_name}",
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=strategy,
        precision=precision,
        max_epochs=config.max_epochs,
        gradient_clip_val=config.clip_grad,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            tqdm_callback
        ]
    )

    trainer.fit(method)