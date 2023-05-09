import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from cli.train import TrainArguments
from configs.core.training_config import TrainingConfig
from methods import get_method

if __name__ == "__main__":
    args = TrainArguments().parse_args()
    method_class = get_method(args.method)
    method = method_class(args.config)
    config: TrainingConfig = method.config
    strategy = "ddp" if args.ddp else 'auto'
    precision = 16 if args.fp16 else 32
    pl.seed_everything(config.seed)



    wandb_logger = WandbLogger(save_dir=f'checkpoint/{config.run_name}', project=config.project, name=config.run_name, log_model='all')
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoint/{config.run_name}", save_on_train_epoch_end=True, save_top_k=3, monitor='val/total_loss', )
    lr_callback = LearningRateMonitor(logging_interval='step')
    tqdm_callback = TQDMProgressBar(refresh_rate=5)

    trainer = pl.Trainer(
        default_root_dir=f"checkpoint/{config.run_name}",
        logger=wandb_logger,
        accelerator=config.accelerator,
        log_every_n_steps=5,
        devices=config.devices,
        accumulate_grad_batches=config.grad_accum_steps,
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

    trainer.fit(method, ckpt_path=args.checkpoint)
