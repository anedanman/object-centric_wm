import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

import argparse

from models.slate_nocross_attn import SlateNoCAWM
from envs.causal_world import CausalWorldPush


def main(args):
    dummy_train_loader = DataLoader([0 for i in range(args.steps_log)], batch_size=1)
    dummy_val_loader = DataLoader([0], batch_size=1)
    env = CausalWorldPush(image_size=args.image_size)
    model = SlateNoCAWM(args, env)
    logger = WandbLogger(project="SLATE-based-WM", name=args.name)
    trainer = pl.Trainer(
        # accelerator="cpu",
        num_sanity_val_steps=0,
        logger=logger,
        gpus=-1,
        max_epochs=args.total_steps // (args.steps_log*args.collect_steps), # number of env.steps per epoch is args.steps_log*args.collect_steps
        deterministic=False
    )
    trainer.fit(model, dummy_train_loader, dummy_val_loader)
    wandb.finish()
    model.save(f'./{args.name}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--update_steps', type=int, default=5)
    parser.add_argument('--steps_log', type=int, default=100)
    parser.add_argument('--collect_steps', type=int, default=100)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--action_size', type=int, default=9)
    parser.add_argument('--slot_size', type=int, default=64)
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--dense_activation_function', type=str, default='relu')
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--train_seq_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dvae_lr', type=float, default=3e-4)
    parser.add_argument('--rssm_lr', type=float, default=1e-4)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--reward_lr', type=float, default=1e-4)
    parser.add_argument('--seed_steps', type=int, default=10000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_slots', type=int, default=6)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--image_size', type=int, default=96)
    parser.add_argument('--gen_len', type=int, default=16)
    parser.add_argument('--imagine_horizon', type=int, default=16)
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_iterations', type=int, default=4)
    parser.add_argument('--mlp_hidden_size', type=int, default=128)
    parser.add_argument('--num_slot_heads', type=int, default=1)
    parser.add_argument('--use_detach', type=bool, default=True)
    args = parser.parse_args()
    main(args)
