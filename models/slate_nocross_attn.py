import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import torchvision.utils as vutils
import wandb

from utils.replay_buffer import ReplayBuffer
from utils.params import FreezeParameters, compute_return

from modules.oc_noca_rssm import OC_NOCA_RSSM
from modules.ac_modules import OC_ActionDecoder, OC_DenseDecoder
from modules.slate_modules import dVAE_encoder, dVAE_decoder


def preprocess_obs(obs):
    obs = obs.to(torch.float32)/255.0
    return obs


class SlateNoCAWM(pl.LightningModule):
    def __init__(self, args, env):
        super(SlateNoCAWM, self).__init__()
        self.args = args
        self.save_hyperparameters()
        
        self.rssm = OC_NOCA_RSSM(args)
        self.actor = OC_ActionDecoder(action_size = args.action_size,
            slots_size=args.slot_size,
            units = args.num_units,
            n_layers=3,
            activation=args.dense_activation_function).to(self.device)
        self.reward_model = OC_DenseDecoder(
            slots_size=args.slot_size,
            output_shape = (1,),
            n_layers = 2,
            units=self.args.num_units,
            activation= self.args.dense_activation_function,
            dist = 'normal').to(self.device)
        self.value_model  = OC_DenseDecoder(
            slots_size=args.slot_size,
            output_shape = (1,),
            n_layers = 3,
            units = self.args.num_units,
            activation= self.args.dense_activation_function,
            dist = 'normal').to(self.device) 
        
        self.env = env
        obs_shape = self.env.reset()['image'].shape
        self.obs_encoder = dVAE_encoder(args.vocab_size, args.image_size)
        self.obs_decoder = dVAE_decoder(args.vocab_size, args.image_size)

        self.dvae_optim = optim.Adam(list(self.obs_encoder.parameters()) + list(self.obs_decoder.parameters()), lr=args.dvae_lr)
        self.rssm_optim = optim.Adam(self.rssm.parameters(), lr=args.rssm_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.value_optim = optim.Adam(self.value_model.parameters(), lr=args.value_lr)
        self.reward_optim = optim.Adam(self.reward_model.parameters(), lr=args.reward_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size, obs_shape, args.action_size,
                                                    args.train_seq_len, args.batch_size)
        
        self.collect_random_episodes(args.seed_steps)
        self.step = self.replay_buffer.steps
        self.automatic_optimization=False

    def world_model_loss(self, obs, acs, rews, nonterms):
        L, B, C, H, W = obs.size()
        obs = preprocess_obs(obs)[1:]
        obs_embed = self.obs_encoder(obs.reshape(L*B, C, H, W))
        _, voc, h_enc, w_enc = obs_embed.size()
        obs_embed = obs_embed.reshape(L, B, voc, h_enc, w_enc)
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len - 1)
        target = self.posterior['tokens'][1:].permute(0, 1, 3, 4, 2).flatten(start_dim=2, end_dim=3)
        cross_entropy = -(target * prior['logits'][1:]).flatten(start_dim=2).sum(-1).mean()
        _, _, voc, h_enc, w_enc = self.posterior['tokens'].size()
        recon = self.obs_decoder(self.posterior['tokens'].reshape(L*B, voc, h_enc, w_enc))
        mse = ((obs - recon)**2).sum() / (L * B)

        features = self.posterior['slots']
        rew_dist = self.reward_model(features)
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        
        return cross_entropy, mse, rew_loss
    
    def actor_loss(self):
        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states = self.rssm.imagine_rollout(self.actor, posterior, self.args.imagine_horizon)

        self.imag_feat = imag_states['slots']

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.value_model(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts =  self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1],discounts[:-1] \
                                         ,self.args.td_lambda, imag_vals[-1])
        
        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        return actor_loss
    
    def value_loss(self):
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            discount   = self.discounts.detach()
            value_targ = self.returns.detach()
        value_dist = self.value_model(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))
        return value_loss
    
    def train_one_batch(self):
        obs, acs, rews, terms = self.replay_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device).unsqueeze(-2)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1).unsqueeze(-1)

        cross_entropy, mse, rew_loss = self.world_model_loss(obs, acs, rews, nonterms)
        self.dvae_optim.zero_grad()
        mse.backward()
        nn.utils.clip_grad_norm_(list(self.obs_encoder.parameters()) + list(self.obs_decoder.parameters()), self.args.grad_clip_norm)
        self.dvae_optim.step()

        self.rssm_optim.zero_grad()
        cross_entropy.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), self.args.grad_clip_norm)
        self.rssm_optim.step()

        self.reward_optim.zero_grad()
        rew_loss.backward()
        nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.args.grad_clip_norm)
        self.reward_optim.step()

        actor_loss = self.actor_loss()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
        self.actor_opt.step()

        value_loss = self.value_loss()
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm)
        self.value_opt.step()

        return cross_entropy.item(), mse.item(), rew_loss.item(), actor_loss.item(), value_loss.item()
    
    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):
        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        prior, posterior, attn = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = posterior['slots']
        action = self.actor(features, deter=not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)
        return  posterior, action, prior, attn
    
    def act_and_collect_data(self, collect_steps):
        obs = self.env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.args.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action, _, _ = self.act_with_world_model(obs, prev_state, prev_action, explore=True)
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = self.env.step(action)
            self.replay_buffer.add(obs, action, rew, done)

            episode_rewards[-1] += rew

            if done:
                obs = self.env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.args.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

        return np.array(episode_rewards)
    
    def collect_random_episodes(self, seed_steps):
        obs = self.env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = self.env.action_space.sample()
            next_obs, rew, done, _ = self.env.step(action)
            
            self.replay_buffer.add(obs, action, rew, done)
            seed_episode_rews[-1] += rew
            if done:
                obs = self.env.reset()
                if i!= seed_steps-1:
                    seed_episode_rews.append(0.0)
                done=False  
            else:
                obs = next_obs

        return np.array(seed_episode_rews)
    
    def save(self, save_path):
        torch.save(
            {'rssm' : self.rssm.state_dict(),
            'actor': self.actor.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.args.use_disc_model else None,
            'actor_optimizer': self.actor_opt.state_dict(),
            'value_optimizer': self.value_opt.state_dict(),
            'world_model_optimizer': self.world_model_opt.state_dict(),}, save_path)

    def restore_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        if self.args.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])

        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.value_opt.load_state_dict(checkpoint['value_optimizer'])

    def training_step(self, batch):
        cross_entropy_mean, mse_mean, rew_loss_mean, actor_loss_mean, value_loss_mean = 0, 0, 0, 0, 0
        for _ in range(self.args.update_steps):
            cross_entropy, mse, rew_loss, actor_loss, value_loss = self.update(batch)
            cross_entropy_mean += cross_entropy
            mse_mean += mse
            rew_loss_mean += rew_loss
            actor_loss_mean += actor_loss
            value_loss_mean += value_loss
        cross_entropy_mean /= self.args.update_steps
        mse_mean /= self.args.update_steps
        rew_loss_mean /= self.args.update_steps
        actor_loss_mean /= self.args.update_steps
        value_loss_mean /= self.args.update_steps
        self.log({
            'cross_entropy': cross_entropy_mean,
            'mse': mse_mean,
            'rew_loss': rew_loss_mean,
            'actor_loss': actor_loss_mean,
            'value_loss': value_loss_mean,
            'loss': cross_entropy_mean + mse_mean + rew_loss_mean + actor_loss_mean + value_loss_mean,
        })
        rews = self.act_and_collect_data(self.args.collect_steps)
        self.log({
            'mean_reward': np.mean(rews),
            'max_reward': np.max(rews),
            'min_reward': np.min(rews),
            'std_reward': np.std(rews),
        })
        return cross_entropy
    
    def validation_step(self, *args, **kwargs):
        gen_grid, imag_grid = self.eval_logs(25)
        self.log({
            'generated 1step': wandb.Video(gen_grid, fps=12, format="gif"),
            'imagined rollout': wandb.Video(imag_grid, fps=12, format="gif")
        })
        return None
    
    def eval_logs(self, collect_steps):
        observations = []
        gen_obs = []
        gen_attns = []
        imag_obs = []
        imag_attns = []

        obs = self.env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.args.action_size).to(self.device)

        episode_rewards = [0.0]
        prior = None
        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action, _, attn = self.act_with_world_model(obs, prev_state, prev_action)

                if prior is None:
                    first_step = True
                    prior = posterior
                    img_action = action
                else:
                    first_step = False
                    img_action = self.actor(prior['slots'])
                    prior, img_attn = self.rssm.imagine_step(prior, img_action)
            gen_img = self.obs_decoder(posterior['tokens'])
            imag_img = self.obs_decoder(prior['tokens'])
            observations.append(obs['image'])
            gen_obs.append(gen_img)
            gen_attns.append(attn)
            imag_obs.append(imag_img)
            if not first_step:
                imag_attns.append(img_attn)
            else:
                imag_attns.append(attn)

            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = self.env.step(action)
            episode_rewards[-1] += rew

            if done:
                obs = self.env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.args.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)
        observations = np.concatenate(observations, axis=0)
        gen_obs = np.concatenate(gen_obs.cpu().numpy(), axis=0)
        gen_attns = np.concatenate(gen_attns.cpu().numpy(), axis=0)
        imag_obs = np.concatenate(imag_obs.cpu().numpy(), axis=0)
        imag_attns = np.concatenate(imag_attns.cpu().numpy(), axis=0)

        gen_grid = visualize(observations, gen_obs, gen_attns)
        imag_grid = visualize(observations, imag_obs, imag_attns)

        return gen_grid, imag_grid
    
    def configure_optimizers(self):
        return [self.rssm_optim, self.actor_optim, self.value_optim, self.dvae_optim, self.reward_optim]


def visualize(image, recon_orig, attns, N=25):
    B, n_vecs, num_slots = attns.shape
    H_enc, W_enc = int(n_vecs**0.5), int(n_vecs**0.5)
    attns = attns.transpose(-1, -2)
    attns = attns.reshape(B, num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
    attns = recon_orig.unsqueeze(1) * attns + 1. - attns

    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    vis_recon = torch.cat((image, recon_orig, attns), dim=1)
    grids = torch.stack([vutils.make_grid(
        vis_recon[i], nrow=num_slots + 2, pad_value=0.2)[:, 2:-2, 2:-2] for i in range(vis_recon.shape[0])
    ])
    return grids
