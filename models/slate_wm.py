import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import ReplayBuffer
from utils import FreezeParameters, compute_return

from modules.oc_rssm import OC_RSSM
from modules.ac_modules import OC_ActionDecoder, OC_DenseDecoder
from modules.slots.slate_modules import dVAE_encoder, dVAE_decoder

def preprocess_obs(obs):
    obs = obs.to(torch.float32)/255.0
    return obs

class SlateWM:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.data_buffer = ReplayBuffer(args.buffer_size, args.obs_shape, args.action_size,
                                                    args.train_seq_len, args.batch_size)
        
        self.rssm = OC_RSSM(
            action_size=args.action_size,
            slots_size=args.slots_size,
            hidden_size=args.hidden_size,
            activation=args.dense_activation_function,
            num_slots=args.num_slots,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_dec_blocks=args.num_dec_blocks,
            image_size=args.image_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_slot_heads=args.num_slot_heads,
            num_iterations=args.num_iterations,
            slot_size=args.slot_size,
            mlp_hidden_size=args.mlp_hidden_size,
            use_detach=args.use_detach).to(device)
        
        self.actor = OC_ActionDecoder(
            actison_size = self.action_size,
            slots_size=args.slot_size,
            units = self.args.num_units,
            n_layers=3,
            activation=self.args.dense_activation_function).to(self.device)
        self.obs_encoder = dVAE_encoder(args.vocab_size, args.obs_shape)
        self.obs_decoder = dVAE_decoder(args.vocab_size, args.obs_shape)
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
        
        self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters())
        self.world_model_opt = optim.Adam(self.world_model_params, self.args.model_learning_rate)
        self.value_opt = optim.Adam(self.value_model.parameters(), self.args.value_learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), self.args.actor_learning_rate)

        self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model]
        self.value_modules = [self.value_model]
        self.actor_modules = [self.actor]
        
    def world_model_loss(self, obs, acs, rews, nonterms):
        L, B, C, H, W = obs.size()
        obs = preprocess_obs(obs)[1:]
        obs_embed = self.obs_encoder(obs.reshape(L*B, C, H, W))
        obs_embed = obs_embed.reshape(L, B, C, H, W)
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len - 1)
        target = self.posterior['tokens'][1:].permute(0, 1, 3, 4, 2).flatten(start_dim=2, end_dim=3)
        cross_entropy = -(target * prior['logits'][:-1]).flatten(start_dim=2).sum(-1).mean()
        _, _, voc, h_enc, w_enc = self.posterior['tokens'].size()
        recon = self.obs_decoder(self.posterior['tokens'].reshape(L*B, voc, h_enc, w_enc))
        mse = ((obs - recon)**2).sum() / (L * B)

        features = self.posterior['slots']
        rew_dist = self.reward_model(features)
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        
        model_loss = cross_entropy + mse + rew_loss
        return model_loss
    
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
        
        # print('returns shape', self.returns.shape)
        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        # print('actor loss', actor_loss)
        # print('actor loss grad', actor_loss.grad)
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
        obs, acs, rews, terms = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device).unsqueeze(-2)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1).unsqueeze(-1)

        model_loss = self.world_model_loss(obs, acs, rews, nonterms)
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        self.world_model_opt.step()

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

        return model_loss.item(), actor_loss.item(), value_loss.item()
    
    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):
        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = posterior['slots']
        action = self.actor(features, deter=not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)
        return  posterior, action
    
    def act_and_collect_data(self, env, collect_steps):
        obs = env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action = self.act_with_world_model(obs, prev_state, prev_action, explore=True)
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, done)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

        return np.array(episode_rewards)
    
    def evaluate(self, env, eval_episodes):
        pass

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)
            
            self.data_buffer.add(obs, action, rew, done)
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
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

