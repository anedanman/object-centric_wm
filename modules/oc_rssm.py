import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from modules.dynamics.dynam import DynamicsTransformer
from modules.slots.slate_modules import TransformerDecoder, PositionalEncoding
from modules.slots.slot_attn import SlotAttentionEncoder
from utils.gumbel import gumbel_softmax

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class OC_RSSM(nn.Module):

    def __init__(self, 
                action_size=32, 
                slots_size=64,  
                hidden_size=64, 
                activation='relu',
                num_slots=5,
                vocab_size=64,
                d_model=64,
                num_dec_blocks=2,
                image_size=64,
                num_heads=4,
                dropout=0.1,
                num_slot_heads=1,
                num_iterations=3,
                slot_size=64,
                mlp_hidden_size=64,
                use_detach=True):

        super().__init__()

        self.action_size = action_size
        self.slots_size  = slots_size
        self.hidden_size = hidden_size  
        self.num_slots = num_slots
        self.vocab_size = vocab_size

        self.act_fn = _str_to_activation[activation]
        self.oc_rnn = DynamicsTransformer(
            self.slots_size, 
            n_heads=4, 
            d_head=self.slots_size // 4,
            depth=1,
            dropout=0.1,
            context_dim=self.slots_size
        )
        #slate modules
        self.image_size = image_size
        self.gen_len = (image_size // 4) ** 2
        self.dictionary = OneHotDictionary(vocab_size + 1, d_model)
        self.tf_dec = TransformerDecoder(
            num_dec_blocks, self.gen_len, d_model, num_heads, dropout)
        self.slot_attn = SlotAttentionEncoder(
            num_iterations, num_slots,
            d_model, slot_size, mlp_hidden_size,
            num_slot_heads, use_detach)
        self.positional_encoder = PositionalEncoding(1 + (image_size // 4) ** 2, d_model, dropout)

        self.fc_state_action = nn.Linear(self.action_size, self.slots_size)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def init_state(self, batch_size, device):

        return dict(
            prior  = torch.tensor([]).to(device),
            tokens = torch.tensor([]).to(device),
            slots = None)

    def get_dist(self, mean, std):
        distribution = distributions.Normal(mean, std)
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)

        z_logits = F.log_softmax(obs_embed, dim=1)
        B, _, H_enc, W_enc = z_logits.size()
        z_hard = gumbel_softmax(z_logits, hard=True, dim=1).detach()
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)
        z_transformer_input[:, 0, 0] = 1.0

        emb_input = self.dictionary(z_transformer_input)
        emb_input = self.positional_encoder(emb_input)

        slots, attns = self.slot_attn(emb_input[:, 1:], slots=prior['slots'])
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B, self.num_slots, 1, H_enc, W_enc).repeat_interleave(self.image_size // H_enc, dim=-2).repeat_interleave(self.image_size // W_enc, dim=-1)

        posterior = {'logits': z_logits, 'tokens': z_hard, 'slots': slots}
        return prior, posterior, attns

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):
        # prev_action = prev_action.reshape(*prev_state['slots'].shape[:-2], 1, -1)
        prev_action = torch.unsqueeze(prev_action, dim=-2)
        state_action = self.act_fn(self.fc_state_action(prev_action))
            
        slots = self.oc_rnn(
            context=state_action, 
            x=prev_state['slots']*nonterm if prev_state['slots'] is not None else prev_action.new_zeros((prev_action.shape[0], self.num_slots, self.slots_size))
        )
        z_gen, logits = self.slots2tokens(slots)
        prior = {'tokens': z_gen, 'slots': slots, 'logits': logits}
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t]* nonterms[t]
            prior_state, posterior_state, _ = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon):
        rssm_state = prev_state
        next_states = []

        for t in range(horizon):
            action = actor(rssm_state['slots'].detach())
            rssm_state = self.imagine_step(rssm_state, action.view(-1, 1, action.shape[-1]))
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        return next_states
    
    def slots2tokens(self, slots):
        B, _, _ = slots.size()
        z_gen = slots.new_zeros(0)
        logits = slots.new_zeros(0)
        z_transformer_input = z_gen.new_zeros(B, 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0
        for t in range(self.gen_len):
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)),
                slots
            )
            logits_next = self.out(decoder_output)[:, -1:]
            z_next = F.one_hot(logits_next.argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            logits = torch.cat((logits, logits_next), dim=1)
            z_transformer_input = torch.cat([
                z_transformer_input,
                torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
            ], dim=1)
        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, self.image_size//4, self.image_size//4)
        return z_gen, torch.log_softmax(logits, dim=-1)
    
    def stack_states(self, states, dim=0):

        return dict(
            logits  = torch.stack([state['logits'] for state in states], dim=dim),
            tokens = torch.stack([state['tokens'] for state in states], dim=dim),
            slots = torch.stack([state['slots'] for state in states], dim=dim))

    def detach_state(self, state):

        return dict(
            logits  = state['logits'].detach(),
            tokens = state['tokens'].detach(),
            slots = state['slots'].detach())

    def seq_to_batch(self, state):

        return dict(
            logits = torch.reshape(state['logits'], (state['logits'].shape[0]* state['logits'].shape[1], *state['logits'].shape[2:])),
            tokens = torch.reshape(state['tokens'], (state['tokens'].shape[0]* state['tokens'].shape[1], *state['tokens'].shape[2:])),
            slots = torch.reshape(state['slots'], (state['slots'].shape[0]* state['slots'].shape[1], *state['slots'].shape[2:])))


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs
