from itertools import chain

import torch
from torch import nn


def make_mlp_block(in_units, out_units, bn):
    """
    Builds a simple dense block with ReLU activation and (optional) Batch Normalization.
    """
    return [nn.Linear(in_units, out_units), nn.LayerNorm(out_units), nn.ReLU()] if bn else [nn.Linear(in_units, out_units), nn.ReLU()]


class InverseModel(nn.Module):
    """
    Inverse model mapping a pair of state embeddings to the action that takes from the first one to the second one.
    """
    def __init__(self, embedding_size, action_space_size, inverse_layers, inverse_units, inverse_ln):
        super(InverseModel, self).__init__()
        self.embedding_size = embedding_size
        self.action_space_size = action_space_size
        self.layers = inverse_layers
        self.units = inverse_units
        self.ln = inverse_ln
        if self.layers < 2:
            self.dense = nn.Sequential(*make_mlp_block(2*self.embedding_size, self.action_space_size, self.ln))
        else:
            self.dense = nn.Sequential(*(
                    make_mlp_block(2 * self.embedding_size, self.units, self.ln) +
                    list(chain.from_iterable(make_mlp_block(self.units, self.units, self.ln) for _ in range(self.layers-2))) +
                    [nn.Linear(self.units, self.action_space_size)]))
        self.is_frozen = False

    def forward(self, start_state, target_state):
        x = torch.cat([start_state, target_state], -1)
        return self.dense(x)