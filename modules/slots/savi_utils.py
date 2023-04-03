import torch
import torch.nn as nn


def get_rnn(rnn_cell: str, input_size: int, hidden_size: int, num_layers: int):
    rnn_cell = rnn_cell.upper()
    assert rnn_cell in ['LSTM', 'GRU', 'RNN']
    if rnn_cell == 'LSTM':
        return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif rnn_cell == 'GRU':
        return nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif rnn_cell == 'RNN':
        return nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f'RNN cell "{rnn_cell}" is not supported!')


def get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
):
    """Get Conv layer."""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=bias)


def deconv_norm_act(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        norm='bn',
        act='relu',
):
    """ConvTranspose - Norm - Act."""
    deconv = get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'])
    normalizer = get_normalizer(norm, out_channels)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)


def get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
):
    """Get Conv layer."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias)


def get_normalizer(norm, channels, groups=16):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return nn.BatchNorm2d(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return nn.InstanceNorm2d(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')


def deconv_out_shape(
        in_size,
        stride,
        padding,
        kernel_size,
        out_padding,
        dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1


def deconv_out_shape(
        in_size,
        stride,
        padding,
        kernel_size,
        out_padding,
        dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1


def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')


def conv_norm_act(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        norm='bn',
        act='relu',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in']
    )
    normalizer = get_normalizer(norm, out_channels)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)



def get_lr(optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']


def torch_stack(tensor_list, dim):
    if len(tensor_list[0].shape) < dim:
        return torch.stack(tensor_list)
    return torch.stack(tensor_list, dim=dim)


def torch_cat(tensor_list, dim):
    if len(tensor_list[0].shape) <= dim:
        return torch.cat(tensor_list)
    return torch.cat(tensor_list, dim=dim)


def clip_tensor_norm(tensor, norm, dim=-1, eps=1e-6):
    """Clip the norm of tensor along `dim`."""
    assert norm > 0.
    tensor_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale_factor = norm / (tensor_norm + eps)
    scale_factor = torch.clip(scale_factor, max=1.)
    clip_tensor = tensor * scale_factor
    return clip_tensor


def assert_shape(actual, expected, message=""):
    assert list(actual) == list(expected), \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def to_rgb_from_tensor(x):
    """Reverse the Normalize operation in torchvision."""
    return (x * 0.5 + 0.5).clamp(0, 1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj
