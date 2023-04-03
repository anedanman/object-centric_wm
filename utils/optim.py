import torch.optim


def get_optimizer(name: 'str'):
    l_name = name.lower()
    if l_name == 'adam':
        return torch.optim.Adam
    elif l_name.lower() == 'adamw':
        return torch.optim.AdamW
    else:
        raise NotImplementedError(f'Optimizer {name} is not implemented')