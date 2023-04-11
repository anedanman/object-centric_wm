import attrs

from configs.core.training_config import TrainingConfig
from configs.dvae.utils import register_dvae_config
from configs.dvae.dvae_base import DVAEBaseConfig


@register_dvae_config('shapes')
class DVAEShapesConfig(DVAEBaseConfig):
    project = "SlotFormer"
    run_name = "DVAE Inverse Actions"

    accelerator: str = 'gpu'
    devices = 1

    # data settings
    dataset = 'shapes'
    data_root = './data/shapes'
    n_sample_frames = 6
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 128
    val_batch_size = 128
    num_workers = 1
    n_samples = 4
    max_epochs = 10

    warmup_steps_pct = 0.05

    # model configs
    model = 'dVAE'
    resolution = (64, 64)
    vocab_size = 512  # codebook size

    # temperature for gumbel softmax
    # decay from 1.0 to 0.1 in the first 15% of total steps
    init_tau = 1.
    final_tau = 0.1
    tau_decay_pct = 0.15

    loss_dict = dict(
        use_inverse_actions_loss=True
    )

    next_actions = True
    reverse_color = True