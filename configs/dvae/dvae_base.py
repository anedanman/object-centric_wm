from configs.core.training_config import TrainingConfig


class DVAEBaseConfig(TrainingConfig):
    project = "SlotFormer"
    run_name = "DVAE Base"

    # data settings
    dataset = 'shapes'
    data_root = './data/shapes'
    n_sample_frames = 1  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 32
    val_batch_size = 32
    num_workers = 1
    n_samples = 4
    max_epochs = 10

    warmup_steps_pct = 0.05

    # model configs
    model = 'dVAE'
    resolution = (64, 64)
    vocab_size = 4096  # codebook size

    # temperature for gumbel softmax
    # decay from 1.0 to 0.1 in the first 15% of total steps
    init_tau = 1.
    final_tau = 0.1
    tau_decay_pct = 0.15

