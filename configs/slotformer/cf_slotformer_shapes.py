import attrs

from configs.slotformer.slotformer_base import SlotFormerBaseConfig
from configs.slotformer.utils import register_slotformer_config


@register_slotformer_config('shapes')
class SlotFormerShapes(SlotFormerBaseConfig):
    project = "SlotFormer"
    run_name = "Slotformer Shapes Baseline"

    accelerator: str = 'gpu'
    devices = 1

    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay
    clip_grad = 0.08

    # data settings
    dataset = 'shapes_slots'
    data_root = './data/shapes'
    slots_root = './data/shapes/slots.pkl'
    n_sample_frames = 15 + 10
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 32
    val_batch_size = 32
    num_workers = 1
    n_samples = 4

    # model configs
    slots_encoder = 'STEVE'
    resolution = (64, 64)
    input_frames = 15  # burn-in frames

    num_slots = 6
    slot_size = 128
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
    )

    # Rollouter
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        act_pe='sin',
        # Transformer-related configs
        d_model=slot_size,
        num_layers=2,
        num_heads=8,
        ffn_dim=slot_size * 4,
        norm_first=True,
        action_conditioning=True,
        discrete_actions=True,
        actions_dim=16,
        max_discrete_actions=20
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='checkpoint/dvae_shapes_params/models/epoch/model_20.pth',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        atten_type='linear',
        dec_ckp_path='checkpoint/steve_shapes_params/models/epoch/model_4.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=True,  # STEVE recon img is too memory-intensive
    )

    losses_weights = dict(
        slot_recon_loss=1,
        img_recon_loss=1,
    )

    next_actions = True