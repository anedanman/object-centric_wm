import attrs

from configs.slotformer.slotformer_base import SlotFormerBaseConfig
from configs.slotformer.utils import register_slotformer_config


@register_slotformer_config('shapes')
class SlotFormerShapes(SlotFormerBaseConfig):
    project = "SlotFormer"
    run_name = "Slotformer Shapes Teacher Forcing Inv Loss"

    accelerator: str = 'gpu'
    devices = 1

    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'AdamW'
    lr = 1e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay
    clip_grad = 0.1

    # data settings
    dataset = 'shapes_slots'
    data_root = './data/shapes'
    slots_root = './data/shapes/slots.pkl'
    n_sample_frames = 10 + 10
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 1
    n_samples = 4

    # model configs
    slots_encoder = 'STEVE'
    resolution = (64, 64)
    input_frames = 10  # burn-in frames

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
        use_all_slots = False,
        d_model=slot_size,
        num_layers=5,
        num_heads=8,
        ffn_dim=slot_size * 4,
        norm_first=True,
        action_conditioning=True,
        use_rotary_pe = False,
        discrete_actions=True,
        actions_dim=8,
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
        dec_ckp_path='/code/checkpoint/STEVE Shapes/epoch=6-step=103907.ckpt',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # STEVE recon img is too memory-intensive
        use_inverse_actions_loss=True,
        use_inv_loss_teacher_forcing=True,
    )
    
    inverse_dict = dict(
        embedding_size=slot_size * num_slots,
        action_space_size=20,
        inverse_layers=3,
        inverse_units=64,
        inverse_ln=True
    )

    losses_weights = dict(
        slot_recon_loss=1,
        img_recon_loss=1,
    )
    

    next_actions = True
    reverse_color = True