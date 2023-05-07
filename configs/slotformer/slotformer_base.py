import attrs

from configs.core.training_config import TrainingConfig


class SlotFormerBaseConfig(TrainingConfig):
    project = "SlotFormer"
    run_name = "Slotformer Base"

    # data settings
    dataset = 'shapes_slots'
    data_root = './data/shapes'
    slots_root = './data/shapes/slots.pkl'
    n_sample_frames = 6 + 10  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 1
    n_samples = 4

    warmup_steps_pct = 0.05

    # model configs
    slots_encoder = 'STEVE'
    resolution = (64, 64)
    input_frames = 1  # burn-in frames

    pretrained = ''

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
        num_layers=2,
        num_heads=8,
        ffn_dim=slot_size * 4,
        norm_first=True,
        use_rotary_pe = False,
        action_conditioning=True,
        discrete_actions=True,
        actions_dim=16,
        max_discrete_actions=20
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        dec_ckp_path='',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=True,  # STEVE recon img is too memory-intensive
        use_inverse_actions_loss=True,
        use_inv_loss_teacher_forcing=False,
    )

    losses_weights = dict(
        slot_recon_loss=1,
        img_recon_loss=1,
    )

    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='',
    )
    
    inverse_dict = dict(
        embedding_size=slot_size * slot_dict['num_slots'],
        action_space_size=20,
        inverse_layers=3,
        inverse_units=64,
        inverse_ln=True
    )

    loss_decay_pct: int = 0

    reverse_color = True

    def get_model_config_dict(self):
        cfg = dict(
            resolution=self.resolution,
            clip_len=self.input_frames,
            slot_dict=self.slot_dict,
            rollout_dict=self.rollout_dict,
            dec_dict=self.dec_dict,
            loss_dict=self.loss_dict,
            dvae_dict=self.dvae_dict,
            inverse_dict=self.inverse_dict
        )

        if self.slots_encoder.upper() == "STEVE":
            cfg['dvae_dict'] = self.dvae_dict
        return cfg

