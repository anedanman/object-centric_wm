from attrs import define

from configs.core.training_config import TrainingConfig


@define(kw_only=True)
class SlotFormerBaseConfig(TrainingConfig):
    project = "SlotFormer"
    run_name = "Slotformer Base"

    # data settings
    dataset = 'shapes_slots'
    data_root = './data/shapes'
    slots_root = './data/shapes/slots.pkl'
    n_sample_frames = 15 + 10  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 100
    train_batch_size = 32
    val_batch_size = 32
    num_workers = 1
    n_samples=4

    warmup_steps_pct = 0.05

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
    )

    losses_weights = dict(
        slot_recon_loss=1,
        img_recon_loss=1,
    )

    loss_decay_pct: int = 0

    def get_model_config_dict(self):
        return dict(
            resolution=self.resolution,
            clip_len=self.clip_grad,
            slot_dict=self.slot_dict,
            rollout_dict=self.rollout_dict,
            dec_dict=self.dec_dict,
            loss_dict=self.loss_dict,
        )
