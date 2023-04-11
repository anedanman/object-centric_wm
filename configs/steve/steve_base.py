from configs.core.training_config import TrainingConfig


class STEVEBaseConfig(TrainingConfig):
    project = "SlotFormer"
    run_name = "STEVE Base"

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

    dec_lr = 3e-5
    inv_lr = 1e-4

    warmup_steps_pct = 0.05
    clip_grad = 0.08

    # model configs
    resolution = (64, 64)
    vocab_size = 4096  # codebook size

    # temperature for gumbel softmax
    # decay from 1.0 to 0.1 in the first 15% of total steps
    init_tau = 1.
    final_tau = 0.1
    tau_decay_pct = 0.15

    resolution = (64, 64)
    input_frames = n_sample_frames

    # Slot Attention
    slot_size = 128
    slot_dict = dict(
        # the objects are harder to define in Physion than in e.g. CLEVRER
        # e.g. should a stack of 6 boxes be considered as 1 or 6 objects?
        #      among these boxes, some move together, some usually fall, etc.
        # we don't ablate this `num_slots`, so we are not sure if it will make
        # a huge difference to the results
        # qualitatively, 6 achieves a reasonable scene decomposition result
        num_slots=6,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=1,
        slots_init='param',
        truncate='none',
        sigma=1
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=526,
        dvae_ckp_path='checkpoint/dvae_shapes_params/models/epoch/model_20.pth',
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        atten_type='linear'
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_size * 4,
        pred_sg_every=None,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=False,  # additional img recon loss via dVAE decoder
        use_slots_correlation_loss=False,
        use_cossine_similarity_loss=False,
        use_inverse_actions_loss=True,
    )

    inverse_dict = dict(
        embedding_size=slot_size * slot_dict['num_slots'],
        action_space_size=20,
        inverse_layers=3,
        inverse_units=64,
        inverse_ln=True
    )

    losses_weights = {}

    next_actions = False
    recon_video = False

    def get_model_config(self):
        return dict(
            resolution=self.resolution,
            clip_len=self.input_frames,
            inverse_dict=self.inverse_dict,
            loss_dict=self.loss_dict,
            pred_dict=self.pred_dict,
            dec_dict=self.dec_dict,
            enc_dict=self.enc_dict,
            dvae_dict=self.dvae_dict,
            slot_dict=self.slot_dict,
        )



