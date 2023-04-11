from configs.slotformer.slotformer_base import SlotFormerBaseConfig
from models.slotformer import SlotFormer
from models.steve_slotformer import STEVESlotFormer


def get_slotformer(config: SlotFormerBaseConfig):
    slots_encoder = config.slots_encoder
    slots_encoder.upper()
    params = config.get_model_config_dict()
    if slots_encoder == 'STEVE':
        return STEVESlotFormer(**params)
    elif slots_encoder == 'SAVI':
        return SlotFormer(**params)
    else:
        raise ValueError(f"Slot encoder {slots_encoder} is not supported")
    