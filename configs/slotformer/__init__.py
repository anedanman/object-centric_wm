from configs.core.config_registry import create_category
from utils.get_all_modules import get_all_modules

__all__ = get_all_modules(__file__)

register_slotformer_config, get_slotformer_config = create_category('slotformer')
