from pathlib import Path
from typing import Literal, Optional, List

from tap import Tap

from data_collection.constants import Environments


class CollectArgs(Tap):
    environment: Environments
    steps: int = 50
    seed: int = 42
    episodes: int = 1000
    split: Literal['train', 'test', 'val'] = 'train'
    black_list: Optional[List[Path]] = None
    device: Optional[str] = None

    def configure(self) -> None:
        self.add_argument('environment',
                          type=str,
                          choices=[e.value for e in Environments])