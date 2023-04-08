from typing import Optional

from tap import Tap


class TrainArguments(Tap):
    method: str
    config: str
    ddp: bool = False
    fp16: bool = False
    checkpoint: Optional[str] = None