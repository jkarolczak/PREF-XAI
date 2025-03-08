import random

import numpy as np
import torch

from lib import RANDOM_SEED


def set_seed(seed: int | None = None) -> None:
    if seed is None:
        seed = RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
