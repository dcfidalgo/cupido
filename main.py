from pathlib import Path
from typing import List, Tuple
import random

from train import train
from data import Data, Example
from config import Cfg


def split(examples: List[Example], valid_size: int = 200, seed: int = 42) -> Tuple[List[Example], List[Example]]:
    random.seed(seed)
    random.shuffle(examples)
    return examples[:-valid_size], examples[-valid_size:]

if __name__ == "__main__":
    cfg = Cfg()

    data_path = Path(cfg.data)
    data = Data.model_validate_json(data_path.read_text())

    examples = data.examples
    if cfg.only_non_empty_examples:
        examples = [ex for ex in data.examples if ex.refs]

    train_data, valid_data = split(examples)

    train(train_data, valid_data, cfg=cfg)