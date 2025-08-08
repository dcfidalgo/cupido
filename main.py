from pathlib import Path

from train import train
from data import Data, split
from config import Cfg


if __name__ == "__main__":
    cfg = Cfg()

    data_path = Path(cfg.data)
    data = Data.model_validate_json(data_path.read_text())

    examples = data.examples
    if cfg.only_non_empty_examples:
        examples = [ex for ex in data.examples if ex.refs]

    train_data, valid_data = split(examples)

    train(train_data, valid_data, cfg=cfg)