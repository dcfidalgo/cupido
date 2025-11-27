from pathlib import Path

from train import train
from data import Data, split, dataset_to_examples
from config import Cfg

from datasets import load_dataset


if __name__ == "__main__":
    cfg = Cfg()

    if cfg.data is None:
        dataset = load_dataset("llamore/plos_1000_single_page", split="train")
        data = dataset_to_examples(dataset, pdf_output_path=cfg.pdf_dir)
    else:
        data_path = Path(cfg.data)
        data = Data.model_validate_json(data_path.read_text())

    examples = data.examples
    if cfg.only_non_empty_examples:
        examples = [ex for ex in data.examples if ex.refs]

    train_data, valid_data = split(examples)

    print(cfg.model)
    train(train_data, valid_data, cfg=cfg)