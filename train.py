from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from trl import SFTTrainer
import torch
from typing import List, Dict
from data import Example, to_messages
from qwen_vl_utils import process_vision_info
import tempfile
from pathlib import Path
from llamore import SchemaPrompter, F1
from transformers.integrations import WandbCallback
import wandb
from tqdm.auto import tqdm
from config import Cfg



class CollateFn:
    def __init__(
        self,
        processor,
        input_dir: Path | str = "/home/david/mpcdf/mplhlt/cupido/data/PLOS_1000",
        dpi: int = 100,
        include_template: bool = True,
        exclude_defaults: bool = True,
    ):
        self.processor = processor
        self.input_dir = Path(input_dir)
        self.dpi = dpi
        self.include_template = include_template
        self.exclude_defaults = exclude_defaults

    def __call__(self, examples: List[Example]) -> Dict:
        # examples = [Example.model_validate_json(example) for example in examples_dict]

        message_batch = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for example in examples:
                messages = to_messages(
                    example,
                    self.input_dir,
                    tmp_dir,
                    dpi=self.dpi,
                    include_template=self.include_template,
                    exclude_defaults=self.exclude_defaults,
                )
                message_batch.append(messages)

            user_texts = [
                self.processor.apply_chat_template(messages[:2])
                for messages in message_batch
            ]
            full_texts = [
                self.processor.apply_chat_template(messages)
                for messages in message_batch
            ]
            images = process_vision_info(message_batch)[0]

            user_batch = self.processor(
                text=user_texts, images=images, return_tensors="pt", padding=True
            )
            full_batch = self.processor(
                text=full_texts, images=images, return_tensors="pt", padding=True
            )

        # mask padding tokens
        labels = full_batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask user message tokens for each example in the batch
        for i in range(len(examples)):
            # length of prompt message (accounting for possible padding)
            user_len = user_batch["attention_mask"][i].sum().item()

            # mask prompt part of label
            labels[i, : user_len - 1] = -100

        full_batch["labels"] = labels

        return full_batch


class ComputeF1(WandbCallback):
    def __init__(self, max_new_tokens: int = 10000):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.refs_model = SchemaPrompter().schema_model
        self.f1 = F1()

    def on_evaluate(self, args, state, control, **kwargs):
        model, tokenizer = kwargs["model"], kwargs["processing_class"]

        generation_table = wandb.Table(columns=["generation", "label"])

        gold_references, predicted_references = [], []
        total = 25  # len(kwargs["eval_dataloader"])
        for i, batch in tqdm(
            enumerate(kwargs["eval_dataloader"]),
            total=total,
            leave=False,
            desc="Generating",
        ):
            # TODO: generate batch-wise
            if batch["input_ids"].shape[0] > 1:
                raise ValueError("Batch size > 1 is not supported for generation")

            # Generate output
            idx = (batch["labels"][0] != -100).nonzero()[0][0].item()
            batch["input_ids"] = batch["input_ids"][:, :idx + 1]
            batch["attention_mask"] = batch["attention_mask"][:, :idx + 1]
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=self.max_new_tokens)
            output = tokenizer.decode(
                output[0][idx:], skip_special_tokens=True
            )  # Only keep the generated tokens
            idx = output.find("{")
            try:
                refs = self.refs_model.model_validate_json(output[idx:].strip())
            except Exception:
                references = []
            else:
                references = refs.references
            predicted_references.append(references)

            # Get gold references
            labels = batch["labels"][0]
            label = tokenizer.decode(labels[labels != -100], skip_special_tokens=True)
            idx = label.find("{")
            refs = self.refs_model.model_validate_json(label[idx:].strip())
            gold_references.append(refs.references)

            generation_table.add_data(output, label)

            if i == 24:
                break

        f1 = self.f1.compute_macro_average(
            predicted_references, gold_references, show_progress=False
        )

        self._wandb.log({"f1": f1, "generation_table": generation_table})


def train(
    train_data: List[Example],
    valid_data: List[Example],
    cfg: Cfg,
):
    model_name, config = cfg.model, None

    if cfg.is_mock_model:
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        # make model really small for testing
        config = AutoConfig.from_pretrained(model_name)
        config.text_config.update(
            {
                "num_attention_heads": 3,
                "num_hidden_layers": 1,
                "head_dim": 4,
                "hidden_size": 4,
                "intermediate_size": 4,
            }
        )
        config.vision_config.update(
            {
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 4,
                "hidden_size": 4,
            }
        )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # make sure to set padding to right for training
        use_fast=True,
    )
    processor.eos_token = processor.tokenizer.eos_token
    processor.eos_token_id = processor.tokenizer.eos_token_id

    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if cfg.use_flashattn else "eager",
        device_map="auto",
        # use_cache=False, # for training,
        config=config,
        ignore_mismatched_sizes=cfg.is_mock_model,
        # quantization_config=bnb_config
    )
    # print("PARAMETERS:", sum(p.numel() for p in model.parameters())/1e6, "M")

    training_args = cfg.sft_cfg

    # allow for proper loading of images during collation
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # get_peft_model(model, cfg.lora_cfg).print_trainable_parameters()

    # Create the data collator
    data_collator = CollateFn(
        processor=processor,
        input_dir=cfg.pdf_dir,
        dpi=100 if not cfg.is_mock_model else 1,
        include_template=True if not cfg.is_mock_model else False,
        exclude_defaults=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=processor.tokenizer,
        peft_config=cfg.lora_cfg if cfg.use_lora else None,
        callbacks=[ComputeF1()] if cfg.use_f1_callback else None,
    )

    trainer.train()


if __name__ == "__main__":
    from pathlib import Path
    from data import Data

    data_path = Path("./data/data.json")
    data = Data.model_validate_json(data_path.read_text())
    examples = data.examples
    examples = [ex for ex in data.examples if ex.refs]
    train_data, valid_data = examples[:-200], examples[-200:]

    train(train_data, valid_data, cfg=Cfg())
