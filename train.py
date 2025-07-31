from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from trl import SFTConfig, SFTTrainer
import torch
from typing import List, Dict
from data import Example, to_messages
from qwen_vl_utils import process_vision_info
import tempfile
from pathlib import Path
from llamore import SchemaPrompter, F1
from transformers.integrations import WandbCallback
import wandb


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
    def __init__(self, max_new_tokens: int = 1000):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.refs_model = SchemaPrompter().schema_model
        self.f1 = F1()

    def on_evaluate(self, args, state, control, **kwargs):
        model, tokenizer = kwargs["model"], kwargs["processing_class"]

        generation_table = wandb.Table(columns=["generation", "label"])

        gold_references, predicted_references = [], []
        for batch in kwargs["eval_dataloader"]:
            # TODO: generate batch-wise
            for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
                # Generate output
                idx = (labels != -100).nonzero()[0][0].item()
                output = model.generate(input_ids[:idx].unsqueeze(0), max_new_tokens=self.max_new_tokens)
                output = tokenizer.decode(output[0][idx:], skip_special_tokens=True) # Only keep the generated tokens
                idx = output.find("{")
                try:
                    refs = self.refs_model.model_validate_json(output[idx:].strip())
                except Exception:
                    references = []
                else: 
                    references = refs.references
                predicted_references.append(references)

                # Get gold references
                label = tokenizer.decode(labels[labels != -100], skip_special_tokens=True)
                idx = label.find("{")
                refs = self.refs_model.model_validate_json(label[idx:].strip())
                gold_references.append(refs.references)

                generation_table.add_data(output, label)

        f1 = self.f1.compute_macro_average(
            predicted_references, gold_references, show_progress=False
        )

        self._wandb.log({"f1": f1, "generation_table": generation_table})


def train(
    train_data: List[Example],
    valid_data: List[Example],
    input_dir: Path,
    is_mock_model: bool = False,
    use_flashattn: bool = True,
):
    model_name, config = "numind/NuExtract-2.0-2B", None

    if is_mock_model:
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
        attn_implementation="flash_attention_2" if use_flashattn else "eager",
        device_map="auto",
        # use_cache=False, # for training,
        config=config,
        ignore_mismatched_sizes=is_mock_model,
        # quantization_config=bnb_config
    )
    # print("PARAMETERS:", sum(p.numel() for p in model.parameters())/1e6, "M")

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="test_finetune",  # Directory to save the model
        num_train_epochs=1,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        # gradient_accumulation_steps=1,  # Steps to accumulate gradients
        learning_rate=1e-5,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        logging_steps=1,  # Steps interval for logging
        eval_steps=1,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        # save_strategy="steps",  # Strategy for saving the model
        # save_steps=20,  # Steps interval for saving
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        report_to="wandb",  # Reporting tool for tracking metrics
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Options for gradient checkpointing
        # eval_accumulation_steps=1,
    )

    # allow for proper loading of images during collation
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # from peft import LoraConfig, get_peft_model
    # peft_config = LoraConfig(
    #     lora_alpha=16,  # Scaling factor for LoRA
    #     lora_dropout=0.05,  # Dropout rate for LoRA layers
    #     r=8,  # Rank of the low-rank decomposition
    #     bias="none",  # Bias handling in LoRA layers
    #     target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    #     task_type="CAUSAL_LM",  # Task type for the model
    # )
    # get_peft_model(model, peft_config).print_trainable_parameters()

    # def preprocess_logits_for_metrics(logits, labels):
    #     # This function is used to preprocess logits for metrics computation
    #     return logits[0].argmax(axis=-1)

    # Create the data collator
    data_collator = CollateFn(
        processor=processor,
        input_dir=input_dir,
        dpi=100 if not is_mock_model else 1,
        include_template=True if not is_mock_model else False,
        exclude_defaults=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=processor.tokenizer,
        # compute_metrics=ComputeF1(processor.tokenizer),
        # peft_config=peft_config,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        callbacks=[ComputeF1()],
    )

    trainer.train()


if __name__ == "__main__":
    from pathlib import Path
    from data import Data

    data_path = Path("./data/data.json")
    data = Data.model_validate_json(data_path.read_text())
    examples = data.examples
    # examples = [ex for ex in data.examples if ex.refs]
    train_data, valid_data = data.examples[:2], data.examples[2:4]

    # # pdfs_dir = Path("/u/dcfidalgo/projects/cupido/data/PLOS_1000")
    pdfs_dir = Path("/home/david/mpcdf/mplhlt/cupido/data/PLOS_1000")
    train(train_data, valid_data, pdfs_dir, is_mock_model=True, use_flashattn=False)
