from pydantic import BaseModel
from typing import Dict, List, Optional
from llamore import Reference, SchemaPrompter
from pathlib import Path
import tempfile
import pymupdf
from qwen_vl_utils import process_vision_info


class Example(BaseModel):
    file: str
    page: int
    refs: Optional[List[Reference]] = None


class Data(BaseModel):
    examples: List[Example]


def to_messages(example: Example, input_dir: str | Path, tmp_dir: str | Path) -> List[dict]:
    input_dir, tmp_dir = Path(input_dir), Path(tmp_dir)
    image_path = extract_page(example.file, example.page, input_dir, tmp_dir)

    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    text = f"""# Template:\n{SchemaPrompter().json_schema}\n#Context:\n{image_placeholder}"""
    image = {"type": "image", "image": f"file://{image_path.absolute()}"}

    messages = [
        {"role": "system", "content": SchemaPrompter().system_prompt()},
        {"role": "user", "content": [{"type": "text", "text": text}] + [image]},
    ]
    label = SchemaPrompter().schema_model(references=example.refs or []).model_dump_json(indent=4)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

    return messages


def extract_page(file: str, page: int, input_dir: Path, tmp_dir: Path, dpi: int = 100) -> Path:
    pdf_file = input_dir / file / f"{file}.pdf"
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    png_path = tmp_dir / f"{file}_{page}.png"
    doc = pymupdf.open(pdf_file)
    doc[page - 1].get_pixmap(dpi=dpi).save(png_path)
    doc.close()

    return png_path


def collate_fn(examples_dict: List[str]) -> Dict:
    examples = [Example.model_validate_json(example) for example in examples_dict]

    message_batch = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for example in examples:
            messages = to_messages(example, "/home/david/mpcdf/mplhlt/cupido/data/PLOS_1000", tmp_dir)
            message_batch.append(messages)

        user_texts = [PROCESSOR.apply_chat_template(messages[:2]) for messages in message_batch]
        full_texts = [PROCESSOR.apply_chat_template(messages) for messages in message_batch]
        images = process_vision_info(message_batch)[0]

        user_batch = PROCESSOR(text=user_texts, images=images, return_tensors="pt", padding=True)
        full_batch = PROCESSOR(text=full_texts, images=images, return_tensors="pt", padding=True)
    
    # mask padding tokens
    labels = full_batch["input_ids"].clone()
    labels[labels == PROCESSOR.tokenizer.pad_token_id] = -100

    # mask user message tokens for each example in the batch
    for i in range(len(examples)):
        # length of prompt message (accounting for possible padding)
        user_len = user_batch["attention_mask"][i].sum().item()
        
        # mask prompt part of label
        labels[i, :user_len - 1] = -100
    
    full_batch["labels"] = labels
    return full_batch


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from trl import SFTConfig, SFTTrainer
    import torch

    model_name = "numind/NuExtract-2.0-2B"

    PROCESSOR = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side='right', # make sure to set padding to right for training
        use_fast=True,
    )
    PROCESSOR.eos_token = PROCESSOR.tokenizer.eos_token
    PROCESSOR.eos_token_id = PROCESSOR.tokenizer.eos_token_id
    MODEL = AutoModelForVision2Seq.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=False, # for training
    )

    data_path = Path("./data.json")
    data = Data.model_validate_json(data_path.read_text())
    examples = [example.model_dump_json(indent=4) for example in data.examples[:10]]

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="test_finetune",  # Directory to save the model
        num_train_epochs=2,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        # gradient_accumulation_steps=1,  # Steps to accumulate gradients
        learning_rate=1e-5,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        logging_steps=1,  # Steps interval for logging
        eval_steps=2,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        # save_strategy="steps",  # Strategy for saving the model
        # save_steps=20,  # Steps interval for saving
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        report_to="none",  # Reporting tool for tracking metrics
        # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # max_seq_length=1024  # Maximum sequence length for input
    )

    # allow for proper loading of images during collation
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=MODEL,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=examples[:8],
        eval_dataset=examples[8:],
        processing_class=PROCESSOR.tokenizer,
    )
    pass