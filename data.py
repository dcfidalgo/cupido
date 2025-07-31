from pydantic import BaseModel
from typing import List, Optional
from llamore import Reference, SchemaPrompter
from pathlib import Path
import pymupdf


class Example(BaseModel):
    file: str
    page: int
    refs: Optional[List[Reference]] = None


class Data(BaseModel):
    examples: List[Example]


def to_messages(
    example: Example,
    input_dir: str | Path,
    tmp_dir: str | Path,
    indent: int = 4,
    dpi: int = 100,
    include_template: bool = True,
    exclude_defaults: bool = True,
) -> List[dict]:
    input_dir, tmp_dir = Path(input_dir), Path(tmp_dir)
    image_path = extract_page(example.file, example.page, input_dir, tmp_dir, dpi=dpi)

    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    text = ""
    if include_template:
        text += f"""# Template:\n{SchemaPrompter().json_schema}\n"""
    text += f"""#Context:\n{image_placeholder}"""
    image = {"type": "image", "image": f"file://{image_path.absolute()}"}

    messages = [
        {"role": "system", "content": SchemaPrompter().system_prompt()},
        {"role": "user", "content": [{"type": "text", "text": text}] + [image]},
    ]
    label = (
        SchemaPrompter()
        .schema_model(references=example.refs or [])
        .model_dump_json(indent=indent, exclude_defaults=exclude_defaults)
    )
    messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

    return messages


def extract_page(
    file: str, page: int, input_dir: Path, tmp_dir: Path, dpi: int = 100
) -> Path:
    pdf_file = input_dir / file / f"{file}.pdf"
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    png_path = tmp_dir / f"{file}_{page}.png"
    doc = pymupdf.open(pdf_file)
    doc[page - 1].get_pixmap(dpi=dpi).save(png_path)
    doc.close()

    return png_path
