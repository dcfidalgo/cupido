import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pymupdf
from llamore import Reference, SchemaPrompter, References
from pydantic import BaseModel

import datasets

from rich.progress import track


class Example(BaseModel):
    file: str
    page: Optional[int] = None
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
    if example.refs is not None:
        label = (
            SchemaPrompter()
            .schema_model(references=example.refs)
            .model_dump_json(indent=indent, exclude_defaults=exclude_defaults)
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

    return messages


def extract_page(
    file: str, page: int | None, input_dir: Path, tmp_dir: Path, dpi: int = 100
) -> Path:
    if page is None:
        pdf_file = input_dir / file
        png_path = tmp_dir / f"{Path(file).stem}.png"
        page = 1
    else:
        pdf_file = input_dir / file / f"{file}.pdf"
        png_path = tmp_dir / f"{file}_{page}.png"

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")


    doc = pymupdf.open(pdf_file)
    doc[page - 1].get_pixmap(dpi=dpi).save(png_path)
    doc.close()

    return png_path


def export_page_as_pdf(
    file: Union[str, Path], page: int, output: Union[str, Path]
) -> Path:
    file = Path(file)
    doc = pymupdf.open(file)
    new_doc = pymupdf.open()
    new_doc.insert_pdf(doc, from_page=page - 1, to_page=page - 1)

    output = Path(output)
    new_doc.save(output)

    return output


def split(
    examples: List[Example],
    only_with_refs: bool = False,
    valid_size: int = 200,
    seed: int = 42,
) -> Tuple[List[Example], List[Example]]:
    if only_with_refs:
        examples = [ex for ex in examples if ex.refs]

    random.seed(seed)
    random.shuffle(examples)
    return examples[:-valid_size], examples[-valid_size:]


def dataset_to_examples(dataset: datasets.Dataset, pdf_output_path: str | Path) -> Data:
    """
    Convert a Hugging Face dataset to a Data object containing examples.

    Args:
        dataset (datasets.Dataset): A Hugging Face dataset containing PDF files and references.
            Each row should have:
            - 'pdf': A dictionary with 'path' and 'bytes' keys
            - 'references': Optional XML string of references
        pdf_output_path (str | Path): Directory path where the single-page PDF files will be saved.

    Returns:
        Data: A Data object containing a list of Example instances, where each Example
            has a file path and optional References object.

    Note:
        - PDF files are only written to disk if they don't already exist at the target location
        - References are parsed from XML format if present in the dataset row
    """
    examples, pdf_output_path = [], Path(pdf_output_path)
    for row in track(dataset, description="Converting dataset and exporting PDFs"):
        output_path = pdf_output_path / row["pdf"]["path"]
        if not output_path.exists():
            output_path.write_bytes(row["pdf"]["bytes"])
        
        file = row["pdf"]["path"]
        refs = None
        if row["references"]:
            refs = References.from_xml(xml_str=row["references"])

        examples.append(Example(file=file, refs=refs))

    return Data(examples=examples)