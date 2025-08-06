from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource
)
from pydantic import Field
import os
from trl import SFTConfig as SFTConfigOriginal
from peft import LoraConfig as LoraConfigOriginal
from pydantic.dataclasses import dataclass


# Making them pydantic dataclasses allows for better compatibility with Pydantic's BaseSettings.
# Inheriting from pydantic's BaseModel leads to errors for certain default values (dict).
@dataclass
class SFTConfig(SFTConfigOriginal):
    ...


@dataclass
class LoraConfig(LoraConfigOriginal):
    ...


class Cfg(BaseSettings):
    GEMINI_API_KEY: str = Field(description="The API key for Gemini.", default="")

    model: str = "numind/NuExtract-2.0-2B"
    data: str = "./data/data.json"
    pdf_dir: str = "data/PLOS_1000"
    dpi: int = 100
    only_non_empty_examples: bool = True
    is_mock_model: bool = False
    use_flashattn: bool = True
    use_lora: bool = True
    lora_cfg: LoraConfig = LoraConfig()
    use_f1_callback: bool = False

    sft_cfg: SFTConfig = SFTConfig()


    model_config = SettingsConfigDict(
        env_file=".env",
        cli_parse_args=True,
        toml_file=os.getenv("CUPIDO_TOML", "cupido.toml"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = super().settings_customise_sources(
            settings_cls,
            **kwargs,
        )
        return sources + (TomlConfigSettingsSource(settings_cls),)
