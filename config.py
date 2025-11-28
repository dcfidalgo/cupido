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
    """
    Configuration class for the Cupido application.

    This class manages configuration settings for the Cupido machine learning pipeline,
    including API keys, model parameters, data paths, and training configurations.

    Attributes:
        GEMINI_API_KEY (str): The API key for accessing Gemini services. Defaults to empty string.
        model (str): The model identifier to use. Defaults to "numind/NuExtract-2.0-2B".
        data (None | str): Path to the data source. If None, we will download 
            the `llamore/plos_1000_single_page` dataset and export it to `pdf_dir`.
        pdf_dir (str): Directory containing PDF files. Defaults to "data/plos_1000_single_page".
        dpi (int): DPI resolution for processing. Defaults to 100.
        only_non_empty_examples (bool): Whether to use only non-empty examples. Defaults to True.
        is_mock_model (bool): Flag to use a mock model for testing. Defaults to False.
        use_flashattn (bool): Whether to use Flash Attention optimization. Defaults to True.
        use_lora (bool): Whether to use LoRA (Low-Rank Adaptation). Defaults to True.
        lora_cfg (LoraConfig): Configuration object for LoRA settings.
        use_f1_callback (bool): Whether to enable F1 score callback during training. Defaults to False.
        nr_for_f1_callback (int): Number of samples for F1 callback evaluation. Defaults to 25.
        sft_cfg (SFTConfig): Configuration object for supervised fine-tuning settings.

    Configuration Sources:
        Settings are loaded from multiple sources in the following priority:
        1. Environment variables
        2. Command-line arguments
        3. .env file
        4. TOML configuration file (specified by CUPIDO_TOML environment variable or defaults to "cupido.toml")

    Methods:
        settings_customise_sources: Customizes the configuration sources to include TOML file support.
    """
    GEMINI_API_KEY: str = Field(description="The API key for Gemini.", default="")

    model: str = "numind/NuExtract-2.0-2B"
    data: None | str = None
    pdf_dir: str = "data/plos_1000_single_page"
    dpi: int = 100
    only_non_empty_examples: bool = True
    is_mock_model: bool = False
    use_flashattn: bool = True
    use_lora: bool = True
    lora_cfg: LoraConfig = LoraConfig()
    use_f1_callback: bool = False
    nr_for_f1_callback: int = 25

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
