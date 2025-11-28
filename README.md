# Cupido

## Quick start

```bash
uv sync
python ./main.py --help
python ./main.py
```

## Config

By default, the `main.py` script reads the `cupido.toml` file as config file in the same directory.
You can specify the path via the env variable `CUPIDO_TOML`.

You can overwrite all configuration in the CLI (see `main.py --help`).

If no `--data` is provided, it will download the `llamore/plos_1000_single_page` dataset from the HF hub.
Make sure you are a member of the llamore organization, since this dataset is private.