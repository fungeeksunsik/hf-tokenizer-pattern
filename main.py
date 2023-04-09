import pathlib
import logging
import sys

from configparser import ConfigParser
from typer import Typer

app = Typer()
config = ConfigParser()
config.read("config.ini")


@app.command(
    "preprocess",
    help="Process for downloading archived IMDb data and preprocessing it"
)
def run_preprocess():
    from preprocess import preprocess
    local_dir = pathlib.Path(config["DEFAULT"].get("local_dir"))
    local_dir.mkdir(exist_ok=True, parents=True)
    preprocess(
        local_dir=local_dir,
        source_url=config["imdb.config"].get("source_url"),
        archive_name=config["imdb.config"].get("archive_name"),
        logger=_make_logger("preprocess")
    )


@app.command(
    "train",
    help="Process for training tokenizer and language models using preprocessed data"
)
def run_train():
    from train import train
    local_dir = pathlib.Path(config["DEFAULT"].get("local_dir"))
    tokenizer_config = config["tokenizer.config"]
    train(
        local_dir=local_dir,
        pad_token_info=(
            tokenizer_config.getint("pad_token_id"), tokenizer_config.get("pad_token")
        ),
        unk_token_info=(
            tokenizer_config.getint("unk_token_id"), tokenizer_config.get("unk_token")
        ),
        cls_token_info=(
            tokenizer_config.getint("cls_token_id"), tokenizer_config.get("cls_token")
        ),
        sep_token_info=(
            tokenizer_config.getint("sep_token_id"), tokenizer_config.get("sep_token")
        ),
        mask_token_info=(
            tokenizer_config.getint("mask_token_id"), tokenizer_config.get("mask_token")
        ),
        vocab_size=tokenizer_config.getint("vocab_size"),
        result_dir_name=tokenizer_config.get("result_dir"),
        logger=_make_logger("train"),
    )


def _make_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s (%(funcName)s) : %(msg)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    app()
