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
