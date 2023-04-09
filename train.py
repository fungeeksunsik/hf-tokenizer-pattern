import logging
import pathlib
import pandas as pd

from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from typing import Tuple


def train(
    local_dir: pathlib.Path,
    result_dir_name: str,
    pad_token_info: Tuple[int, str],
    unk_token_info: Tuple[int, str],
    cls_token_info: Tuple[int, str],
    sep_token_info: Tuple[int, str],
    mask_token_info: Tuple[int, str],
    vocab_size: int,
    logger: logging.Logger,
):
    logger.info("Extract reviews from train data and save into local directory")
    corpus_file_path = _extract_corpus(local_dir=local_dir)

    logger.info("Train tokenizer using extracted corpus")
    tokenizer = _train_tokenizer(
        corpus_file_path=corpus_file_path,
        pad_token_info=pad_token_info,
        unk_token_info=unk_token_info,
        cls_token_info=cls_token_info,
        sep_token_info=sep_token_info,
        mask_token_info=mask_token_info,
        vocab_size=vocab_size,
        local_dir=local_dir,
    )

    logger.info("Save tokenizer into local directory")
    _save_tokenizer(tokenizer, archive_path=f"{local_dir}/{result_dir_name}")


def _extract_corpus(local_dir: pathlib.Path) -> str:
    train_data_path = local_dir.joinpath("train.csv")
    corpus = pd.read_csv(train_data_path)["review"].values
    corpus_file_path = str(local_dir.joinpath("corpus.txt"))
    with open(corpus_file_path, "w") as file:
        file.writelines("\n".join(corpus))
    return corpus_file_path


def _train_tokenizer(
    pad_token_info: Tuple[int, str],
    unk_token_info: Tuple[int, str],
    cls_token_info: Tuple[int, str],
    sep_token_info: Tuple[int, str],
    mask_token_info: Tuple[int, str],
    vocab_size: int,
    local_dir: pathlib.Path,
    corpus_file_path: str,
) -> PreTrainedTokenizerFast:
    """
    Most of the implementation follows process explained in: https://huggingface.co/docs/tokenizers/pipeline
     1. Prepare corpus file(s) in local directory
     2. Define Tokenizer, normalizer, pre-tokenizer, trainer, post-processor in order
     3. Train tokenizer with saved corpus
     4. Save tokenizers.Tokenizer and reload it as PreTrainedTokenizerFast
    :param pad_token_info: pair of (pad_token_id, pad_token)
    :param unk_token_info: pair of (unk_token_id, unk_token)
    :param cls_token_info: pair of (cls_token_id, cls_token)
    :param sep_token_info: pair of (sep_token_id, sep_token)
    :param mask_token_info: pair of (mask_token_id, mask_token)
    :param vocab_size: number of vocabularies to include inside dictionary
    :param local_dir: directory to save tokenizers.Tokenizer to be loaded as PreTrainedTokenizerFast
    :param corpus_file_path: path to corpus file(s)
    :return: PreTrainedTokenizerFast
    """
    tokenizer = Tokenizer(model=BPE())
    tokenizer.normalizer = _define_normalizer()
    tokenizer.pre_tokenizer = _define_pre_tokenizer()
    tokenizer.model.unk_token = unk_token_info[1]
    special_tokens = dict([
        pad_token_info,
        unk_token_info,
        cls_token_info,
        sep_token_info,
        mask_token_info
    ])
    special_tokens = [
        token[1]
        for token in sorted(special_tokens.items(), key=lambda x: x[0])
    ]
    bpe_trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=5,
        show_progress=False,
        special_tokens=special_tokens,
    )
    tokenizer.post_processor = _define_post_processor(cls_token_info, sep_token_info)
    tokenizer.train(files=[corpus_file_path], trainer=bpe_trainer)
    tokenizer_temp_path = str(local_dir.joinpath("tokenizer_temp.json"))
    tokenizer.save(tokenizer_temp_path)
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_temp_path,
        padding_side="right",
        truncation_side="right",
        pad_token=pad_token_info[1],
        unk_token=unk_token_info[1],
        cls_token=cls_token_info[1],
        sep_token=sep_token_info[1],
        mask_token=mask_token_info[1],
    )


def _define_normalizer() -> normalizers.Sequence:
    """
    Other text normalize options can be found at: https://huggingface.co/docs/tokenizers/api/normalizers
    Note that huggingface normalizer cannot handle regular expression groups. That is, there can be certain normalization
    process has to be done prior to be passed into tokenizer(https://github.com/huggingface/tokenizers/issues/996)
    :return: sequence of Normalizers
    """
    return normalizers.Sequence([
        normalizers.Strip(),
        normalizers.Lowercase(),
        normalizers.Replace(pattern="<br />", content=" "),
    ])


def _define_pre_tokenizer() -> pre_tokenizers.Sequence:
    """
    Other text pretokenize options can be found at: https://huggingface.co/docs/tokenizers/api/pre-tokenizers
    :return: sequence of PreTokenizer
    """
    return pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=True)
    ])


def _define_post_processor(
    cls_token_info: Tuple[int, str],
    sep_token_info: Tuple[int, str],
) -> processors.TemplateProcessing:
    """
    https://huggingface.co/docs/tokenizers/api/post-processors#tokenizers.processors.TemplateProcessing
    :param cls_token_info: pair of (cls_token_id, cls_token)
    :param sep_token_info: pair of (sep_token_id, sep_token)
    :return:
    """
    cls_token_id, cls_token = cls_token_info
    sep_token_id, sep_token = sep_token_info
    return processors.TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
        special_tokens=[cls_token_info, sep_token_info]
    )


def _save_tokenizer(tokenizer: PreTrainedTokenizerFast, archive_path: str):
    tokenizer.save_pretrained(archive_path)
