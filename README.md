# hf-tokenizer-pattern

This repository implements common pattern appears when training huggingface tokenizer from raw corpus. This repository uses IMDb review data to compose corpus to train tokenizer from. Explanation on two main components follows:   

* `preprocess.py` : fetch data from source, extract information from it and save train, test data
* `train.py` : extract corpus from train data, train tokenizer and save tokenzier

These codes are implemented and executed on macOS environment with Python 3.9 version.

## Execute

To execute implemented process, change directory into the cloned repository and execute following:  

```shell
python3 --version  # Python 3.9.5
python3 -m venv venv  # generate virtual environment for this project
source venv/bin/activate  # activate generated virtual environment
pip install -r requirements.txt  # install required packages
```

Thanks to [typer](https://typer.tiangolo.com/) package, codes within this repository can be directly executed from console with following command:

```shell
python3 main.py preprocess
```

After the code being executed, preprocessed IMDb data will be saved in `/tmp/hfTokenizer` directory. Then, execute following command to save trained huggingface tokenizer into local directory.

```shell
python3 main.py train
```

Then this saved tokenizer can be loaded and can be used to tokenize list of documents as following:

```python
import pandas as pd
from transformers import AutoTokenizer


reviews = pd.read_csv("/tmp/hfTokenizer/test.csv")
samples = list(reviews.sample(n=3)["review"].values)
tokenizer = AutoTokenizer.from_pretrained("/tmp/hfTokenizer/tokenizer")
tokenizer(
  text=samples,
  max_length=250,
  padding="max_length",
  truncation=True,
  is_split_into_words=False,
)
```