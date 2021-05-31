# HerBERT 
**[HerBERT](https://en.wikipedia.org/wiki/Zbigniew_Herbert)** is a series of BERT-based language models trained for Polish language understanding.

All three HerBERT models are summarized below:

| Model | Tokenizer | Vocab Size | Batch Size | Train Steps | KLEJ Score |
| :---- | --------: | ---------: | ---------: | ----------: | ---------: | 
| `herbert-klej-cased-v1` | BPE | 50K | 570 | 180k | 80.5 |
| `herbert-base-cased` | BPE-Dropout | 50K | 2560 | 50k | 86.3 |
| `herbert-large-cased` | BPE-Dropout | 50K | 2560 | 60k | 88.4 |

Full KLEJ Benchmark leaderboard is available [here](https://klejbenchmark.com/leaderboard). 

For more details about model architecture, training process, used corpora and evaluation please refer to: 
- [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111/)
- [HerBERT: Efficiently Pretrained Transformer-based Language Model for Polish](https://www.aclweb.org/anthology/2021.bsnlp-1.1/).


## Usage
Example of how to load the model:
```python
from transformers import AutoTokenizer, AutoModel

model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1", 
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased", 
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased", 
        "model": "allegro/herbert-large-cased",
    },
}

tokenizer = AutoTokenizer.from_pretrained(model_names["allegro/herbert-base-cased"]["tokenizer"])
model = AutoModel.from_pretrained(model_names["allegro/herbert-base-cased"]["model"])
```

And how to use the model:
```python
output = model(
    **tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
    )
)
```

## License
CC BY 4.0

## Citation
If you use this model, please cite the following papers:

The `herbert-klej-cased-v1` version of the model:
```
@inproceedings{rybak-etal-2020-klej,
    title = "{KLEJ}: Comprehensive Benchmark for Polish Language Understanding",
    author = "Rybak, Piotr and Mroczkowski, Robert and Tracz, Janusz and Gawlik, Ireneusz",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.111",
    pages = "1191--1201",
}
```

The `herbert-base-cased` or `herbert-large-cased` version of the model:
```
@inproceedings{mroczkowski-etal-2021-herbert,
    title = "{H}er{BERT}: Efficiently Pretrained Transformer-based Language Model for {P}olish",
    author = "Mroczkowski, Robert  and
      Rybak, Piotr  and
      Wr{\'o}blewska, Alina  and
      Gawlik, Ireneusz",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.1",
    pages = "1--10",
}
```

## Contact
You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
