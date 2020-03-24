# IE-eval

This repository contains evaluation tools and data for the DGT task at Workshop on Neural Generation and Translation.

## Preparation

* Download the [Rotowire English-German dataset](https://github.com/neulab/DGT/releases/download/v1.5/rotowire_english-german_1.5.tar.bz2).

* Clone this repository.

* Download the model weights used to run content accuracy evaluation.

```sh
$ wget https://github.com/neulab/ie-eval/releases/download/v1.0/models.tar.bz2
$ mv models.tar.bz2 /path/to/this/repo/models/ && tar vxjf models.tar.bz2
```
* Install python requirements

```
pip install -r requirements.txt
```

## Evaluate your output

We re-implement the evaluation tool from [harvardnlp (bug-fixed by Ratish)](https://github.com/ratishsp/data2text-1) in PyTorch.

1. Open the script `scripts/ie_eval.sh` and edit the environment variables to suitable values:

* `SYS_OUTPUT`: The system output file, formatted according to the evaluation guideline.
* `EXP`: Any name that identifies the specific run. Extraction results are going to be stored in the directory with this name.
* `LANG`: Either of `en` or `de`.
* `SPLIT`: Either of `valid` or `test`.
* `ORIGINAL_PATH`: Path to our RotoWire English-German dataset, specifically the `original` version.

For example, the following setting will evaluate content accuracy against the gold extraction results on English instances in the validation set.
```sh
SYS_OUTPUT=/path/to/system.json     # path to the formatted json file containing the system output.
EXP=trial_1                         # Experiment name, whichever identifiable.
LANG=en                             # Language, either of `en` or `de`
SPLIT=valid                         # valid or test
ORIGINAL_PATH=/path/to/original-1.5 # Path to the `original` portion of the provided data.
```

2. Run the script

```sh
$ ./ie_eval.sh
```

## Train your own IE model

We release [pretrained weights](https://github.com/neulab/ie-eval/releases/download/v1.0/models.tar.bz2)
for 6 models (3 CNN models, 3 LSTM models) for each language, which we use as an ensemble for evaluation.
However, the following commands can be used to train a new model with different configurations.

```sh
# Set accordingly
SEED=1234
LANG=en

# Prepare IE data
python data_utils.py \
    --mode make_ie_data \
    --input-path /path/to/original-1.5 \
    --output-fi $LANG.pkl \
    --min-freq 0

# If training a LSTM model
python src/extractor.py \
    --data $LANG.pkl \
    --exp $LANG-lstm-${seed} \
    --save \
    --overwrite \
    --cuda \
    --log-interval 10 \
    --initial-lr 1 \
    --seed $LANG

# If training a CNN model
python src/extractor.py \
    --data $LANG.pkl \
    --exp $LANG-cnn-${seed} \
    --model CNN
    --save \
    --overwrite \
    --cuda \
    --log-interval 10 \
    --initial-lr 0.7 \
    --seed $LANG
```

---

For more information, please take a look at the [shared task overview](https://www.aclweb.org/anthology/D19-5601):

```bibtex
@inproceedings{hayashi2019findings,
    title = "Findings of the Third Workshop on Neural Generation and Translation",
    author = "Hayashi, Hiroaki  and
      Oda, Yusuke  and
      Birch, Alexandra  and
      Konstas, Ioannis  and
      Finch, Andrew  and
      Luong, Minh-Thang  and
      Neubig, Graham  and
      Sudoh, Katsuhito",
    booktitle = "Proceedings of the 3rd Workshop on Neural Generation and Translation",
    month = nov,
    year = "2019",
    address = "Hong Kong",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5601",
    doi = "10.18653/v1/D19-5601",
    pages = "1--14",
}
```

