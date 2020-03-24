#!/usr/bin/env bash

SYS_OUTPUT=                         # path to the formatted json file containing the system output.
EXP=                                # Experiment name, whichever identifiable.
LANG=en                             # Language, either of `en` or `de`
SPLIT=valid                         # valid or test
ORIGINAL_PATH=/path/to/original-1.5 # Path to the `original` portion of the provided data.
PRETRAINED_PATH=/path/to/models/    # Path to the pretrained models

# Make a working directory to store intermediate outputs
[[ ! -d ./work ]] && mkdir work

[[ $SPLIT == test ]] && TEST_FLAG="--test"


# Preprocess the data so that the IE model can evaluate
python src/data_utils.py \
    --mode prep_gen_data \
    --gen-fi $SYS_OUTPUT \
    --dict-pfx data/dicts/$LANG \
    --output-fi work/$EXP-output.pkl \
    --input-path $ORIGINAL_PATH \
    --lang $LANG \
    $TEST_FLAG


# Run the IE model, get RG score.
python src/extractor.py \
    --output work/$EXP-output.pkl \
    --vocab data/vocab/vocab.$LANG.pt \
    --exp $EXP \
    --save \
    --cuda \
    --eval \
    --pretrained-dir models/$LANG


# CS and CO scores.
python src/non_rg_metrics.py \
    data/gold_ie/gold_ie.$SPLIT.$LANG \
    $EXP/system_ie_output


# delete intermediate data
rm -rf work
