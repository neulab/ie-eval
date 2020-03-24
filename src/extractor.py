"""pytorch port of the IE system from https://github.com/harvardnlp/data2text"""
import argparse
import logging
import os
import sys
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import utils
from dataset import IEDataset
from models import BLSTMModel, ConvModel

LOGGER = logging.getLogger(__name__)


def create_model_and_optimizer(args, dataset, params=None):
    """creates model from the dataset and load/initializes its parameters if necessary."""

    vocab_sizes = [len(dataset.vocab[k]["w2i"]) for k in ["word", "ent", "num"]]
    embed_sizes = [
        args.word_embed_size,
        args.entdist_embed_size,
        args.numdist_embed_size,
    ]
    pads = [dataset.vocab[k]["w2i"]["PAD"] for k in ["word", "ent", "num"]]

    if args.model == "LSTM":
        model = BLSTMModel(
            vocab_sizes,
            embed_sizes,
            sum(embed_sizes),
            args.blstm_fc_hidden_dim,
            len(dataset.vocab["label"]["w2i"]),
            pads,
            args.dropout,
        )
    elif args.model == "CNN":
        model = ConvModel(
            vocab_sizes,
            embed_sizes,
            sum(embed_sizes),
            args.conv_fc_hidden_dim,
            len(dataset.vocab["label"]["w2i"]),
            pads,
            args.num_filters,
            args.dropout,
        )

    if params is not None:
        model.load_state_dict(params)

    else:
        # initialize all the model weights
        for p in model.parameters():
            torch.nn.init.uniform_(p, -args.uniform_init, args.uniform_init)

        # Make sure that pad vectors are zero
        model.embed.pad_init()

    if args.cuda:
        model = model.to("cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr)

    return model, optimizer


def train_model(args, dataset, model, optimizer):

    best_checkpoint = None
    train_batches = dataset.get_batches("train", shuffle=True)

    for eph in utils.progress(range(1, args.num_epochs + 1), desc="Epoch"):
        model.train()

        total_loss = 0
        step = 0
        correct, total, nonnolabel = 0, 0, 0
        pred_stats = []
        for batch in utils.progress(train_batches, desc="Batch"):
            optimizer.zero_grad()
            if args.cuda:
                batch = batch.to("cuda", persistent=True)
            step += 1

            loss, preds = model.get_loss(batch, predict=True)
            total_loss += loss.item()
            loss.backward()

            if args.model == "LSTM":
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            # Exclude NONE labels
            nonnolabel += sum([1 for ls in batch.labels if 0 not in ls])
            total += sum([1 for pred in preds if pred.item() != 0])
            correct += sum(
                [
                    1
                    for pred, ls in zip(preds, batch.labels)
                    if pred.item() != 0 and pred.item() in ls
                ]
            )
            pred_stats += [p.item() for p in preds]

            if (step + 1) % args.log_interval == 0:
                LOGGER.info(f"Eph {eph:2d} Train loss at step {step+1}: {loss.item()}")
                LOGGER.info(f"{Counter(pred_stats).most_common(5)}")
                pred_stats = []

        acc = correct / total if total > 0 else 0
        rec = correct / nonnolabel if nonnolabel > 0 else 0

        LOGGER.info(
            f"Eph {eph:2d} Train loss : {total_loss / dataset.nexamples['train']:4.5f}"
            f" Acc: {acc*100:2.2f}"
            f" Rec: {rec*100:2.2f}"
        )

        model.eval()
        val_batches = dataset.get_batches("valid", shuffle=False)
        val_loss = 0
        correct, total, nonnolabel = 0, 0, 0
        pred_stats = []
        with torch.no_grad():
            for batch in utils.progress(val_batches, desc="Eval"):
                if args.cuda:
                    batch = batch.to("cuda")
                loss, preds = model.get_loss(batch, predict=True)
                val_loss += loss.item()

                # Exclude NONE labels
                nonnolabel += sum([1 for ls in batch.labels if 0 not in ls])
                total += sum([1 for pred in preds if pred.item() != 0])
                correct += sum(
                    [
                        1
                        for pred, ls in zip(preds, batch.labels)
                        if pred.item() != 0 and pred.item() in ls
                    ]
                )
                pred_stats += [p.item() for p in preds]

        acc = correct / total if total > 0 else 0
        rec = correct / nonnolabel if nonnolabel > 0 else 0
        LOGGER.info(f"{Counter(pred_stats)}")

        # Original code looks at the multilabel acc to determine learning rate scheduling.
        if best_checkpoint is None or acc > best_checkpoint.val_loss:
            best_checkpoint = utils.Checkpoint(
                eph,
                acc,
                utils.cpu_state_dict(model.state_dict()),
                utils.cpu_state_dict(optimizer.state_dict()),
            )

            if args.save:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(args.exp, f"model{eph}.pt"))
                LOGGER.info(f"      | Best model achieved at eph {eph}, saved.")

        # Otherwise revert to the previous best checkpoint
        elif args.optimizer_strategy == "reset":
            # Reset both
            new_lr = optimizer.param_groups[0]["lr"] * args.lr_scaler
            model.load_state_dict(best_checkpoint.model_state)
            optimizer.load_state_dict(best_checkpoint.optim_state)
            optimizer.param_groups[0]["lr"] = new_lr
            LOGGER.info(
                f"Loaded best model at epoch {best_checkpoint.epoch}. ==> LR: {new_lr}\n"
            )

        elif args.optimizer_strategy == "continue":
            # Reset both
            new_lr = optimizer.param_groups[0]["lr"] * args.lr_scaler
            optimizer.param_groups[0]["lr"] = new_lr
            LOGGER.info(f"New LR: {new_lr} at epoch {eph}\n")

        LOGGER.info(
            f"Eph {eph:2d} Valid loss : {val_loss / dataset.nexamples['valid']:4.5f}"
            f" Acc: {acc*100:2.2f}"
            f" Rec: {rec*100:2.2f}"
        )


def test_model(args, dataset, model, ensemble=False):
    """evaluate on test set."""

    test_batches = dataset.get_batches("test", shuffle=False)

    if not ensemble:
        model.eval()
    else:
        for m in model:
            m.eval()

    total_loss = 0
    total, correct, nonnolabel = 0, 0, 0
    for batch in utils.progress(test_batches, desc="Batch"):
        if args.cuda:
            batch = batch.to("cuda", persistent=True)

        if ensemble:
            losses = []
            cumulative_probs = torch.zeros(
                len(batch), len(dataset.vocab["label"]["i2w"])
            )
            with torch.no_grad():
                for m in model:
                    loss, probs = m.get_loss(batch, return_prob=True)
                    losses.append(loss.item())
                    cumulative_probs += probs.cpu()  # following Wiseman+

            total_loss += np.mean(losses)
            preds = torch.max(probs, dim=1)[1]

        else:
            loss, preds = model.get_loss(batch, predict=True)
            total_loss += loss.item()

        # Exclude NONE labels
        nonnolabel += sum([1 for ls in batch.labels if 0 not in ls])
        total += sum([1 for pred in preds if pred.item() != 0])
        correct += sum(
            [
                1
                for pred, ls in zip(preds, batch.labels)
                if pred.item() != 0 and pred.item() in ls
            ]
        )

    acc = correct / total
    rec = correct / nonnolabel

    LOGGER.info(
        f"Test loss : {total_loss / dataset.nexamples['test']:4.5f}"
        f" Acc: {acc*100:2.2f}"
        f" Rec: {rec*100:2.2f}"
    )


def get_args(sent, ent_dists, num_dists, vocab):
    entvocab = vocab["ent"]["i2w"]
    numvocab = vocab["num"]["i2w"]
    wrdvocab = vocab["word"]["i2w"]
    ent, num = [], []
    for i, (eidx, nidx) in enumerate(zip(ent_dists, num_dists)):
        if entvocab[eidx] == 0:
            ent.append(sent[i])

        if numvocab[nidx] == 0:
            num.append(sent[i])

    return " ".join([wrdvocab[i] for i in ent]), " ".join([wrdvocab[i] for i in num])


def evaluate_outputs(args, dataset, model, ensemble=False):
    """evaluate on test set. This produces system IE output, which will be used to
    evaluate IE-based scores.
    """
    if hasattr(dataset, "output_data"):
        eval_batches = dataset.get_batches("output", shuffle=False)
    else:
        eval_batches = dataset.get_batches(
            "test" if args.test else "valid", shuffle=False
        )

    output_file_path = Path(args.exp) / "system_ie_output"
    output_file = open(output_file_path, "w")

    box_restarts = {}
    for idx in dataset.box_restarts:
        box_restarts[idx] = True

    if not ensemble:
        model.eval()
    else:
        for m in model:
            m.eval()

    total_loss = 0
    total, correct = 0, 0
    cand_num = 0
    seen = {}
    ndupcorrects, nduptotal = 0, 0
    for batch in utils.progress(eval_batches, desc="Evaluating"):
        if args.cuda:
            batch = batch.to("cuda", persistent=True)

        if ensemble:
            losses = []
            cumulative_probs = torch.zeros(
                len(batch), len(dataset.vocab["label"]["i2w"])
            )
            with torch.no_grad():
                for m in model:
                    loss, probs = m.get_loss(batch, return_prob=True)
                    losses.append(loss.item())
                    cumulative_probs += probs.cpu()  # following Wiseman+

            total_loss += np.mean(losses)
            preds = torch.max(probs, dim=1)[1]

        else:
            loss, preds = model.get_loss(batch, predict=True)
            total_loss += loss.item()

        for idx in range(batch.sentences.size(0)):
            cand_num += 1
            if box_restarts and box_restarts.get(cand_num, False):
                print(file=output_file)
                seen = {}

            if preds[idx].item() != 0:
                ent_arg, num_arg = get_args(
                    batch.sentences[idx],
                    batch.features["entdists"][idx],
                    batch.features["numdists"][idx],
                    dataset.vocab,
                )
                pred_key_list = [
                    ent_arg,
                    num_arg,
                    dataset.vocab["label"]["i2w"][preds[idx].item()],
                ]
                pred_key = " ".join(pred_key_list)
                print(" | ".join(pred_key_list), file=output_file)

                if preds[idx].item() in batch.labels[idx] and pred_key in seen:
                    ndupcorrects += 1
                if pred_key in seen:
                    nduptotal += 1

                seen[pred_key] = True

        # Exclude NONE labels
        total += sum([1 for pred in preds if pred.item() != 0])
        correct += sum(
            [
                1
                for pred, ls in zip(preds, batch.labels)
                if pred.item() != 0 and pred.item() in ls
            ]
        )

    acc = correct / total
    ndup_prec = (correct - ndupcorrects) / (total - nduptotal)

    result_str = (
        f" RG Prec: {acc*100:2.2f}\n"
        f"  ├-- Nodup Prec: {ndup_prec*100:2.2f}\n"
        f"  ├-- Total Correct: {correct:4d}\n"
        f"  └-- Nodup Coorect: {correct-ndupcorrects:4d}\n"
    )

    LOGGER.info(result_str)
    print()
    print(result_str)  # to stdout
    output_file.close()


def load_pretrained_models(args, dataset):

    models = []
    if args.pretrained_dir is not None:
        for p in args.pretrained_dir.iterdir():
            if "cnn" in p.name:
                args.model = "CNN"
            else:
                args.model = "LSTM"
            path = utils.get_best_model(p)[0]
            checkpoint = torch.load(path, map_location="cpu")
            lstm_model, _ = create_model_and_optimizer(
                args, dataset, params=checkpoint["model"]
            )
            models.append(lstm_model)
            LOGGER.info(f"Loaded: {path}")

    else:
        if args.pretrained_lstm is not None:
            assert len(args.pretrained_lstm) > 0
            args.model = "LSTM"

            for path in args.pretrained_lstm:
                # model is put to gpu when constructed
                path = utils.get_best_model(path)[0]
                checkpoint = torch.load(path, map_location="cpu")
                lstm_model, _ = create_model_and_optimizer(
                    args, dataset, params=checkpoint["model"]
                )
                models.append(lstm_model)
                LOGGER.info("Loaded: {path}")

        if args.pretrained_cnn is not None:
            assert len(args.pretrained_cnn) > 0
            args.model = "CNN"

            for path in args.pretrained_cnn:
                path = utils.get_best_model(path)[0]
                checkpoint = torch.load(path, map_location="cpu")
                cnn_model, _ = create_model_and_optimizer(
                    args, dataset, params=checkpoint["model"]
                )
                models.append(cnn_model)
                LOGGER.info("Loaded: {path}")

    return models


def evaluate(args):

    dataset = IEDataset(args)
    models = load_pretrained_models(args, dataset)

    LOGGER.info(f"{len(models)} models are loaded.")

    # test_model(args, dataset, models, ensemble=True)
    evaluate_outputs(args, dataset, models, ensemble=True)


def main(args):

    # Seed RNGs for reproducibility
    if args.seed > 0:
        print(f"Random seed set to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # Configure logging
    if args.save:
        logfile = utils.create_exp_dir(args.exp, args.script, overwrite=args.overwrite)
    else:
        logfile = None

    logging.basicConfig(
        datefmt="%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.getLevelName(args.logging_level),
        filename=logfile,
    )

    n_gpus = torch.cuda.device_count()
    LOGGER.debug(
        "Running the model on " + (f"CUDA with {n_gpus} GPU(s)" if args.cuda else "CPU")
    )
    for device in range(n_gpus):
        props = torch.cuda.get_device_properties(device)
        LOGGER.debug(
            f"GPU ({device}) name: {props.name}, CUDA version {props.major}.{props.minor}, "
            f"available memory: {props.total_memory / 1024 / 1024:.2f}MB."
        )

    if args.eval or args.output is not None:
        evaluate(args)

    else:
        # Create dataset
        dataset = IEDataset(args)

        # Create model
        model, optimizer = create_model_and_optimizer(args, dataset)

        # Print model parameter info
        n_params = sum(p.nelement() for p in model.parameters())
        LOGGER.info(f"Model parameters: {n_params}")
        LOGGER.info(f"Model structure:\n{str(model)}")

        if args.pretrained:
            path, _ = utils.get_best_model(args.pretrained)
            states = torch.load(path, map_location="cuda")
            model.load_state_dict(states["model"])

        try:
            train_model(args, dataset, model, optimizer)
        except KeyboardInterrupt:
            LOGGER.info("Training halted.")

        path, _ = utils.get_best_model(args.exp)
        states = torch.load(path, map_location="cuda")
        model.load_state_dict(states["model"])

        test_model(args, dataset, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("IE model.")
    # training
    parser.add_argument("--data", type=Path, default=None, help="Path to the dataset (for training).")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--initial-lr", type=float, default=0.7)
    parser.add_argument("--lr-scaler", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--optimizer-strategy", type=str, default="reset")
    parser.add_argument("--uniform-init", type=float, default=0.1)
    parser.add_argument("--model", default="LSTM", choices=["LSTM", "CNN"])
    parser.add_argument("--logging-level", default="DEBUG")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--script", type=str, default=None)
    parser.add_argument("--pretrained", type=Path, default=None)

    # eval only
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument(
        "--pretrained-lstm",
        type=Path,
        nargs="+",
        default=None,
        help="Use this flag when you need to individually load LSTM models.",
    )
    parser.add_argument(
        "--pretrained-cnn",
        type=Path,
        nargs="+",
        default=None,
        help="Use this flag when you need to individually load CNN models.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)

    # common
    parser.add_argument("--word-embed-size", type=int, default=200)
    parser.add_argument("--entdist-embed-size", type=int, default=100)
    parser.add_argument("--numdist-embed-size", type=int, default=100)

    # conv
    parser.add_argument("--num-filters", type=int, default=200)
    parser.add_argument("--conv-fc-hidden-dim", type=int, default=500)

    # lstm
    parser.add_argument("--blstm-fc-hidden-dim", type=int, default=700)

    args = parser.parse_args()

    if args.data is None and args.output is None:
        print("Specify either training (--data) or inference data (--output)")
        sys.exit(1)
    elif args.output is not None and args.vocab is None:
        print("If inference only, specify the path to the vocabulary.")
        sys.exit(1)

    main(args)
