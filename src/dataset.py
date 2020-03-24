import copy
import pickle
import random
from collections import namedtuple
from typing import List

from tqdm import tqdm

import torch

Example = namedtuple("Example", ["sentence", "length", "entdist", "numdist", "label"])


class Batch:
    def __init__(self, sentences, lengths, labels, **kwargs):
        self.sentences = sentences
        self.lengths = lengths
        self.labels = labels

        self.features = {}
        for k, v in kwargs.items():
            self.features[k] = v

        self.sort_lengths, self.sort_idx = torch.sort(
            self.lengths, dim=0, descending=True
        )
        _, self.org_idx = torch.sort(self.sort_idx)
        self.sort_lengths = self.sort_lengths.tolist()

    def to(self, device, persistent=False):
        if not persistent:
            obj = copy.copy(self)
        else:
            obj = self
        obj.sentences = self.sentences.to(device)
        obj.lengths = self.lengths.to(device)
        obj.sentences = self.sentences.to(device)
        obj.features = {name: seq.to(device) for name, seq in self.features.items()}
        obj.labels = [l.to(device) for l in self.labels]
        obj.org_idx = self.org_idx.to(device)

        return obj

    def __len__(self):
        return self.sentences.size(0)


class IEDataset:
    def __init__(self, args):

        # set attribute "data" | "output_data", "vocab"
        self.load_data(args)

        if hasattr(self, "data"):
            self.examples = self.prepare_data(data)
            self.batches = self.make_batches(args.batch_size, self.examples)
            self.nexamples = {k: len(v) for k, v in self.examples.items()}

        # Generation model output is also prepared and stored as "output"-key.
        if hasattr(self, "output_data"):
            self.output_data, self.box_restarts = self.prepare_output_data(self.output_data)
            if hasattr(self, "batches"):
                self.batches.update(self.make_batches(args.batch_size, self.output_data))
            else:
                self.batches = self.make_batches(args.batch_size, self.output_data)

    def load_data(self, args):
        """load the base data and their vocab according to the file name match."""

        if args.data is not None:
            path = args.data

            with path.open("rb") as f:
                data = pickle.load(f)

            vocab = {k: {} for k in ["word", "label", "ent", "num"]}
            # Word vocabulary is dumped in ".dict" file, line by line.
            with path.with_suffix(".dict").open("r") as f:
                w2i = {
                    l.split()[0]: int(l.split()[1]) for l in f.read().strip().split("\n")
                }
                i2w = []
                for k, _ in sorted(w2i.items(), key=lambda x: x[1]):
                    i2w.append(k)
                vocab["word"]["i2w"] = i2w
                vocab["word"]["w2i"] = w2i

            # Label vocabulary is dumped in ".labels" file, line by line.
            with path.with_suffix(".labels").open("r") as f:
                l2i = {
                    l.split()[0]: int(l.split()[1]) for l in f.read().strip().split("\n")
                }
                i2l = []
                for k, _ in sorted(l2i.items(), key=lambda x: x[1]):
                    i2l.append(k)
                vocab["label"]["i2w"] = i2l
                vocab["label"]["w2i"] = l2i

            # Make vocab for ent/numdists
            vocab["ent"]["i2w"] = sorted(
                set(d for sent in data["trentdists"] + data["valentdists"] for d in sent)
            )
            vocab["ent"]["w2i"] = {}
            for idx, ed in enumerate(vocab["ent"]["i2w"]):
                vocab["ent"]["w2i"][ed] = idx
            vocab["ent"]["w2i"]["PAD"] = len(vocab["ent"]["i2w"])
            vocab["ent"]["i2w"].append("PAD")

            vocab["num"]["i2w"] = sorted(
                set(d for sent in data["trnumdists"] + data["valnumdists"] for d in sent)
            )
            vocab["num"]["w2i"] = {}
            for idx, nd in enumerate(vocab["num"]["i2w"]):
                vocab["num"]["w2i"][nd] = idx
            vocab["num"]["w2i"]["PAD"] = len(vocab["num"]["i2w"])
            vocab["num"]["i2w"].append("PAD")

            torch.save(vocab, args.data.parent / "vocab.pt")

            setattr(self, "data", data)

        if args.output is not None and args.output.exists():
            with args.output.open("rb") as f:
                output_data = pickle.load(f)

            vocab = torch.load(args.vocab)
            setattr(self, "output_data", output_data)

        setattr(self, "vocab", vocab)

    def prepare_data(self, data):
        splits = ["train", "valid", "test"]
        examples = {k: [] for k in splits}

        # train
        for features in zip(
            data["trsents"],
            data["trlens"],
            data["trentdists"],
            data["trnumdists"],
            data["trlabels"],
        ):
            examples["train"].append(Example(*features))

        for features in zip(
            data["valsents"],
            data["vallens"],
            data["valentdists"],
            data["valnumdists"],
            data["vallabels"],
        ):
            examples["valid"].append(Example(*features))

        for features in zip(
            data["testsents"],
            data["testlens"],
            data["testentdists"],
            data["testnumdists"],
            data["testlabels"],
        ):
            examples["test"].append(Example(*features))

        return examples

    def prepare_output_data(self, data):
        examples = {"output": []}
        boxrestarts = []

        # train
        for features in zip(
            data["valsents"],
            data["vallens"],
            data["valentdists"],
            data["valnumdists"],
            data["vallabels"],
        ):
            examples["output"].append(Example(*features))

        return examples, data["boxrestartidxs"]

    def make_batches(self, batch_size, examples, extras=None):
        """split examples into batch_size and construct batches. Additionally `extras` are segmented
        in the same way and returned.
        """

        split_extras = {}
        batches = {}
        for split, exs in examples.items():
            # exs = sorted(exs, key=lambda x: x.length, reverse=True)

            if extras is not None:
                extra = extras[split]
                split_extras[split] = []

            split_batches = []
            num_batches = len(exs) // batch_size
            for i in tqdm(
                range(num_batches + 1), ncols=80, ascii=True, desc="Creating batches"
            ):
                ex_batch = exs[i * batch_size : (i + 1) * batch_size]
                split_batches.append(self.make_one_batch(ex_batch))

                if extras:
                    split_extras[split].append(
                        extra[i * batch_size : (i + 1) * batch_size]
                    )

            batches[split] = split_batches

        if extras is not None:
            return batches, split_extras

        return batches

    def make_one_batch(self, examples: List[Example]):
        max_length = max([ex.length for ex in examples])
        sentences = torch.LongTensor([ex.sentence[:max_length] for ex in examples])
        lengths = torch.LongTensor([ex.length for ex in examples])

        entdists = [
            [
                self.vocab["ent"]["w2i"].get(ed, self.vocab["ent"]["w2i"]["PAD"])
                if i < ex.length
                else self.vocab["ent"]["w2i"]["PAD"]
                for i, ed in enumerate(ex.entdist[:max_length])
            ]
            for ex in examples
        ]
        numdists = [
            [
                self.vocab["num"]["w2i"].get(nd, self.vocab["num"]["w2i"]["PAD"])
                if i < ex.length
                else self.vocab["num"]["w2i"]["PAD"]
                for i, nd in enumerate(ex.numdist[:max_length])
            ]
            for ex in examples
        ]
        labels = [torch.LongTensor(ex.label[: ex.label[-1]]) for ex in examples]
        return Batch(
            sentences,
            lengths,
            labels,
            entdists=torch.LongTensor(entdists),
            numdists=torch.LongTensor(numdists),
        )

    def get_batches(self, split, shuffle=True):
        if shuffle:
            random.shuffle(self.batches[split])
        return self.batches[split]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
