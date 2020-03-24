"""A script to construct a set of keywords in two languages."""
from typing import NamedTuple, Set
from pathlib import Path

DE_NUMBER_DATA = Path("data/numbers/roto_german_numbers.txt")

# English keywords
class Keyword(NamedTuple):
    prons: Set[str]
    singular: Set[str]
    plural: Set[str]
    number: Set[str]
    ignores: Set[str]


def prepare_de_numbers():
    with DE_NUMBER_DATA.open("r") as f:
        nums = f.read().strip().split("\n")
    nums = [l.split("\t") for l in nums]
    # Lowering and concatenating all the whitespace, based on Torsten's advice.
    text2num_de = {l[2].lower().replace(" ", ""): l[0] for l in nums}
    return text2num_de


def get_keywords(lang):
    res = keyword_dict[lang]
    sn, pl, nm, ig = res["singular"], res["plural"], res["number"], res["ignores"]
    return Keyword(sn | pl, sn, pl, nm, ig)


text2num = prepare_de_numbers()

# constructing keywords dirctionary
keyword_dict = {}
keyword_dict["en"] = dict(
    singular=set(["he", "He", "him", "Him", "his", "His"]),
    plural=set(["they", "They", "them", "Them", "their", "Their"]),
    number=set(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
        ]
    ),
    ignores=set(
        [
            "three 's",
            "three - pointers",
            "three - pointer",
            "three pointer",
            "three point",
            "three - point",
            "three - pt",
            "three pt",
        ]
    ),
)

keyword_dict["de"] = dict(
    singular=set(["er", "Er", "ihm", "Ihm", "seine", "Seine"]),
    plural=set(["Sie", "Sie", "Sie", "Sie", "ihr", "Ihr"]),
    number=set(list(text2num.keys())),
    ignores=set(
        [
            "three 's",
            "three - pointers",
            "three - pointer",
            "three pointer",
            "three point",
            "three - point",
            "three - pt",
            "three pt",
            "Dreier",
            "Dreiern",
            "Drei -",
        ]
    ),
)


