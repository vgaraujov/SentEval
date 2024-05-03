from collections import Counter
from pathlib import Path
from typing import Iterable, Union
import numpy.random as npr
from string import ascii_lowercase
import itertools
import wonderwords as ww
from tqdm.auto import tqdm
import numpy as np

RNG = npr.default_rng(0)

PATH_EXAMPLES = Path(__file__).resolve().parent
PATH_ROOT = PATH_EXAMPLES.parent
PATH_DATA = PATH_ROOT / "data"


def generateSentences(loop: bool):
    while True:
        with open(PATH_DATA / "probing" / "sentence_length.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                split, _, sentence = line.strip().split("\t")
                yield sentence

        if loop:
            print("End of sentences reached. Recommencing...")
        else:
            break


def generateWords(lengths: Union[Iterable[int],int]):
    generator = ww.RandomWord()

    if isinstance(lengths, int):
        lengths = [lengths]
    length_iterator = itertools.cycle(lengths)

    while True:
        yield " ".join(generator.random_words(amount=next(length_iterator)))


def generateCharacters(lengths: Union[Iterable[int], int]):
    if isinstance(lengths, int):
        lengths = [lengths]
    length_iterator = itertools.cycle(lengths)

    characters = ascii_lowercase + " "
    while True:
        length = next(length_iterator)
        yield "".join([characters[i] for i in RNG.choice(len(characters), size=length)])


def dataset_countCharacter(generator: Iterable[str], examples_per_sentence: int):
    def normalised_softmax(x: np.ndarray, tau: float):
        # Normalise
        x = x / np.sum(x)

        # Softmax (first apply temperature, then apply invariant shift)
        x = x / tau
        x = np.exp(x - min(x))
        return x / np.sum(x)

    for string in generator:
        counts = Counter(string)
        for k in list(counts.keys()):
            if not k.isalpha():
                counts.pop(k)

        if not counts:  # Sentence consists entirely of numbers.
            continue

        keys          = list(counts.keys())
        probabilities = normalised_softmax(np.array(list(counts.values())), tau=0.05)  # Lower temperature (towards 0) makes the sample less uniform. Less uniform likely means fewer 1-count samples.
        for index in RNG.choice(len(keys), p=probabilities, size=min(examples_per_sentence, len(keys)), replace=False):
            yield counts[keys[index]], string, keys[index]


def dataset_unicodeInsert(generator: Iterable[str]):
    # 50% should be with and 50% without
    character_ords = list(range(1568,1610+1)) + list(range(1654,1725+1))  # Arabic characters that aren't just diacritics.
    for string in generator:
        if RNG.random() < 0.5:
            yield 0, string
        else:
            index = RNG.choice(len(string))
            character_ord = RNG.choice(character_ords)
            yield 1, string[:index] + chr(character_ord) + string[index:]


def dataset_maxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string)
        if " " in counts:
            counts.pop(" ")

        first,second = counts.most_common(n=2)
        if first[1] != second[1]:  # Unique max
            yield first[1], string


def dataset_argmaxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string)
        if " " in counts:
            counts.pop(" ")

        first,second = counts.most_common(n=2)
        if first[1] != second[1]:  # Unique max
            yield first[0], string


def take(n: int, generator: Iterable):
    i = 0
    for thing in generator:
        if i >= n:
            break
        yield thing
        i += 1

    if i != n:  # Not necessarily after a break. Generator could've also just yielded exactly n items.
        print(f"Tried to take {n} examples but only {i} were generated.")


def addHoldoutPrefix(total_size: int, generator: Iterable[tuple]):
    SPLITS = ["tr", "va", "te"]
    for i,tup in tqdm(enumerate(take(total_size, generator)), total=total_size):
        if i < 0.8*total_size:
            yield (SPLITS[0],) + tup
        elif i < 0.9*total_size:
            yield (SPLITS[1],) + tup
        else:
            yield (SPLITS[2],) + tup


def tuplesToFile(path: Path, generator: Iterable[tuple], sep: str="\t", lastsep: str="\t"):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as handle:
        for tup in generator:
            handle.write(sep.join(map(str,tup[:-2])) + sep + lastsep.join(map(str,tup[-2:])) + "\n")


def histogramOfTsv(path: Path, column: int):
    """
    Makes a histogram of the ith value on each line of a TSV file.
    """
    from fiject import MultiHistogram

    h = MultiHistogram(f"counts-{path.stem}")
    with open(path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle):
            parts = line.strip().split("\t")
            h.add(f"column {column}", float(parts[column]))

    h.commitWithArgs_histplot(
        MultiHistogram.ArgsGlobal(binwidth=1, relative_counts=True, center_ticks=True, x_label="Label", y_label="Fraction of examples")
    )


if __name__ == "__main__":
    # for s in generateCharacters(n=10, lengths=5):
    #     print(s)
    # for example in dataset_countCharacter(generateCharacters(n=10, lengths=20), examples_per_sentence=5):
    # for example in dataset_countCharacter(generateWords(lengths=10), examples_per_sentence=5):
    # for example in dataset_countCharacter(generateSentences(), examples_per_sentence=3):
    #     print("\t".join(map(str,example)))
    # for example in take(10, dataset_argmaxCharacter(generateCharacters(lengths=20))):
    # for example in dataset_unicodeInsert(generateSentences()):
    #     print("\t".join(map(str,example)))
    # for example in addHoldoutPrefix(20, dataset_unicodeInsert(generateSentences())):
    #     print(example)

    out_path = PATH_DATA / "probing" / "Visual" / "count_character_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateSentences(loop=False), examples_per_sentence=3)), lastsep="|")
    histogramOfTsv(out_path, column=1)

    out_path = PATH_DATA / "probing" / "Visual" / "count_character_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))), examples_per_sentence=3)), lastsep="|")

    out_path = PATH_DATA / "probing" / "Visual" / "odd_character_out_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateSentences(loop=False))))

    out_path = PATH_DATA / "probing" / "Visual" / "odd_character_out_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))

    out_path = PATH_DATA / "probing" / "Visual" / "max_count_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(80_000, dataset_maxCharacter(generateSentences(loop=False))))

    out_path = PATH_DATA / "probing" / "Visual" / "max_count_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_maxCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))
