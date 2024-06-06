from typing import Iterable, Union, List, Tuple, Dict, Generator
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import time
import warnings
import itertools
import numpy as np
import numpy.random as npr
import wonderwords as ww
from tqdm.auto import tqdm
from natsort import natsorted
from string import ascii_lowercase

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

##################################################################################################################

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

##################################################################################################################

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

        if len(counts) < 3:  # Sentence consists of spacing/numbers/punctuation, or very few letters, such that counting characters becomes counting length.
            continue

        keys          = list(counts.keys())
        probabilities = normalised_softmax(np.array(list(counts.values())), tau=0.05)  # Lower temperature (towards 0) makes the sample less uniform and more likely to sample the argmax, which helps to avoid 1-count samples.
        for index in RNG.choice(len(keys), p=probabilities, size=min(examples_per_sentence, len(keys)), replace=False):
            yield counts[keys[index]], string, keys[index]


def dataset_maxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string)
        for k in list(counts.keys()):
            if not k.isalpha():
                counts.pop(k)

        if len(counts) < 3:  # Sentence consists of spacing/numbers/punctuation, or very few letters, such that counting characters becomes counting length.
            continue

        first,second = counts.most_common(n=2)
        if first[1] != second[1]:  # Unique max
            yield first[1], string


def dataset_argmaxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string.lower())  # Count in lowercase

        # Filter weird characters
        keys = list(counts.keys())
        for key in keys:
            if not key.isalpha():
                counts.pop(key)

        if len(counts) < 3:  # This is a nonsense string. It doesn't even contain 3 different unique letters...
            continue

        # Only if the maximum count is unique do we use this string, and only if that maximum is a lowercase letter.
        first,second = counts.most_common(n=2)
        if first[1] != second[1] and first[0].islower():
            yield first[0], string


def take(n: int, generator: Iterable):
    i = 0
    for thing in generator:
        if i >= n:
            break
        yield thing
        i += 1

    if i != n:  # Not the same as "no break <=> bad" because when the generator yields exactly n items, you still quit the loop without a break but in that case it's fine.
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


def iterateTsv(path: Path) -> Generator[tuple, None, None]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            yield tuple(line.strip().split("\t"))


def tuplesToFile(path: Path, generator: Iterable[tuple], sep: str="\t", lastsep: str="\t"):
    path.parent.mkdir(exist_ok=True, parents=True)
    time.sleep(0.5)
    print("Writing", path.as_posix(), "...")
    time.sleep(0.5)
    with open(path, "w", encoding="utf-8") as handle:
        for tup in generator:
            handle.write(sep.join(map(str,tup[:-2])) + sep + lastsep.join(map(str,tup[-2:])) + "\n")
    return path


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


class BinningAlgorithm(ABC):
    """
    Given a bunch of values that have a frequency, bin them into k bins with roughly uniform cumulative frequency.

    The amount of possible binnings grows worse than factorial, namely with the Bell numbers: https://en.wikipedia.org/wiki/Bell_number
    The amount of binnings into exactly k bins is equally complex, namely the 2nd-kind Stirling numbers: https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

    That means this is a particularly hard problem to solve. (In fact, it is NP-hard.)
    """

    Bin: Tuple[int, set]  # (frequency, keys)

    @abstractmethod
    def _partition(self, frequencies: Counter, k: int) -> List["BinningAlgorithm.Bin"]:
        pass

    def partition(self, frequencies: Counter, k: int) -> List[set]:
        if len(frequencies) < k:
            k = len(frequencies)
            warnings.warn(f"Can't bin {len(frequencies)} values into k={k} bins. Will use that as k instead.")

        sizes_and_bins = self._partition(frequencies, k)

        # Visualise
        BinningAlgorithm.printBins(sizes_and_bins)
        # print("Mean absolute deviation from uniform distribution:", 1 / n_bins * sum(abs(bin_size / total_weight - target_weight / total_weight) for bin_size,_ in bins))
        return [b for _,b in sizes_and_bins]

    def binTsv(self, path: Path, column: int, k_bins: int) -> List[set]:
        """
        Read a column from the TSV and create sets of values such that all the sets appear with equal frequency.
        """
        frequencies = BinningAlgorithm.countTsvValues(path, column)
        return self.partition(frequencies, k_bins)

    @staticmethod
    def printBins(bins: List["BinningAlgorithm.Bin"]):
        total = sum(map(lambda t: t[0], bins))
        print(f"{len(bins)} bins (target size {int(total/len(bins))} = {total}/{len(bins)}):")
        for s, b in bins:
            print("\t", s, natsorted(b))

    @staticmethod
    def countTsvValues(path: Path, column: int) -> Counter:
        frequencies = Counter()
        for row in iterateTsv(path):
            frequencies[row[column]] += 1
        return frequencies

    def applyBinsToTsv(self, in_path: Path, column: int, k_bins: int):
        bins = self.binTsv(in_path, column, k_bins)
        bin_map = dict()
        for bin_index, bin in enumerate(bins):
            for value in bin:
                bin_map[value] = bin_index

        out_path = in_path.with_stem(in_path.stem + f"_binned-{k_bins}")
        return tuplesToFile(
            out_path,
            (row[:column] + (str(bin_map[row[column]]),) + row[column+1:] for row in iterateTsv(in_path))
        )


class NewBinWhenBiggestFitsNowhere(BinningAlgorithm):
    """
    Make bin with biggest unused key, then fill it with keys in descending order without exceeding the expected
    bin size of total/k. Repeat this process until you have k bins. Then add remaining values like BiggestKeySmallestBinFirst.
    This is a quadratic algorithm since it iterates in a triangle.

    Note: will not produce k bins if total/k is so large (due to an outlier) that you can fill one of the later bins
    with all remaining values without reaching the target_weight, leaving a bunch of empty buckets.
    """

    def _partition(self, frequencies: Counter, k: int) -> List["BinningAlgorithm.Bin"]:
        keys_and_counts = list(sorted(frequencies.items(), key=lambda t: t[1], reverse=True))  # Big to small.

        total_weight  = frequencies.total()
        target_weight = total_weight / k
        bins = []
        keys_used = set()
        for i in range(len(keys_and_counts)):
            if len(bins) == k:
                break

            key, count = keys_and_counts[i]
            if key in keys_used:
                continue

            current_bin      = {key}
            current_bin_size = count
            for j in range(i+1,len(keys_and_counts)):
                extra_key, extra_count = keys_and_counts[j]
                if extra_key in keys_used:
                    continue

                if current_bin_size + extra_count <= target_weight:
                    current_bin.add(extra_key)
                    current_bin_size += extra_count

            bins.append((current_bin_size,current_bin))
            keys_used.update(current_bin)

        if len(keys_used) != len(keys_and_counts):
            for i in range(len(keys_and_counts)):  # These are already sorted from big to small. We will now add the biggest elements to the smallest bin.
                key, count = keys_and_counts[i]
                if key in keys_used:
                    continue

                idx_of_smallest_bin = min(range(k), key=lambda i: bins[i][1])  # Would be faster if you did this with a priority queue, but eh, realistically, how many times are you going to run this?
                current_bin_size, current_bin = bins[idx_of_smallest_bin]
                current_bin.add(key)
                current_bin_size += count
                bins[idx_of_smallest_bin] = (current_bin_size, current_bin)

        return bins


class BiggestKeySmallestBinFirst(BinningAlgorithm):
    """
    Iterate over keys in order of descending frequency and put them in the currently smallest bin.
    """

    def _partition(self, frequencies: Counter, k: int) -> List["BinningAlgorithm.Bin"]:
        keys_and_counts = list(sorted(frequencies.items(), key=lambda t: t[1], reverse=True))  # Big to small.

        bins = [(0, set()) for _ in range(k)]
        for i in range(len(keys_and_counts)):
            key, count = keys_and_counts[i]
            first_bin_size, first_bin = bins[0]
            first_bin.add(key)

            # Re-insert bin. If you want to make this fast, you should use a heap or an insertion-sort insert.
            bins[0] = (first_bin_size + count, first_bin)
            bins = sorted(bins)

        return bins


class QueueWithCounts:

    def __init__(self):
        self.ordered_keys: List[str] = []
        self.counts: Dict[str,int]   = dict()
        self.total: int              = 0

    def push(self, key: str, count: int):
        if key in self.counts:
            raise KeyError(f"Key {key} is already in queue.")

        self.ordered_keys.append(key)
        self.counts[key] = count
        self.total += count

    def pop(self) -> Tuple[str, int]:
        if not self.ordered_keys:
            raise IndexError("Queue is empty, so there is no first element to pop.")

        first_key = self.ordered_keys.pop(0)
        count = self.counts.pop(first_key)
        self.total -= count
        return first_key, count


class OrderPreservingOverflow(BinningAlgorithm):
    """
    Sort the values themselves, not their frequency. Take the first k to initialise the bins. Then, add successive values
    to the last bin until it exceeds total/k in size. At that point, push the smallest value (again, not by frequency)
    to the previous bin and apply the same rule there. Don't push to a bin that has already exceeded total/k (which can't
    happen in all bins, because then sum_{i=1}^k bin_i > sum_{i=1}^k total/k = total.

    By pushing values in from the right, the tendency is for the rightmost bin to receive most values, which simulates an
    infinite half-open bin in that case.
    """

    def __init__(self, margin: float=1.0):
        """
        :param margin: Multiplier on total/k to use for the bucket size that makes a bucket ripple.
        """
        self.margin_fraction = margin

    def _partition(self, frequencies: Counter, k: int) -> List["BinningAlgorithm.Bin"]:
        # Sort by key, not frequency.
        if all(isinstance(key, str) for key in frequencies):  # Needs natural sort
            keys_and_counts = list(natsorted(frequencies.items(), key=lambda t: t[0]))
        else:  # Regular sort
            keys_and_counts = list(sorted(frequencies.items(), key=lambda t: t[0]))

        max_bin_size = frequencies.total()/k * self.margin_fraction
        bins = [QueueWithCounts() for _ in range(k)]

        # Initialise bins
        for i in range(k):
            key, count = keys_and_counts[i]
            bins[i].push(key, count)

        # Add the rest
        for i in range(k,len(keys_and_counts)):
            key, count = keys_and_counts[i]
            bins[k-1].push(key, count)

            any_overflow = True
            while any_overflow:
                any_overflow = False
                for current_bin_index in range(k-1, 0, -1):  # 0 is not included because bin 0 can never overflow.
                    current_bin   = bins[current_bin_index]
                    preceding_bin = bins[current_bin_index-1]
                    if current_bin.total > max_bin_size and preceding_bin.total <= max_bin_size:  # Current bin overflows to the previous bin, UNLESS the previous bin is already over capacity OR this is the last bin.
                        key, count = current_bin.pop()
                        preceding_bin.push(key, count)
                        # print("Attempting to add count", count, f"to bin {current_bin_index-1} of", [q.total for q in bins], "with threshold", max_bin_size)
                        any_overflow = True

        # Convert to the expected bin format
        return [(queue.total, set(queue.ordered_keys)) for queue in bins]


class OrderPreservingRipple(BinningAlgorithm):
    """
    Same as the above class, except you initialise by adding all values to the bins, and then let the bins with excess
    ripple their contents outwards.

    There are three possible situations:
        1. The bin to the left and right have excess.
        2. Only one of the bins on the left and right have excess.
        3. Neither of the bins on the left and right have excess.

    Situation 2 means you need to ripple your excess to that bin. The others are more difficult to deal with, and also,
    situation 2 can make the bucket with excess not have excess and the left/right bucket you ripple to have excess, so
    you get oscillating behaviour... Too difficult :/
    """

    def _partition(self, frequencies: Counter, k: int) -> List["BinningAlgorithm.Bin"]:
        pass


def limitTsvValueCount(path: Path, column: int, max_frequency: int):
    """
    In the given column, find values that appear more than the max_frequency. Then sample max_frequency of its examples
    and ditch the rest.
    """
    value_counts = BinningAlgorithm.countTsvValues(path, column)
    values_to_be_reduced = {key for key,count in value_counts.items() if count > max_frequency}
    rows_to_keep = {value: set(RNG.choice(value_counts[value], size=max_frequency, replace=False))
        for value in values_to_be_reduced
    }

    def filteredTuples() -> Generator[tuple, None, None]:
        value_specific_enumerate = Counter()
        for row in iterateTsv(path):
            value = row[column]
            if value not in values_to_be_reduced or value_specific_enumerate[value] in rows_to_keep[value]:
                yield row
            value_specific_enumerate[value] += 1


    out_path = path.with_stem(path.stem + f"_limited-{max_frequency}")
    return tuplesToFile(
        out_path,
        filteredTuples()
    )


def testBinAmounts(binner: BinningAlgorithm, path: Path, column: int, max_k: int):
    print(f"Testing {max_k} bin amounts for {path.stem}:")
    for k in range(1,max_k+1):
        binner.binTsv(path, column=column, k_bins=k)



if __name__ == "__main__":
    out_path = PATH_DATA / "probing" / "Visual" / "odd_character_out_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateSentences(loop=False))))

    out_path = PATH_DATA / "probing" / "Visual" / "odd_character_out_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))

    ##############################################################################

    binner = OrderPreservingOverflow(margin=1.1)

    out_path = PATH_DATA / "probing" / "Visual" / "count_character_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateSentences(loop=False), examples_per_sentence=3)), lastsep="|")
    # histogramOfTsv(out_path, column=1)
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "Visual" / "count_character_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))), examples_per_sentence=3)), lastsep="|")
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 2 or 4
    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "Visual" / "max_count_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(80_000, dataset_maxCharacter(generateSentences(loop=False))))
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "Visual" / "max_count_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_maxCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    ##############################################################################################################

    binner = BiggestKeySmallestBinFirst()

    out_path = PATH_DATA / "probing" / "Visual" / "argmax_count_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(80_000, dataset_argmaxCharacter(generateSentences(loop=False))))
    # testBinAmounts(binner, out_path, column=1, max_k=7)  # 5 is nice IF you subsample the 'e' and 't' examples to 8k (making the dataset 40k instead).
    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=5)
    filtered_path = limitTsvValueCount(binned_path, column=1, max_frequency=8_000)
    binner.binTsv(filtered_path, column=1, k_bins=5)

    out_path = PATH_DATA / "probing" / "Visual" / "argmax_count_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_argmaxCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))
    # testBinAmounts(binner, out_path, column=1, max_k=7)  # 5 is nice but you need to cap every class to about 9900

    binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=5)
    filtered_path = limitTsvValueCount(binned_path, column=1, max_frequency=9900)
    binner.binTsv(filtered_path, column=1, k_bins=5)
