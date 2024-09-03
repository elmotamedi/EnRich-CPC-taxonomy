"""Near Duplicate Text Filter using LSH."""
import json
from typing import List, Tuple, Union, Dict, Any, Optional
from functools import partial
from multiprocessing import Pool
# from textpreprocess import word_analyzer
from tqdm import tqdm

from datasketch import LeanMinHash, MinHash, MinHashLSH
import xxhash

def word_analyzer(text: str, ngram_range: Tuple[int, int], length: int, is_preprocessed: bool) -> List[str]:
    """
    Generate n-grams from the input text.
    
    Args:
        text (str): The input text.
        ngram_range (Tuple[int, int]): The range of n-grams to generate (min, max).
        length (int): Token length.
        is_preprocessed (bool): Flag indicating if text is preprocessed.

    Returns:
        List[str]: A list of n-grams.
    """
    from nltk import ngrams
    from nltk.tokenize import word_tokenize

    if not is_preprocessed:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()

    ngrams_list = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_list.extend([' '.join(gram) for gram in ngrams(tokens, n)])

    return ngrams_list
class DedupLSH:
    """Filter.

    This version does not double check the jaccard sim

        For small data (few fps), bands=42, rows=3?
        For big data (many fps), bands=8, rows=16?
    """

    lsh: MinHashLSH

    def __init__(
        self,
        threshold: float = 0.90,
        n_perm: int = 128,
        ngram_range: Tuple[int, int] = (1, 3),
        min_char_len: int = 10,
        token_len: int = 128,
        preprocess_text: bool = True,
        reset_after: Optional[int] = None,
    ):
        """Initialize the duplicate filter."""
        self.analyzer_f = partial(
            word_analyzer,
            ngram_range=ngram_range,
            length=token_len,
            is_preprocessed=not preprocess_text,
        )
        self.n_perm = n_perm
        self.min_len = min_char_len

        self.threshold = threshold
        self.reset_after = reset_after
        self.lsh = MinHashLSH(threshold=threshold, num_perm=n_perm)
        self.ii = 1

    def reset(self):
        """Reset the filter by deleting the index."""
        print(f"resetting lsh filter")
        del self.lsh
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.n_perm)
        self.ii = 1

    @staticmethod
    def _hash_func(d):
        """Return hash function used by LSH structures and queries."""
        return xxhash.xxh3_64(d).intdigest()

    @staticmethod
    def _make_hash(s, n_perm, analyzer_f):
        """Return the minhash of a string.

        Args:
            s (str): The string.
            n_perm (int): The number of permutations for the minhash.

        Returns:
            A tuple containing:
                1. MinHash of the string
        """
        tset = set(analyzer_f(s))
        if len(tset) == 0:
            raise ValueError(f"bad string (length=0): {s} -> {tset}")

        m = MinHash(num_perm=n_perm, hashfunc=DedupLSH._hash_func)  # type: ignore
        for d in tset:
            m.update(d.encode("utf8"))
        m = LeanMinHash(m)

        return m

    def parallel_hash_strings(
        self, strings: List[str], pool
    ) -> List[LeanMinHash]:
        """Parallel hash strings."""
        f = partial(
            self._make_hash, n_perm=self.n_perm, analyzer_f=self.analyzer_f
        )
        hashes = pool.map(f, strings)
        return hashes

    def _is_near_duplicate_or_add_hash(self, h: LeanMinHash) -> bool:
        """Check if a document is a near duplicate.

        Return: True if the document is a near duplicate, False otherwise.
        """
        res = self.lsh.query(h)
        if res and len(res) > 0:
            return True

        if self.reset_after is not None and self.ii >= self.reset_after:
            self.reset()

        id_str = str(self.ii)
        self.ii += 1
        self.lsh.insert(id_str, h)
        return False

    def is_near_duplicate(self, doc: Union[str, Dict[str, Any]]) -> bool:
        """Check if a document is a near duplicate.

        Return: True if the document is a near duplicate, False otherwise.
        """
        text: str
        if isinstance(doc, str):
            text = doc
        elif isinstance(doc, dict):
            text = doc["text"]
        else:
            raise ValueError("doc must be a string or a dictionary")
        m = DedupLSH._make_hash(text, self.n_perm, self.analyzer_f)
        res = self.lsh.query(m)
        if res and len(res) > 0:
            return True
        return False

    def _is_near_duplicate_or_add(
        self, doc: Union[str, Dict[str, Any]]
    ) -> bool:
        """Check if a document is a near duplicate.

        Return: True if the document is a near duplicate, False otherwise.
        """
        text: str
        if isinstance(doc, str):
            text = doc
        elif isinstance(doc, dict):
            text = doc["text"]
        else:
            raise ValueError("doc must be a string or a dictionary")
        m = DedupLSH._make_hash(text, self.n_perm, self.analyzer_f)
        res = self.lsh.query(m)
        if res and len(res) > 0:
            return True

        if self.reset_after is not None and self.ii >= self.reset_after:
            self.reset()

        id_str = str(self.ii)
        self.ii += 1
        self.lsh.insert(id_str, m)
        return False

    def force_add_from_list(self, docs: Union[List[Dict[str, Any]], List[str]]):
        """Force add a list of documents to the filter."""
        for doc in docs:
            text: str
            if isinstance(doc, str):
                text = doc
            elif isinstance(doc, dict):
                text = doc["text"]
            else:
                raise ValueError("doc must be a string or a dictionary")
            m = DedupLSH._make_hash(text, self.n_perm, self.analyzer_f)
            id_str = str(self.ii)
            self.ii += 1
            self.lsh.insert(id_str, m)

    def uniq(
        self, docs: Union[List[str], List[Dict[str, Any]]]
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Filter list, keep only non duplicates."""
        res = []
        for s in docs:
            truth = self._is_near_duplicate_or_add(s)
            if not truth:
                res.append(s)
        return res

    def filter(
        self, docs: Union[List[str], List[Dict[str, Any]]]
    ) -> List[bool]:
        """Filter list, return boolean array."""
        res = []
        for s in docs:
            is_dup = self._is_near_duplicate_or_add(s)
            res.append(not is_dup)
        return res

    def filter_hashes(self, hashes: List[LeanMinHash]) -> List[bool]:
        """Filter list, return boolean array."""
        res = []
        for h in hashes:
            is_dup = self._is_near_duplicate_or_add_hash(h)
            res.append(not is_dup)
        return res


def load_json(line):
    """Load json from a line."""
    try:
        line = line.strip()
        r = json.loads(line)
    except:
        return None
    if "text" not in r:
        return None
    return r


class DedupLSHFile:
    """Deduplicate a file using MinHashLSH.

    Expects file to be json lines with a "text" key.
    """

    def __init__(
        self,
        input_file,
        output_file,
        dict_reader_f=load_json,
        skip_header: bool = False,
        batch_size: int = 1024,
    ):
        """Initialize the file deduplicator.

        Args:
            input_file (str): The input file path (json lines with "text" key).
            output_file (str): The output file path (will append if exists).
            batch_size (int): The batch size.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.dict_reader_f = dict_reader_f
        self.skip_header = skip_header

    def dedup(self, deduper, pool, exta_dict_filter=None):
        """Deduplicate the input file write to the outputfile."""
        n_seen = 0
        n_dups = 0
        n_res = 0
        last_print = 0

        def gen_lines(input_file):
            """Generate lines from the input file."""
            f = open(input_file, "r")
            if self.skip_header:
                f.readline()
            # generate lines
            lines = []
            for line in f:
                lines.append(line.strip())
                if len(lines) == self.batch_size:
                    yield lines
                    lines = []
            if lines:
                yield lines

        for lines in tqdm(
            gen_lines(self.input_file),
            desc=f"Deduping {self.input_file} (batch size: {self.batch_size})",
        ):
            n_seen += len(lines)
            # load the json
            ds = pool.map(self.dict_reader_f, lines)
            # extra filtering
            if exta_dict_filter is not None:
                filter_mask = pool.map(exta_dict_filter, ds)
                ds = [d for d, t in zip(ds, filter_mask) if t]
                lines = [line for line, t in zip(lines, filter_mask) if t]
            # get the texts
            texts = [d["text"] for d in ds]
            # create the hashes
            hashes = deduper.parallel_hash_strings(texts, pool)
            # check for duplicates
            filter_mask = deduper.filter_hashes(hashes)
            # keep only the non-duplicate lines
            lines = [line for line, m in zip(lines, filter_mask) if m]
            n_dups += len(filter_mask) - len(lines)
            n_res += len(lines)
            # write to the output file
            with open(self.output_file, "a") as f:
                for line in lines:
                    print(f"{line}", file=f)
            if n_seen - last_print >= 500000:
                last_print = n_seen
                print(f"seen: {n_seen}, dups: {n_dups}, res: {n_res}")

        print(f"Finsihed deduping {self.input_file}")
        print(f"seen: {n_seen}, dups: {n_dups}, res: {n_res}")
        print()
