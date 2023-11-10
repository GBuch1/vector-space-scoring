"""Abstract data type definitions for vector space model that supports
   cosine similarity queries using TF-IDF matrix built from the corpus.
"""
import math
import sys
import concurrent.futures

from math import sqrt, log10
from typing import Callable, Iterable
from nltk.stem import StemmerI
from nltk.stem.snowball import SnowballStemmer

__author__ = "Garrett Buchanan"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Garrett Buchanan", "Mike Ryu"]
__license__ = "MIT"
__email__ = "gbuchanan@westmont.edu"


class Vector:
    """This class is used to create and manipulate vectors and calculate different values on them, such as computing
       the euclidian norm of a vector, computing the dot product of two vectors, and calculating the cossimilarity of
       two vectors."""

    def __init__(self, elements: list[float] | None = None):
        self._vec = elements if elements else []

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= len(self._vec):
            raise IndexError(f"Index out of range: {index}")
        else:
            return self._vec[index]

    def __setitem__(self, index: int, element: float) -> None:
        if 0 <= index < len(self._vec):
            self._vec[index] = element
        else:
            raise IndexError(f"Index out of range: {index}")

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Vector):
            return False
        else:
            return self._vec == other.vec

    def __str__(self) -> str:
        return str(self._vec)

    @property
    def vec(self):
        return self._vec

    @staticmethod
    def _get_cannot_compute_msg(computation: str, instance: object):
        return f"Cannot compute {computation} with an instance that is not a DocumentVector: {instance}"

    def norm(self) -> float:
        """Computes Euclidean norm of the vector."""
        return sqrt(sum([x ** 2 for x in self]))

    def dot(self, other: object) -> float:
        """Computes the Dot product of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("dot product", other))
        else:
            return sum([v1 * v2 for v1, v2 in zip(self.vec, other.vec)])

    def cossim(self, other: object) -> float:
        """Computes the Cosine similarity of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("cosine similarity", other))
        else:
            denominator = (self.norm() * other.norm())  # create a variable for the denominator
            if denominator:
                return self.dot(other) / denominator
            else:
                return 0.0  # return 0.0 if there is no denominator because otherwise it would be undefined

    def boolean_intersect(self, other: object) -> list[tuple[float, float]]:
        """Returns a list of tuples of elements where both `self` and `other` had nonzero values."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("boolean intersection", other))
        else:
            return [(e1, e2) for e1, e2 in zip(self._vec, other._vec) if e1 and e2]


class Document:
    """This class creates a Document containing a title and a list of words. There are methods that allow for
       things like filtration of a set of words, and stemming of words in a Document. There is also a method
       that when given a term will the term frequency of that term in a Document."""
    _iid = 0

    def __init__(self, title: str = None, words: list[str] = None, processors: tuple[set[str], StemmerI] = None):
        Document._iid += 1
        self._iid = Document._iid
        self._title = title if title else f"(Untitled {self._iid})"
        self._words = list(words) if words else []

        if processors:
            exclude_words = processors[0]
            stemmer = processors[1]
            if not isinstance(exclude_words, set) or not isinstance(stemmer, StemmerI):
                raise ValueError(f"Invalid processor type(s): ({type(exclude_words)}, {type(stemmer)})")
            else:
                self.stem_words(stemmer)
                self.filter_words(exclude_words)

    def __iter__(self):
        return iter(self._words)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Document):
            return False
        else:
            return self._title == other.title and self._words == other.words

    def __hash__(self) -> int:
        return hash((self._title, tuple(self._words)))

    def __str__(self) -> str:
        words_preview = ["["]
        preview_size = 5
        index = 0

        while index < len(self._words) and index < preview_size:
            words_preview.append(f"{self._words[index]}, ")
            index += 1
        words_preview.append("... ]")

        return "[{i:04d}]: {title} {words}".format(
            i=self._iid,
            title=self._title,
            words="".join(words_preview)
        )

    @property
    def iid(self):
        return self._iid

    @property
    def title(self):
        return self._title

    @property
    def words(self):
        return self._words

    def filter_words(self, exclude_words: set[str]) -> None:
        """Removes any words from `_words` that appear in `exclude_words` passed in."""
        counter = 0
        words = self._words
        while counter < len(words):  # check that words isn't empty
            if words[counter] in exclude_words:
                words.remove(words[counter])
                # if an item is removed don't advance counter because it shifts the index back one
                counter += 0
            else:
                # if no item is removed advance the counter
                counter += 1

    def stem_words(self, stemmer: StemmerI) -> None:
        """Stems each word in `_words` using SnowballStemmer that gets passed in."""
        stemmer = SnowballStemmer(language='english')  # create a variable for the stemmer
        stemmed_words = []  # create an empty list to append the stemmed word to
        words = self._words
        for word in words:
            stem_word = stemmer.stem(word)  # stem the word using the SnowballStemmer
            stemmed_words.append(stem_word)  # add the stemmed word to the stemmed words list
        self._words = stemmed_words

    def tf(self, term: str) -> int:
        """Computes and return the term frequency of the `term` passed in among `_words`."""
        counter = 0
        words = self._words
        # if not type None and if the word matches the term add 1 to the counter
        if not None:
            for word in words:
                if word == term:
                    counter += 1
            return counter


class Corpus:
    """This Class creates a Corpus, which is a list of Documents, and uses the implemented methods
       to perform tasks such as computing a unique, stemmed, and filtered corpus, computing the
       document frequencies of a term, computing the document frequencies of all terms in a document
       and returning an indexed dictionary, computing tf-idf score of a term in a document, computing
       the tf-idf vector for a given document, and computing the tf-idf matrix for an entire corpus."""

    def __init__(self, documents: list[Document], threads=1, debug=False):
        self._docs: list[Document] = documents

        # Setting flags.
        self._threads: int = threads
        self._debug: bool = debug

        # Bulk of the processing (and runtime) occurs here.
        self._terms = self._compute_terms()
        self._dfs = self._compute_dfs()
        self._tf_idf = self._compute_tf_idf_matrix()

    def __getitem__(self, index) -> Document:
        if 0 <= index < len(self._docs):
            return self._docs[index]
        else:
            raise IndexError(f"Index out of range: {index}")

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)

    @property
    def docs(self):
        return self._docs

    @property
    def terms(self):
        return self._terms

    @property
    def dfs(self):
        return self._dfs

    @property
    def tf_idf(self):
        return self._tf_idf

    def _compute_terms(self) -> dict[str, int]:
        """Computes and returns the terms (unique, stemmed, and filtered words) of the corpus."""
        new_set = set()  # create a set so that values are unique
        for document in self._docs:
            new_set.update(document.words)  # update the set with the words in the current document
        new_set = list(new_set)  # make the set into a list
        return self._build_index_dict(new_set)  # return the list of words as an indexed dictionary

    def _compute_df(self, term) -> int:
        """Computes and returns the document frequency of the `term` in the context of this corpus (`self`)."""
        if self._debug:
            print(f"Started working on DF for '{term}'")
            sys.stdout.flush()

        def check_membership(t: str, doc: Document) -> bool:
            """An efficient method to check if the term `t` occurs in a list of words `doc`."""
            for word in doc:
                if word == t:
                    return True
            return False

        return sum([1 if check_membership(term, doc) else 0 for doc in self._docs])

    def _compute_dfs(self) -> dict[str, int]:
        """Computes document frequencies for each term in this corpus and returns a dictionary of {term: df}s."""
        if self._threads > 1:
            return Corpus._compute_dict_multithread(self._threads, self._compute_df, self._terms.keys())
        else:
            return {term: self._compute_df(term) for term in self._terms.keys()}

    def _compute_tf_idf(self, term, doc=None, index=None):
        """Computes and returns the TF-IDF score for the term and a given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.

        """
        dfs = self._dfs
        doc = self._get_doc(doc, index)
        if term in doc.words and len(self.docs) > 1:  # if a term is in a doc and not at end of doc
            tf_idf = (math.log10(1 + doc.tf(term))) * (math.log10(len(self.docs) / (1 + dfs[term])))  # tf-idf formula
            return tf_idf  # return score
        else:
            return 0.0  # if the term is not present in the doc, return a score of 0

    def compute_tf_idf_vector(self, doc=None, index=None) -> Vector:
        """Computes and returns the TF-IDF vector for the given document.
        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.
        """
        doc = self._get_doc(doc, index)
        tf_idf_list = []    # create a list to store tf-idf scores
        for term in self._terms:
            tf_idfs = self._compute_tf_idf(term, doc)   # compute the tf-idf for the current doc
            tf_idf_list.append(tf_idfs)  # append the tf-idf score to the tf-idfs score list
        return Vector(tf_idf_list)  # return the list as a vector

    def _compute_tf_idf_matrix(self) -> dict[str, Vector]:
        """Computes and returns the TF-IDF matrix for the whole corpus.
        The TF-IDF matrix is a dictionary of {document title: TF-IDF vector for the document}.
        """

        def tf_idf(document):
            if self._debug:
                print(f"Processing '{document.title}'")
                sys.stdout.flush()
            vector = self.compute_tf_idf_vector(doc=document)
            return vector

        matrix = {}
        if self._threads > 1:
            matrix = Corpus._compute_dict_multithread(self._threads, tf_idf, self._docs,
                                                      lambda d: d, lambda d: d.title)
        else:
            for doc in self._docs:  # look at one doc at a time
                matrix[doc.title] = self.compute_tf_idf_vector(doc, None)   # update the doc's title and tf-idf
                if self._debug:
                    print(f"Done with doc {doc.title}")
        return matrix

    def _get_doc(self, document, index):
        """A helper function to None-guard the `document` argument and fetch documents per `index` argument."""
        if document is not None and index is None:
            return document
        elif index is not None and document is None:
            if 0 <= index < len(self):
                return self._docs[index]
            else:
                raise IndexError(f"Index out of range: {index}")

        elif document is None and index is None:
            raise ValueError("Either document or index is required")
        else:
            raise ValueError("Either document or index must be passed in, not both")

    @staticmethod
    def _compute_dict_multithread(num_threads: int, op: Callable, iterable: Iterable,
                                  op_arg_func=lambda x: x, key_arg_func=lambda x: x) -> dict:
        """Experimental generic multithreading dispatcher and collector to parallelize dictionary construction.

        Args:
            num_threads (int): maximum number of threads (workers) to utilize.
            op: (Callable): operation (function or method) to execute.
            iterable: (Iterable): iterable to call the `op` on each item.
            op_arg_func: a function that maps an item of the `iterable` to an argument for the `op`.
            key_arg_func: a function that maps an item of the `iterable` to the key to use in the resulting dict.

        Returns:
            A dictionary of {key_arg_func(an item of `iterable`): op(p_arg_func(an item of `iterable`))}.

        """
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_keys = {executor.submit(op, op_arg_func(item)): key_arg_func(item) for item in iterable}
            for future in concurrent.futures.as_completed(future_to_keys):
                key = future_to_keys[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    print(f"Key '{key}' generated exception:", e, file=sys.stderr)
        return result

    @staticmethod
    def _build_index_dict(lst: list) -> dict:
        """Given a list, returns a dictionary of {item from list: index of item}."""
        return {item: index for (index, item) in enumerate(lst)}
