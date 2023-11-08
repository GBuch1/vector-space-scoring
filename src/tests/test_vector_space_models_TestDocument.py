import unittest
from nltk.stem.snowball import SnowballStemmer
from vectorspace.vector_space_models import Document


class TestDocument(unittest.TestCase):
    def setUp(self):
        self.new_doc1 = Document(words=["freedom", "liberty", "justice"])
        self.new_doc2 = Document(words=["fleeing", "yelling", "crawled"])
        self.new_doc3 = Document(words=["taxes", "british", "gunfire", "taxes"])

    def test_filter_words(self):
        exclude_words = {"freedom"}
        self.new_doc1.filter_words(exclude_words)
        self.assertEqual(self.new_doc1.words, ["liberty", "justice"])

    def test_stem_words(self):
        stemmer = SnowballStemmer("english")
        self.new_doc2.stem_words(stemmer)
        self.assertEqual(self.new_doc2.words, ["flee", "yell", "crawl"])

    def test_term_frequency(self):
        self.assertEqual(self.new_doc3.tf("taxes"), 2)
        self.assertEqual(self.new_doc3.tf("british"), 1)
        self.assertEqual(self.new_doc3.tf("gunfire"), 1)
        self.assertEquals(self.new_doc3.tf("tea"), 0)
        self.assertNotEquals(self.new_doc3.tf("tea"), 1)


if __name__ == '__main__':
    unittest.main()
