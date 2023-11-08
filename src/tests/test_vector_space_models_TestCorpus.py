import unittest
from vectorspace.vector_space_models import Document, Corpus


class TestCorpus(unittest.TestCase):
    def setUp(self):
        self.new_doc1 = Document(title = "doc1", words=["freedom", "liberty", "liberty"])
        self.new_doc2 = Document(title = "doc2", words=["liberty", "yelling", "crawled"])
        self.new_doc3 = Document(title = "doc3", words=["justice", "drugs", "oil"])
        self.documents = [self.new_doc1, self.new_doc2, self.new_doc3]
        self.corpus1 = Corpus([self.new_doc1, self.new_doc2])
        self.corpus2 = Corpus(self.documents)

    def test_compute_terms(self):
        terms = self.corpus1._compute_terms()
        self.assertEqual(terms, {"freedom", "liberty", "yell", "crawl"})

    def test_compute_df(self):
        df1 = self.corpus1._compute_df("liberty")
        df2 = self.corpus2._compute_df("liberty")
        self.assertEqual(df1, 2)
        self.assertEqual(df2, 2)

    # def test_check_membership(self):
    # For some reason I can't access the check_membership function
    #   *Make sure to ask Mike later and fix if needed*
    #     term = "freedom"
    #     self.assertEqual()

    def test_compute_tf_idf(self):
        # tf of oil in corpus2 is 1. Because of that we want to calculate the tf for
        # a term that occurs once in a corpus with a size of 3 documents. To do that
        # follow the formula provided in the Google Doc for this assignment
        # the answer is log10(1 + 1) * log10(3 / (1 + 1)) = .053008751
        corpus = Corpus(self.documents)
        score = corpus._compute_tf_idf("oil", doc=self.documents[0])
        self.assertEqual(score, .053008751)


if __name__ == '__main__':
    unittest.main()
