from math import log10
import unittest
from vectorspace.vector_space_models import Document, Corpus, Vector


class TestCorpus(unittest.TestCase):
    def setUp(self):
        self.new_doc1 = Document(title="doc1", words=["freedom", "liberty", "liberty"])
        self.new_doc2 = Document(title="doc2", words=["liberty", "yelling", "crawled"])
        self.new_doc3 = Document(title="doc3", words=["justice", "drugs", "oil"])
        self.new_doc4 = Document(title="doc4", words=["fleeing", "yelling", "crawled"])
        self.new_doc5 = Document(title="doc5", words=["running", "jumping", "fleeing"])
        self.new_doc6 = Document(title="doc_test1", words=["run", "jump"])
        self.new_doc7 = Document(title="doc-test2", words=["run", "cry"])
        self.new_doc8 = Document(title="doc_test3", words=["yell", "anger"])
        self.corpus1 = Corpus(documents=[self.new_doc1, self.new_doc2, self.new_doc4, self.new_doc5])
        self.corpus2 = Corpus(documents=[self.new_doc1, self.new_doc2, self.new_doc3])
        self.corpus3 = Corpus(documents=[self.new_doc6, self.new_doc7, self.new_doc8])

    def test_compute_terms(self):
        terms = self.corpus1._compute_terms()
        expected_outcome = {'crawled': 0, 'fleeing': 1, 'freedom': 2, 'jumping': 3, 'liberty': 4, 'running': 5,
                            'yelling': 6}
        for term in terms:
            self.assertIn(term, expected_outcome)

    def test_compute_df(self):
        df1 = self.corpus1._compute_df("liberty")
        df2 = self.corpus2._compute_df("drugs")
        self.assertEqual(df1, 2)
        self.assertEqual(df2, 1)

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
        score = self.corpus2._compute_tf_idf("oil", self.new_doc3)
        self.assertEqual(score, 0.053008750949996714)

    def test_compute_tf_idf_vector(self):
        test_value1 = self.corpus3.compute_tf_idf_vector(self.new_doc1)
        expected_value1 = Vector([0, -0.08401688247, -0.08401688247])
        self.assertAlmostEqual(test_value1[0], expected_value1[0])

    def test_compute_tf_idf_matrix(self):
        test_vec1 = self.corpus3.compute_tf_idf_vector(doc=self.new_doc6)
        test_vec2 = self.corpus3.compute_tf_idf_vector(doc=self.new_doc7)
        test_vec3 = self.corpus3.compute_tf_idf_vector(doc=self.new_doc8)

        test_mat = {self.new_doc6.title: test_vec1, self.new_doc7.title: test_vec2,
                    self.new_doc8.title: test_vec3}

        expected_mat = self.corpus3._compute_tf_idf_matrix()

        self.assertCountEqual(test_mat, expected_mat)


if __name__ == '__main__':
    unittest.main()
