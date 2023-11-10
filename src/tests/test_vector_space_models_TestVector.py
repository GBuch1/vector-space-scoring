import unittest
from vectorspace.vector_space_models import Vector


class TestVector(unittest.TestCase):
    def setUp(self):
        self.vec1 = Vector([3, 2])
        self.vec2 = Vector([5, 3])
        self.vec3 = Vector([9, 8])
        self.vec4 = Vector([6, 0])
        self.vec5 = Vector([1, 2, 3])
        self.vec6 = Vector([8, 9, 10])
        self.vec7 = Vector([2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
        self.vec8 = Vector([])
        self.vec9 = Vector([])
        self.int = 7

    def test_constructor(self):
        self.assertEqual(Vector([3, 2]), self.vec1)
        self.assertEqual(Vector([5, 3]), self.vec2)

    def test_getters_and_setters(self):
        vec = Vector([3, 9])
        vec.object = [5, 12]
        self.assertEqual([5, 12], vec.object)

    def test_euclidean_norm(self):
        # when two vectors are given, calculate that
        # they are equal to sqt((x**2)+(y**2))
        self.assertAlmostEqual(self.vec1.norm(), 3.605551275)
        self.assertAlmostEqual(self.vec2.norm(), 5.830951895)
        self.assertAlmostEqual(self.vec3.norm(), 12.04159458)
        self.assertEqual(self.vec4.norm(), 6)
        self.assertAlmostEqual(self.vec7.norm(), 19.62141687)
        self.assertEqual(self.vec8.norm(), 0.0)

    def test_dot_product(self):
        # When two vectors are given, calculate their dot product and make sure it
        # is consistent with the formula vec_x.dot(vec_y) = [(vec_x[0] * vec_y[0]) +
        # (vec_x[1] * vec_y[1] + etc....
        self.assertEqual(self.vec5.dot(self.vec6), 56)
        self.assertEqual(self.vec1.dot(self.vec2), 21)
        self.assertEqual(self.vec1.dot(self.vec7), 8)
        self.assertEqual(self.vec7.dot(self.vec1), 8)
        self.assertEqual(self.vec7.dot(self.vec8), 0.0)
        with self.assertRaises(ValueError):
            self.vec5.dot(int)

    def test_cosine_similarity(self):
        # Should return the answer to the formula vec_x.dot(vec_y)/(vec_x.norm()) *
        # (vec_y.norm())
        # vec5.dot(vec6) = 56
        # vec5.norm() = 3.741657387
        # vec6.norm() = 15.65247584
        # 56 / (3.741657387) * (15.65247584) = .956182888
        self.assertAlmostEqual(self.vec5.cossim(self.vec6), 0.956182888)
        self.assertAlmostEqual(self.vec5.cossim(self.vec7), 0.217934306)
        with self.assertRaises(ValueError):
            self.vec5.cossim(int)
        self.assertEqual(self.vec8.cossim(self.vec9), 0.0)


if __name__ == '__main__':
    unittest.main()
