import unittest

from khausdorff.naive import all_partial_hausdorff, nearest_distances, partial_hausdorff
from tests.helpers import line


class TestNaive(unittest.TestCase):
    def test_nearest_distances(self):
        A = line([1, 2, 3, 6, 7, 8, 12])
        B = line([1, 2, 3, 6, 7, 9])
        self.assertEqual(nearest_distances(A, B), [0, 0, 0, 0, 0, 1, 3])

    def test_all_partial_hausdorff_is_sorted_descending(self):
        A = line([1, 2, 3, 6, 7, 8, 12])
        B = line([1, 2, 3, 6, 7, 9])
        self.assertEqual(all_partial_hausdorff(A, B), [3, 1, 0, 0, 0, 0, 0, 0])

    def test_delta_0_is_the_directed_hausdorff_distance(self):
        # Las mismas dos comprobaciones que hace el módulo `hausdorff` original.
        self.assertEqual(
            partial_hausdorff(line([1, 2, 3, 6, 7, 8, 12]), line([1, 2, 3, 6, 7, 9]), 0),
            3,
        )
        self.assertEqual(
            partial_hausdorff(line([1, 2, 3, 6, 7, 8]), line([1, 2, 3, 6, 7, 9]), 0),
            1,
        )

    def test_directed(self):
        A, B = line([0, 10]), line([0])
        self.assertEqual(all_partial_hausdorff(A, B), [10, 0, 0])
        self.assertEqual(all_partial_hausdorff(B, A), [0, 0])

    def test_identical_sets_give_zeros(self):
        A = line([1, 5, 9])
        self.assertEqual(all_partial_hausdorff(A, A), [0, 0, 0, 0])

    def test_empty_reference_set_is_rejected(self):
        with self.assertRaises(ValueError):
            nearest_distances(line([1]), [])

    def test_k_equal_to_n_is_the_trailing_zero(self):
        self.assertEqual(partial_hausdorff(line([1, 2]), line([1]), 2), 0)

    def test_k_out_of_range(self):
        with self.assertRaises(IndexError):
            partial_hausdorff(line([1, 2]), line([1]), 3)


if __name__ == "__main__":
    unittest.main()
