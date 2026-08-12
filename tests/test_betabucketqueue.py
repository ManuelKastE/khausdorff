import random
import unittest
from math import inf

from khausdorff.betabucketqueue import BetaBucketQueue


class TestBetaBucketQueue(unittest.TestCase):
    def test_beta_must_exceed_one(self):
        for beta in (1, 0.5, 0):
            with self.assertRaises(ValueError):
                BetaBucketQueue(beta)

    def test_insert_into_an_empty_queue(self):
        # La regresión que vuelve inutilizable a greedypermutation.fvm.bucketqueue:
        # su `insert` llama a `max()` sobre un dict de buckets vacío y lanza error.
        q = BetaBucketQueue(1.5)
        q.insert("a", 3.0)
        self.assertEqual(len(q), 1)
        self.assertEqual(q.findmax(), "a")

    def test_bucket_invariant(self):
        beta = 1.25
        q = BetaBucketQueue(beta)
        for priority in (0.001, 0.3, 1.0, 1.25, 2.7, 40.0, 1e6):
            level = q.level_of(priority)
            self.assertLess(beta**level, priority)
            self.assertLessEqual(priority, beta ** (level + 1))

    def test_exact_powers_land_in_the_lower_bucket(self):
        # Los buckets son (beta**m, beta**(m+1)], así que beta**m cae en el m-1.
        beta = 2.0
        q = BetaBucketQueue(beta)
        self.assertEqual(q.level_of(beta**5), 4)

    def test_non_positive_priorities_use_the_sentinel(self):
        q = BetaBucketQueue(1.5)
        for priority in (0.0, -1.0, -1e9):
            self.assertEqual(q.level_of(priority), -inf)
        self.assertEqual(q.value(-inf), 0.0)

    def test_sentinel_sorts_below_everything(self):
        q = BetaBucketQueue(1.5)
        q.insert("zero", 0.0)
        q.insert("small", 1e-6)
        self.assertEqual(q.findmax(), "small")
        q.remove("small")
        self.assertEqual(q.findmax(), "zero")

    def test_value_is_the_lower_endpoint(self):
        beta = 1.5
        q = BetaBucketQueue(beta)
        for priority in (0.4, 2.0, 17.0):
            level = q.level_of(priority)
            # Reportar el extremo inferior es lo que mantiene la salida como
            # una cota inferior válida de la distancia verdadera.
            self.assertLess(q.value(level), priority)

    def test_remove_and_contains(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 2.0)
        self.assertIn("a", q)
        q.remove("a")
        self.assertNotIn("a", q)
        self.assertEqual(len(q), 0)
        self.assertIsNone(q.maxlevel())
        with self.assertRaises(RuntimeError):
            q.remove("a")
        with self.assertRaises(RuntimeError):
            q.findmax()

    def test_duplicate_insert_is_rejected(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 2.0)
        with self.assertRaises(RuntimeError):
            q.insert("a", 3.0)

    def test_changepriority_moves_between_buckets(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 1.0)
        q.insert("b", 100.0)
        self.assertEqual(q.findmax(), "b")
        q.changepriority("a", 10000.0)
        self.assertEqual(q.findmax(), "a")
        q.changepriority("a", 0.01)
        self.assertEqual(q.findmax(), "b")

    def test_empty_buckets_are_discarded(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 10.0)
        level = q.maxlevel()
        q.remove("a")
        self.assertNotIn(level, q.buckets)

    def test_levels_at_or_above(self):
        q = BetaBucketQueue(2.0)
        for name, priority in [("a", 1.5), ("b", 3.0), ("c", 6.0), ("d", 100.0)]:
            q.insert(name, priority)
        levels = q.levels_at_or_above(-inf)
        self.assertEqual(levels, sorted(levels, reverse=True))
        self.assertEqual(len(levels), 4)
        high = q.levels_at_or_above(q.level_of(6.0))
        self.assertEqual(len(high), 2)  # c and d

    def test_pop_level_empties_the_bucket(self):
        q = BetaBucketQueue(2.0)
        q.insert("a", 3.0)
        q.insert("b", 3.5)  # same bucket as "a"
        level = q.level_of(3.0)
        self.assertEqual(q.level_of(3.5), level)
        self.assertEqual(q.pop_level(level), {"a", "b"})
        self.assertEqual(len(q), 0)
        self.assertEqual(q.pop_level(level), set())

    def test_removemax_drains_in_non_increasing_buckets(self):
        rng = random.Random(0)
        beta = 1.3
        q = BetaBucketQueue(beta)
        priorities = {i: rng.uniform(0.01, 1000) for i in range(200)}
        for item, priority in priorities.items():
            q.insert(item, priority)
        levels = []
        while len(q):
            item = q.removemax()
            levels.append(q.level_of(priorities[item]))
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_key_function_is_used_when_no_priority_given(self):
        q = BetaBucketQueue(1.5, items=[1.0, 8.0, 4.0])
        self.assertEqual(q.findmax(), 8.0)


if __name__ == "__main__":
    unittest.main()
