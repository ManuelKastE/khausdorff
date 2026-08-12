import random
import unittest

from greedypermutation.maxheap import MaxHeap

from khausdorff.lowerboundheap import LowerBoundHeap


class TestLowerBoundHeap(unittest.TestCase):
    def drain(self, heap):
        return [heap.removemax() for _ in range(len(heap))]

    def test_basic_ordering(self):
        heap = LowerBoundHeap()
        for value in [5, 3, 9, 1, 7]:
            heap.insert(value, value)
        self.assertEqual(self.drain(heap), [9, 7, 5, 3, 1])

    def test_remove_preserves_heap_order(self):
        # El MaxHeap original falla esto: `_remove_at_index` de ds2 rellena el
        # hueco con la última entrada y solo la hunde hacia abajo, nunca arriba.
        heap = LowerBoundHeap()
        values = [6, 63, 103, 19, 69, 142, 187]
        for value in values:
            heap.insert(value, value)
        heap.remove(6)
        self.assertEqual(self.drain(heap), sorted(set(values) - {6}, reverse=True))

    def test_upstream_maxheap_really_is_broken(self):
        # Fija la razón por la que existe esta subclase.  Si una versión futura
        # de ds2 arregla `_remove_at_index`, este test empieza a fallar y la
        # subclase puede eliminarse.
        heap = MaxHeap()
        values = [6, 63, 103, 19, 69, 142, 187]
        for value in values:
            heap.insert(value, value)
        heap.remove(6)
        self.assertNotEqual(self.drain(heap), sorted(set(values) - {6}, reverse=True))

    def test_fuzz_remove(self):
        rng = random.Random(1)
        for _ in range(2000):
            values = rng.sample(range(1, 500), rng.randint(4, 20))
            heap = LowerBoundHeap()
            for value in values:
                heap.insert(value, value)
            removed = rng.choice(values)
            heap.remove(removed)
            expected = sorted(set(values) - {removed}, reverse=True)
            self.assertEqual(self.drain(heap), expected)

    def test_changepriority_in_both_directions(self):
        heap = LowerBoundHeap()
        for value in [1, 2, 3]:
            heap.insert(value, value)
        heap.changepriority(1, 100)
        self.assertEqual(heap.findmax(), 1)
        heap.changepriority(1, -100)
        self.assertEqual(heap.findmax(), 3)

    def test_mutable_key_function(self):
        # Como lo usa KHausdorff: la clave lee un dict que cambia por debajo.
        bounds = {"a": 1.0, "b": 2.0}
        heap = LowerBoundHeap(key=lambda x: bounds[x])
        heap.insert("a")
        heap.insert("b")
        self.assertEqual(heap.findmax(), "b")
        bounds["a"] = 5.0
        heap.changepriority("a")
        self.assertEqual(heap.findmax(), "a")

    def test_contains(self):
        heap = LowerBoundHeap()
        heap.insert(1, 1)
        self.assertIn(1, heap)
        heap.remove(1)
        self.assertNotIn(1, heap)


if __name__ == "__main__":
    unittest.main()
