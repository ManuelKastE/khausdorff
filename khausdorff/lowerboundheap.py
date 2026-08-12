"""
An exact max-heap of local lower bounds, with a working `remove`.

k-HAUSDORFF removes nodes from the middle of the lower bound heap all the time:
whenever an A-side node is split its parent is retired, and whenever the
finishing condition fires a node is cut out.  The heap it inherits from does not
support that correctly, hence this subclass.

The bug is in `ds2`, one level below `greedypermutation`.  `PriorityQueue.
_remove_at_index` fills the hole with the last entry and then only sifts it
*down*:

    def _remove_at_index(self, index):
        L = self._entries
        self._swap(index, len(L) - 1)
        del self._itemmap[L[-1].item]
        L.pop()
        self._downheap(index)          # <- an entry that must move up cannot

The entry moved in comes from an unrelated subtree, so it may well belong
*above* the hole.  When it does, the heap order breaks silently and `findmax`
starts handing back the wrong node.  Fuzzing puts this at roughly 1% of removals
on random inputs, which is more than enough to scramble the output ordering.

Nothing else in `greedypermutation` calls `remove`, which is presumably why the
bug has gone unnoticed; `dist_H` and `DualTreeSearch` only ever insert and pop
the maximum.
"""

from greedypermutation.maxheap import MaxHeap


class LowerBoundHeap(MaxHeap):
    """A `MaxHeap` whose `remove` preserves the heap order."""

    def changepriority(self, item, priority=None):
        """
        Move `item` to `priority`, or to `key(item)` if none is given.

        `MaxHeap` negates the priority in `insert` but inherits
        `changepriority` unchanged, so upstream the two disagree about sign and
        an explicit priority silently inverts the order.  Negating here keeps
        them consistent.
        """
        if priority is not None:
            priority = -priority
        super().changepriority(item, priority)

    def _remove_at_index(self, index):
        entries = self._entries
        self._swap(index, len(entries) - 1)
        del self._itemmap[entries[-1].item]
        entries.pop()
        if index < len(entries):
            # The entry swapped into the hole may belong either above or below
            # it.  Only one of these two calls can have an effect, the same way
            # `changepriority` handles an arbitrary priority change.
            self._upheap(index)
            self._downheap(index)

    def __contains__(self, item):
        return item in self._itemmap
