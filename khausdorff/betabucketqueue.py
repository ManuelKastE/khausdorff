"""
A beta-bucket queue: the approximate max-priority queue of Section 5.2 of
Chubet, Parikh, Sheehy and Sheth (2023).

Items are grouped into geometric buckets: bucket `m` holds every item whose
priority lies in (beta**m, beta**(m+1)].  There are only O(log_beta Delta)
non-empty buckets, and all of the operations k-HAUSDORFF needs run in constant
time except for the descending sweep, which visits each bucket at most once.

Order inside a bucket is arbitrary, which is exactly what buys the speed: the
algorithm never needs to distinguish two nodes whose lower bounds agree to
within a factor of beta.

This is a fresh implementation rather than a reuse of
`greedypermutation.fvm.bucketqueue.BucketQueue`, which cannot be used here:
`insert` on an empty queue raises `ValueError: max() arg is an empty sequence`,
and its bucket index takes `log2` of the priority, which fails outright on the
non-positive priorities that local lower bounds routinely take.
"""

from math import floor, inf, log


class BetaBucketQueue:
    """
    An approximate max-priority queue over geometric buckets of ratio `beta`.

    Non-positive priorities are legal: they all share a sentinel bucket at level
    -inf whose representative value is 0.0.  That bucket sorts below every
    numbered one, so such items are always finished last.
    """

    SENTINEL = -inf

    def __init__(self, beta, items=(), key=lambda x: x):
        if beta <= 1:
            raise ValueError(f"beta must be greater than 1, got {beta}.")
        self.beta = beta
        self._log_beta = log(beta)
        self.key = key
        self.buckets = {}  # level -> set of items; empty buckets are deleted
        self.levels = {}  # item -> level
        for item in items:
            self.insert(item, key(item))

    # -- levels and values --------------------------------------------------

    def level_of(self, priority):
        """
        Return the bucket level of `priority`, i.e. the m with
        beta**m < priority <= beta**(m+1), or the sentinel if priority <= 0.
        """
        if priority <= 0:
            return self.SENTINEL
        # floor gives the m with beta**m <= priority < beta**(m+1); subtracting
        # one at an exact power of beta lands it in the half-open interval the
        # paper uses, (beta**m, beta**(m+1)].
        m = floor(log(priority) / self._log_beta)
        if self.beta**m >= priority:
            m -= 1
        return m

    def value(self, level):
        """
        The representative value of a bucket: its lower endpoint.

        Reporting the lower endpoint is what keeps the algorithm's output a
        valid lower bound, since every item in the bucket has priority above it.
        """
        return 0.0 if level == self.SENTINEL else self.beta**level

    # -- queue operations ---------------------------------------------------

    def insert(self, item, priority=None):
        """Insert `item`, at the bucket for `priority` (or `key(item)`)."""
        if item in self.levels:
            raise RuntimeError(f"{item!r} is already in the queue.")
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        self.levels[item] = level
        self.buckets.setdefault(level, set()).add(item)

    def remove(self, item):
        """Remove `item` from the queue."""
        if item not in self.levels:
            raise RuntimeError(f"{item!r} is not in the queue.")
        level = self.levels.pop(item)
        bucket = self.buckets[level]
        bucket.discard(item)
        if not bucket:
            del self.buckets[level]

    def changepriority(self, item, priority=None):
        """Move `item` to the bucket matching its new priority."""
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        if self.levels.get(item) == level:
            return
        self.remove(item)
        self.insert(item, priority)

    def maxlevel(self):
        """The highest occupied level, or None if the queue is empty."""
        return max(self.buckets) if self.buckets else None

    def findmax(self):
        """
        An arbitrary item from the highest occupied bucket.

        Only approximately the maximum: any item within a factor of beta of the
        true maximum is a legal answer.
        """
        if not self.buckets:
            raise RuntimeError("The queue is empty.")
        return next(iter(self.buckets[self.maxlevel()]))

    def removemax(self):
        """Remove and return an item from the highest occupied bucket."""
        item = self.findmax()
        self.remove(item)
        return item

    def levels_at_or_above(self, threshold):
        """
        The occupied levels `j >= threshold`, in decreasing order.

        Materialised as a list so that the caller can empty buckets while
        iterating.  Pass `-inf` to sweep the whole queue.
        """
        return sorted((j for j in self.buckets if j >= threshold), reverse=True)

    def pop_level(self, level):
        """Remove and return every item in the bucket at `level`, as a set."""
        items = self.buckets.pop(level, set())
        for item in items:
            del self.levels[item]
        return items

    def __contains__(self, item):
        return item in self.levels

    def __len__(self):
        return len(self.levels)
