"""
The k-HAUSDORFF variant of Section 5.2: a beta-bucket queue as the lower bound
heap.

`KHausdorff` keeps an exact max-heap of local lower bounds, which costs
O(log n) per update.  Section 5.2 observes that an approximation is already
being made, so the lower bounds can be bucketed geometrically instead, bringing
the heap operations down to constant time and the whole algorithm to

    (2 + 1/eps)^O(d) n + O(log_beta Delta),      beta = 1 + eps/2.

Two things change with respect to the exact variant.  Nodes are finished a
whole bucket at a time rather than one at a time, and a lower bound update that
would push a node above the current threshold finishes the node instead of
moving it.  Both follow Observation 7 of the paper, which is what guarantees
that the descending sweep visits each bucket only once.
"""

from math import ceil, inf, log

from khausdorff.betabucketqueue import BetaBucketQueue
from khausdorff.khausdorff import KHausdorff


class KHausdorffBucket(KHausdorff):
    """
    k-HAUSDORFF with a beta-bucket queue as the lower bound heap.

    Requires `epsilon > 0`: with epsilon = 0 the bucket ratio beta would be 1
    and the buckets would collapse.  Use `KHausdorff` for exact answers.
    """

    def __init__(self, G_A, G_B, epsilon, monotone=True):
        if epsilon <= 0:
            raise ValueError(
                "The bucket variant needs epsilon > 0 (beta = 1 + epsilon/2 must "
                "exceed 1).  Use KHausdorff for exact answers."
            )
        # Set before super().__init__, which builds the heap and records G_A.
        self.beta = 1 + epsilon / 2
        # The finishing threshold level.  Infinite until the first sweep, so
        # that nothing is finished before the traversal has started.
        self.threshold_level = inf
        super().__init__(G_A, G_B, epsilon, monotone)

    # -- lower bound heap ---------------------------------------------------

    def _make_lb_heap(self):
        return BetaBucketQueue(self.beta, key=lambda x: self.lb[x])

    def _reprioritize(self, node):
        self.L.changepriority(node, self.lb[node])

    def _pop_max_lb(self):
        return self.L.findmax()

    def _record(self, node):
        """
        Recompute l(node) and reposition it, unless the new bound has climbed
        past the current threshold, in which case the node is finished on the
        spot instead of having its key changed.
        """
        new_lb = self.G.lower_bound(node)
        self.lb[node] = new_lb
        level = self.L.level_of(new_lb)
        if level >= self.threshold_level:
            if node in self.L:
                self.L.remove(node)
            self._retire(node, self.L.value(level))
            return False
        if node in self.L:
            self._reprioritize(node)
        else:
            self.L.insert(node, new_lb)
        return True

    # -- finishing ----------------------------------------------------------

    def _threshold_for(self, r):
        """
        The level s at or above which a bucket can be finished.

        The paper uses s = ceil(log_beta(2 r beta / (beta - 1))), which pairs
        with reporting beta**j directly.  This implementation reports
        beta**j - rad(x) so that the value is a lower bound for every point of
        the node and not just for its centre (see `KHausdorff._retire`), and s
        has to tighten to keep the upper bound.

        Sweeping top down, by the time bucket j is reached every live node has
        l <= beta**(j+1), so Lemma 4 gives d_h^(i) <= beta**(j+1) + 2r, while
        the reported value is delta >= beta**j - r.  Requiring
        d_h^(i) <= (1 + eps) delta and using beta = 1 + eps/2:

            beta**(j+1) + 2r <= (1 + eps)(beta**j - r)
            r (3 + eps)      <= (1 + eps - beta) beta**j = (eps/2) beta**j

        so beta**j >= 2 r (3 + eps) / eps suffices, for every j >= s.
        """
        if r <= 0:
            return -inf
        return ceil(log(2 * r * (3 + self.epsilon) / self.epsilon) / log(self.beta))

    def _sweep(self, threshold):
        """Finish every occupied bucket at or above `threshold`, top down."""
        for level in self.L.levels_at_or_above(threshold):
            value = self.L.value(level)
            for node in self.L.pop_level(level):
                self._retire(node, value)

    def finish_pending(self, r):
        """
        Section 5.2's finishing condition: recompute the threshold for the
        current radius and finish whole buckets rather than single nodes.
        """
        self.threshold_level = self._threshold_for(r)
        self._sweep(self.threshold_level)

    def cleanup_all(self):
        """Sweep every remaining bucket, including the sentinel one."""
        self.threshold_level = -inf
        self._sweep(-inf)


def all_k_hausdorff_bucket(G_A, G_B, epsilon, monotone=True):
    """
    Return (delta_0, ..., delta_{n-1}) with delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i,
    computed with the beta-bucket queue variant.  Requires `epsilon > 0`.
    """
    return KHausdorffBucket(G_A, G_B, epsilon, monotone)()


def k_hausdorff_bucket(G_A, G_B, k, epsilon, monotone=True):
    """Return the approximate k-th partial directed Hausdorff distance."""
    distances = all_k_hausdorff_bucket(G_A, G_B, epsilon, monotone)
    if not 0 <= k < len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
