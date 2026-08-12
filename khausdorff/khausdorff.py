"""
k-HAUSDORFF: approximate the k-th partial directed Hausdorff distance for every
k at once, in a single dual-tree traversal.

This implements Section 5 of

    O. A. Chubet, P. M. Parikh, D. R. Sheehy and S. S. Sheth,
    "Approximating the Directed Hausdorff Distance",
    Computing in Geometry and Topology, 4(2):6:1-6:16, 2023.

The HAUSDORFF algorithm of Section 4 stops as soon as it has identified the
point of A that is (approximately) farthest from B.  The observation behind
k-HAUSDORFF is that if, instead of stopping, we discard that point and keep
going, the next time the condition holds we have found the second farthest
point, and so on.  Running the traversal to completion therefore yields the
whole sequence

    (delta_0, ..., delta_{n-1})    with    delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i,

where delta_0 is the ordinary directed Hausdorff distance.

This module holds the reference variant, which uses an exact max-heap as the
lower bound heap.  `khausdorff.bucketkhausdorff` holds the variant of Section
5.2 that replaces it with a beta-bucket queue to reach the running time claimed
in the paper.
"""

from greedypermutation.dualtrees.dualtreesearch import DualTreeSearch

from khausdorff.lowerboundheap import LowerBoundHeap


class KHausdorff(DualTreeSearch):
    """
    Approximate every partial directed Hausdorff distance d_h^(k)(A, B).

    The traversal machinery (viability graph `self.G`, radius heap `self.H`)
    comes from `DualTreeSearch`.  On top of it this class maintains the *lower
    bound heap* `self.L`: a max-heap holding every node of G_A currently in the
    viability graph, keyed by its local lower bound l(x).
    """

    def __init__(self, G_A, G_B, epsilon=0, monotone=True):
        """
        `G_A` and `G_B` are greedy trees (`greedypermutation.balltree.Ball`),
        `epsilon >= 0` is the approximation parameter (0 gives exact answers)
        and `monotone` clamps the output to be non-increasing.
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        super().__init__(G_A, G_B)
        self.epsilon = epsilon
        self.monotone = monotone
        self.out = []
        # Local lower bounds l(x), for every node x of G_A in the graph.
        self.lb = {}
        self.L = self._make_lb_heap()
        # Nodes that have been finished and removed from the viability graph.
        self.done = set()
        self._record(G_A)

    # -- lower bound heap ---------------------------------------------------

    def _make_lb_heap(self):
        """
        Build the lower bound heap.  Its key reads `self.lb`, which is mutable,
        so refreshing a priority is just `changepriority(node)` with no value.

        `LowerBoundHeap` rather than the upstream `MaxHeap` because this
        algorithm removes nodes from the middle of the heap, which upstream
        gets wrong; see that module.
        """
        return LowerBoundHeap(key=lambda x: self.lb[x])

    def _record(self, node):
        """
        Compute l(node) and (re)position `node` in the lower bound heap.

        Returns True if `node` is still in the graph afterwards.  It always is
        here; the bucket variant can finish a node on the spot instead.
        """
        new_lb = self.G.lower_bound(node)
        if node in self.lb:
            self.lb[node] = new_lb
            self._reprioritize(node)
        else:
            self.lb[node] = new_lb
            self.L.insert(node)
        return True

    def _reprioritize(self, node):
        """Refresh the heap position of `node` after `self.lb[node]` changed."""
        self.L.changepriority(node)

    def _pop_max_lb(self):
        """Return the node of G_A with the largest local lower bound."""
        return self.L.findmax()

    # -- viability graph ----------------------------------------------------

    def _prune(self, a):
        """
        Drop the edges out of `a` that cannot hold a nearest neighbour of any
        point in `a`.

        This is the pruning rule of `NNBallGraph.prune` in the upstream
        `hausdorff` module, which `ViabilityGraph` does not provide.  The
        neighbour realising `mindist` always survives, so `self.G.A[a]` never
        becomes empty and `lower_bound(a)` is always well defined.
        """
        self.G.update_mindist(a)
        threshold = self.G.mindist[a] + 2 * a.radius
        for b in [b for b in self.G.A[a] if a.dist(b.center) - b.radius > threshold]:
            self.G.remove_edge(a, b)

    # -- finishing ----------------------------------------------------------

    def _emit(self, value, count):
        """Append `value` to the output once per point, keeping it monotone."""
        if self.monotone and self.out:
            # Clamping downwards is always safe: the guarantee is
            # delta_i <= d_h^(i), and the sequence d_h^(i) is non-increasing.
            value = min(value, self.out[-1])
        self.out.extend([value] * count)

    def _finish(self, x, value=None):
        """
        Finish node `x`: emit its lower bound once per point of pts(x), then cut
        it out of the lower bound heap and the viability graph.
        """
        self.L.remove(x)
        self._retire(x, value)

    def _retire(self, x, value=None):
        """
        Finish `x` when it has *already* been taken out of the lower bound heap.

        The value reported for every point of pts(x) is l(x) - rad(x), not l(x).
        The paper reports l(x), but l(x) only bounds d(ctr(x), B) from below; a
        point of pts(x) sitting closer to B than the centre does would then be
        credited with too large a distance, breaking delta_i <= d_h^(i).
        Subtracting the radius gives a bound valid for *every* point of the
        node, since d(p, B) >= d(ctr(x), B) - rad(x) >= l(x) - rad(x).  See
        `_finish_constant` for how the finishing condition pays for it.
        """
        bound = self.lb[x] if value is None else value
        self._emit(max(0.0, bound - x.radius), len(x))
        del self.lb[x]
        self.G.remove(x)
        self.done.add(x)

    def _finish_constant(self):
        """
        The constant c in the finishing condition r <= c * l(x).

        The paper uses c = eps/2, which secures the upper bound
        d_h^(i) <= (1+eps) delta_i but not the lower bound delta_i <= d_h^(i)
        (see `_retire`).  Reporting l(x) - rad(x) instead of l(x) fixes the
        lower bound but costs accuracy, so c has to shrink to keep the upper
        bound.  Writing L = l(x) for the top of the heap, Lemma 4 of the paper
        gives d_h^(i) <= L + 2r, and the reported value is
        delta = L - rad(x) >= (1-c)L, so

            d_h^(i) <= (1 + 2c) L <= (1 + 2c)/(1 - c) * delta.

        Requiring (1 + 2c)/(1 - c) <= 1 + eps gives c <= eps/(3 + eps).
        """
        return self.epsilon / (3 + self.epsilon)

    def finish_pending(self, r):
        """
        The finishing condition of Section 5.2.

        With `r` the radius of the node about to be processed, finish the top of
        the lower bound heap while r <= c * l(x).
        """
        c = self._finish_constant()
        while len(self.L):
            x = self._pop_max_lb()
            # Written this way rather than as r / l(x) so that epsilon == 0 and
            # l(x) == 0 are both fine.
            if r > c * self.lb[x]:
                return
            self._finish(x)

    def cleanup_all(self):
        """
        Finish everything that is left, with r = 0.

        Draining the lower bound heap in decreasing order of l(x), rather than
        iterating over `self.G.A` as `DualTreeSearch.cleanup` does, is what
        keeps the output sequence ordered.
        """
        while len(self.L):
            self._finish(self._pop_max_lb())

    # -- DualTreeSearch hooks -----------------------------------------------

    def setup_children(self, ball):
        """
        Called when an A-side `ball` is split, after its children have been
        added as vertices but *before* their edges exist, so no lower bound can
        be computed here.  All we do is retire the parent.
        """
        self.L.remove(ball)
        del self.lb[ball]

    def update(self, node, ball):
        """Refresh the affected A-side vertex `node` after `ball` was split."""
        if self._record(node):
            self._prune(node)

    # -- main loop ----------------------------------------------------------

    def _skip(self, ball):
        """
        True if `ball` is no longer part of the viability graph and must not be
        split.

        Two cases.  An A-side node that has been finished is gone from `self.G`,
        and `DualTreeSearch.iteration` would look for it in `self.G.B` and raise
        a KeyError.  A B-side node whose neighbourhood has emptied out can never
        regain an edge, so splitting it is pure waste.
        """
        if ball in self.done:
            return True
        if ball in self.G.B and not self.G.B[ball]:
            del self.G.B[ball]
            return True
        return False

    def __call__(self):
        """
        Run the traversal to completion and return the list of approximate
        partial distances (delta_0, ..., delta_{n-1}).

        `DualTreeSearch.__call__` is not reused because the finishing condition
        has to be tested at the *start* of every iteration, and because nodes
        that have already been finished must be skipped.
        """
        for ball in self.H:
            self.finish_pending(ball.radius)
            if ball.isleaf():
                # The heap is ordered by decreasing radius, so from here on
                # every surviving node is a leaf of radius 0.
                break
            if self._skip(ball):
                continue
            self.iteration(ball)

        self.cleanup_all()
        return self.out


def all_k_hausdorff(G_A, G_B, epsilon=0, monotone=True):
    """
    Return (delta_0, ..., delta_{n-1}) with delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i.

    `G_A` and `G_B` are greedy trees, as produced by
    `greedypermutation.balltree.greedy_tree`.  With `epsilon=0` the result is
    exact.  Note that, like the underlying Hausdorff distance, this is directed:
    swapping the arguments gives a different answer.
    """
    return KHausdorff(G_A, G_B, epsilon, monotone)()


def k_hausdorff(G_A, G_B, k, epsilon=0, monotone=True):
    """Return the approximate k-th partial directed Hausdorff distance."""
    distances = all_k_hausdorff(G_A, G_B, epsilon, monotone)
    if not 0 <= k < len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
