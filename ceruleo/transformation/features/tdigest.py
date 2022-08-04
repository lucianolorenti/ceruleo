import math

import numpy as np
from typing import Iterable, List


def k_to_q(k: float, d: float) -> float:
    """
    Parameters
    ----------
    k : float
        [description]
    d : float
        [description]

    Returns
    -------
    float
        [description]
    """
    k_div_d = k / d
    if k_div_d >= 0.5:
        base = 1 - k_div_d
        return 1 - 2 * base * base
    else:
        return 2 * k_div_d * k_div_d


def inplace_merge(arr, start, mid, end):
    start2 = mid + 1
    if arr[mid] <= arr[start2]:
        return
    while start <= mid and start2 <= end:
        if arr[start] <= arr[start2]:
            start += 1
        else:
            value = arr[start2]
            index = start2
            while index != start:
                arr[index] = arr[index - 1]
                index -= 1

            arr[start] = value
            start += 1
            mid += 1
            start2 += 1


class Centroid:
    def __init__(self, mean: float = 0.0, weight: float = 1.0):
        assert weight > 0
        self.mean = mean
        self.weight = weight

    def add(self, sum: float, weight: float) -> float:
        """Adds the sum/weight to this centroid, and returns the new sum.

        Parameters
        ----------
        sum : float
            [description]
        weight : float
            [description]

        Returns
        -------
        [type]
            [description]
        """
        sum += self.mean * self.weight
        self.weight += weight
        self.mean = sum / self.weight
        return sum

    def __lt__(self, other):
        return self.mean < other.mean


class TDigest:
    def __init__(self, maxSize: int = 100):
        self.maxSize = maxSize
        self.sum = 0.0
        self.count = 0.0
        self.max = np.nan
        self.min = np.nan
        self.centroids = []

    def construct(
        centroids: List[Centroid],
        sum: float,
        count: float,
        max_val: float,
        min_val: float,
        maxSize: int = 100,
    ):
        t = TDigest(maxSize)
        t.sum = sum
        t.count = count
        t.max = max_val
        t.min = min_val
        if len(centroids) < maxSize:
            t.centroids = centroids
        # // Number of centroids is greater than maxSize, we need to compress them
        # // When merging, resulting digest takes the maxSize of the first digest
        sz = len(centroids)
        digests = [
            TDigest(maxSize),
            TDigest.construct(centroids, sum, count, max, min, sz),
        ]
        return t.merge(digests)

    def estimate_quantile(self, q: float) -> float:
        """Estimates the value of the given quantile.

        Parameters
        ----------
        q : float
            Quantile

        Returns
        -------
        float
            Value of the quantile
        """
        if len(self.centroids) == 0:
            return 0.0
        rank = q * self.count
        pos = 0
        t = 0
        if q > 0.5:
            if q >= 1.0:
                return self.max
            pos = 0
            t = self.count
            for i, centroid in enumerate(self.centroids[::-1]):
                t -= centroid.weight
                if rank >= t:
                    pos = len(self.centroids) - i - 1
                    break

        else:
            if q <= 0.0:
                return self.min

            pos = len(self.centroids) - 1
            t = 0
            for i, centroid in enumerate(self.centroids):
                if rank < t + centroid.weight:
                    pos = i
                    break
                t += centroid.weight

        delta = 0
        min = self.min
        max = self.max
        if len(self.centroids) > 1:
            if pos == 0:
                delta = self.centroids[pos + 1].mean - self.centroids[pos].mean
                max = self.centroids[pos + 1].mean
            elif pos == len(self.centroids) - 1:
                delta = self.centroids[pos].mean - self.centroids[pos - 1].mean
                min = self.centroids[pos - 1].mean
            else:
                delta = (
                    self.centroids[pos + 1].mean - self.centroids[pos - 1].mean
                ) / 2
                min = self.centroids[pos - 1].mean
                max = self.centroids[pos + 1].mean

        value = (
            self.centroids[pos].mean
            + ((rank - t) / self.centroids[pos].weight - 0.5) * delta
        )
        return np.clip(value, min, max)

    def merge(self, digests: List["TDigest"]):
        nCentroids = 0
        for digest in digests:
            nCentroids += digest.centroids_.size()

        if nCentroids == 0:
            return TDigest()

        centroids = []
        starts = []

        count = 0

        # We can safely use these limits to avoid isnan checks below because we know
        # nCentroids > 0, so at least one TDigest has a min and max.
        min = np.inf
        max = -np.inf

        for digest in digests:
            starts.append(len(centroids))
            curCount = digest.count()
            if curCount > 0:
                assert not np.isnan(digest.min)
                assert not np.isnan(digest.max)
                min = min(min, digest.min)
                max = max(max, digest.max)
                count += curCount
                for centroid in digest.centroids:
                    centroids.push_back(centroid)

        digestsPerBlock = 1
        while digestsPerBlock < len(starts):
            # Each sorted block is digestPerBlock digests big. For each step, try to
            # merge two blocks together.
            i = 0
            while i < starts.size():
                # It is possible that this block is incomplete (less than digestsPerBlock
                # big). In that case, the rest of the block is sorted and leave it alone
                if i + digestsPerBlock < starts.size():
                    first = starts[i]
                    middle = starts[i + digestsPerBlock]

                    # It is possible that the next block is incomplete (less than
                    # digestsPerBlock big). In that case, merge to end. Otherwise, merge to
                    # the end of that block.
                    if i + (digestsPerBlock * 2) < len(starts):
                        last = starts[i + 2 * digestsPerBlock]
                    else:
                        last = len(centroids) - 1
                    inplace_merge(centroids, first, middle, last)
            i += digestsPerBlock * 2

            digestsPerBlock *= 2

        maxSize = digests[0].maxSize
        result = TDigest(maxSize)

        compressed = []

        k_limit = 1
        q_limit_times_count = k_to_q(k_limit, maxSize) * count

        cur = centroids[0]
        weightSoFar = cur.weight()
        sumsToMerge = 0
        weightsToMerge = 0
        for centroid in centroids:
            weightSoFar += centroid.weight
            if weightSoFar <= q_limit_times_count:
                sumsToMerge += centroid.mean * centroid.weight
                weightsToMerge += centroid.weight
            else:
                result.sum_ += cur.add(sumsToMerge, weightsToMerge)
                sumsToMerge = 0
                weightsToMerge = 0
                compressed.append(cur)
                q_limit_times_count = k_to_q(k_limit, maxSize) * count
                k_limit += 1
                cur = centroid

        result.sum += cur.add(sumsToMerge, weightsToMerge)
        compressed.append(cur)
        compressed.shrink_to_fit()

        # Deal with floating point precision
        compressed = sorted(compressed)

        result.count = count
        result.min = min
        result.max = max
        result.centroids = compressed
        return result

    def merge_unsorted(self, unsortedValues: Iterable[float]) -> "TDigest":
        """Merge unsorted values by first sorting them.


        Parameters
        ----------
        unsortedValues: Iterable[float]
            Values to Merge

        Returns
        -------
        TDigest
            [description]
        """
        return self.merge_sorted(sorted(unsortedValues))

    def merge_sorted(self, sortedValues: Iterable[float]):
        if len(sortedValues) == 0:
            return self

        result = TDigest(self.maxSize)

        result.count = self.count + len(sortedValues)

        maybeMin = sortedValues[0]
        maybeMax = sortedValues[-1]
        if self.count > 0:
            # // We know that min_ and max_ are numbers
            result.min = min(self.min, maybeMin)
            result.max = max(self.max, maybeMax)
        else:
            # // We know that min_ and max_ are NaN.
            result.min = maybeMin
            result.max = maybeMax

        compressed = []

        k_limit = 1
        q_limit_times_count = k_to_q(k_limit, self.maxSize) * result.count
        k_limit += 1

        it_centroids = 0
        it_sortedValues = 0

        cur = None

        if (it_centroids < len(self.centroids) - 1) and (
            self.centroids[it_centroids].mean < sortedValues[it_sortedValues]
        ):
            cur = self.centroids[it_centroids]
            it_centroids += 1
        else:
            cur = Centroid(sortedValues[it_sortedValues], 1.0)
            it_sortedValues += 1

        weightSoFar = cur.weight

        # Keep track of sums along the way to reduce expensive floating points
        sumsToMerge = 0.0
        weightsToMerge = 0.0

        while (it_centroids < len(self.centroids) - 1) or (
            it_sortedValues < len(sortedValues) - 1
        ):
            next = None

            if (it_centroids < len(self.centroids) - 1) and (
                (it_sortedValues == len(sortedValues) - 1)
                or (self.centroids[it_centroids].mean < sortedValues[it_sortedValues])
            ):
                next = self.centroids[it_centroids]
                it_centroids += 1
            else:
                next = Centroid(sortedValues[it_sortedValues], 1.0)
                it_sortedValues += 1

            nextSum = next.mean * next.weight
            weightSoFar += next.weight

            if weightSoFar <= q_limit_times_count:
                sumsToMerge += nextSum
                weightsToMerge += next.weight
            else:
                result.sum += cur.add(sumsToMerge, weightsToMerge)
                sumsToMerge = 0
                weightsToMerge = 0
                compressed.append(cur)
                q_limit_times_count = k_to_q(k_limit, self.maxSize) * result.count
                k_limit += 1
                cur = next

        result.sum += cur.add(sumsToMerge, weightsToMerge)
        compressed.append(cur)

        # Deal with floating point precision
        compressed = sorted(compressed)

        result.centroids = compressed
        return result
