"""
    Program 3
    CMPE-139

    @author Jaylan Tse

    Bisecting K-means
"""

import numpy as np
from k_means import KMeans


class BisectingKMeans(object):

    """
        Bisecting K-means
    """

    def __init__(self, k=7, n_iters=5, k_means_pct_change=0.1, k_means_max_iter=100):
        self.k = k
        self.n_iters = n_iters
        self.k_means_pct_change = k_means_pct_change
        self.k_means_max_iter = k_means_max_iter


    def fit(self, csr):
        """Apply bisecting k-means"""

        # initialize k-means with k=2 for bisection
        kmeans = KMeans(k=2, pct_change=self.k_means_pct_change,
                        max_iter=self.k_means_max_iter)

        # initialize list of clusters with all points
        clusters = [range(0, csr.shape[0])]

        while len(clusters) < self.k:
            cluster = self.select_next_cluster(clusters)

            # bisect cluster iter times and select both clusters from split with lowest SSE
            lowest_sse = None
            best_split = None
            for i in range(self.n_iters):
                print 'Bisecting run # %d/%d, iter # %d/%d' % (len(clusters)+1,
                                                               self.k-1, i+1,
                                                               self.n_iters)

                # split cluster in two using k-means of 2
                bisection = kmeans.fit(csr, cluster)
                split = lambda data, l: [cluster[j] for j, d in enumerate(data) if d == l]
                x, y = split(bisection, 0), split(bisection, 1)

                # calculate total SSE of both clusters and store if lowest so far
                sse_total = self.sse(csr[x, :]) + self.sse(csr[y, :])
                if sse_total < lowest_sse or lowest_sse is None:
                    lowest_sse = sse_total
                    best_split = (x, y)

            # add best cluster split to list
            clusters.extend(best_split)

        return self.label_clusters(csr, clusters)


    @staticmethod
    def label_clusters(csr, clusters):
        """Create list of cluster labels for all data points from 1 to k inclusive"""

        labels = [None] * csr.shape[0]
        for label, cluster in enumerate(clusters):
            for point in cluster:
                labels[point] = label + 1

        return labels


    @staticmethod
    def select_next_cluster(clusters):
        """Select next cluster to split"""

        # pick largest cluster
        clusters.sort(key=len)
        return clusters.pop()


    @staticmethod
    def sse(points):
        """Calculate sum of squared errors of given points"""

        # find mean of rows
        mean = np.ravel(points.sum(axis=0) / float(points.shape[0]))

        # add sum of square deviations
        total = [0] * points.shape[1]
        for point in points:
            deviation = np.ravel(point - mean)
            total += (deviation ** 2)

        return sum(total) / float(points.shape[1])
