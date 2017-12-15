"""
    Program 3
    CMPE-139

    @author Jaylan Tse

    K-means
"""

import numpy as np


class KMeans(object):

    """
        K-means
    """

    def __init__(self, k=8, pct_change=0.1, max_iter=100):
        self.k = k
        self.pct_change = pct_change
        self.max_iter = max_iter


    def fit(self, csr, rows):
        """Apply k-means clustering"""

        if self.k > len(rows):
            raise ValueError('K is greater than number of rows.')

        points = csr[rows, :]
        centroids = self.select_random_centroids(csr, rows, self.k)
        assigned_clusters = [None] * len(rows)

        iters = 0
        done = False
        while not done:
            old_clusters = assigned_clusters[:]

            # assign each point to a cluster
            assigned_clusters = self.assign_clusters(points, centroids)

            # done if percent change less than/equal to pct_change or number of iters equal max_iter
            num_changes = self.differences(old_clusters, assigned_clusters)
            percent_change = (num_changes / float(len(rows))) * 100
            iters += 1
            done = percent_change <= self.pct_change or iters == self.max_iter

            print 'K-means iter: %d/%d, change: %.2f' % (iters, self.max_iter, percent_change)

            if not done:
                centroids = self.update_centroids(points, self.k, assigned_clusters)

        return assigned_clusters


    def assign_clusters(self, points, centroids):
        """Find closest centroid to each point and assign point to cluster"""

        assigned_clusters = [None] * points.shape[0]

        for i, point in enumerate(points):
            curr_point = point.toarray()[0]
            closest_centroid_label = None
            highest_sim = -1

            # find closest centroid to point
            for label, centroid in enumerate(centroids):
                sim = self.cos_sim(curr_point, centroid)

                if sim > highest_sim:
                    closest_centroid_label = label
                    highest_sim = sim

            assigned_clusters[i] = closest_centroid_label

        return assigned_clusters


    @staticmethod
    def cos_sim(v1, v2):
        """Calculate cosine similarity given normalized vectors"""

        return np.dot(v1, v2)


    @staticmethod
    def differences(v1, v2):
        """Find number of differences between two lists of equal sizes"""

        if len(v1) != len(v2):
            raise ValueError("Lists of different length.")
        return sum(i != j for i, j in zip(v1, v2))


    @staticmethod
    def select_random_centroids(csr, rows, k):
        """Choose random k points to be centroids"""

        indices = np.random.choice(len(rows), k, replace=False)
        initial_clusters = [rows[idx] for idx in indices]
        return [csr.getrow(cluster).toarray()[0] for cluster in initial_clusters]


    @staticmethod
    def update_centroids(points, k, assigned_clusters):
        """Calculate and update means of centroids for each cluster"""

        centroids = [None] * k

        for i in range(k):
            new_centroid = [0] * points.shape[1]

            # add all points together from same cluster
            for j, point in enumerate(points):
                if assigned_clusters[j] == i:
                    new_centroid += point.toarray()[0]

            centroids[i] = new_centroid / float(assigned_clusters.count(i))

        return centroids
