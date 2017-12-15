#! /usr/bin/python

"""
    Program 3
    CMPE-139

    @author Jaylan Tse

    Bisecting K-means
"""

import sys
import csr_matrix
from bisecting_k_means import BisectingKMeans
from sklearn.metrics import calinski_harabaz_score


def save(labels, f_name='results.txt'):
    """Save labels to file"""

    f = open(f_name, 'w')
    for label in labels:
        f.write('%d\n' % (label))
    f.close()


def main():
    """Main"""

    data = csr_matrix.read('../data/train.dat')
    csr_matrix.idf(data)
    csr_matrix.l2normalize(data)

#    metrics = []
#    for i in [3,5,7,9,11,13,15,17,19,21]:
#        bkm = BisectingKMeans(k=i, n_iters=5, k_means_pct_change=0, k_means_max_iter=100)
#        labels = bkm.fit(data)
#
#        score = calinski_harabaz_score(data.toarray(), labels)
#        metrics.append(score)
#
#    print metrics

    bkm = BisectingKMeans(k=7, n_iters=5, k_means_pct_change=0, k_means_max_iter=100)
    labels = bkm.fit(data)

    if len(sys.argv) > 1:
        save(labels, str(sys.argv[1]))
    else:
        save(labels)


if __name__ == "__main__":
    main()
