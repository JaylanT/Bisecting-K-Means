"""
    CSR functions from Activity Data 3
"""

from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix


def read(fname, nidx=1):
    r"""
        Read CSR matrix from a text file.

        \param fname File name for CSR matrix
        \param nidx Indexing type in CSR file. What does numbering of feature IDs start with?
    """
    with open(fname) as f:
        lines = f.readlines()

    nrows = len(lines)
    ncols = 0
    nnz = 0
    for i in xrange(nrows):
        p = lines[i].split()
        if len(p) % 2 != 0:
            raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
        nnz += len(p)/2
        for j in xrange(0, len(p), 2):
            cid = int(p[j]) - nidx
            if cid+1 > ncols:
                ncols = cid+1

    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0
    for i in xrange(nrows):
        p = lines[i].split()
        for j in xrange(0, len(p), 2):
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n

    assert n == nnz

    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)


def idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val = mat.indices, mat.data
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k, v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat


def l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    val, ptr = mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum

    if copy is True:
        return mat
