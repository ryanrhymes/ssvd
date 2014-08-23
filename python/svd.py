#!/usr/bin/env pyspark
#
# Liang Wang @ CS Dept, Helsinki University, Finland
# 2014.05.01
#

import re
import os
import sys

import numpy
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools

from utils import *
from operator import add

from pyspark import SparkConf, SparkContext
conf = (SparkConf()
###        .setMaster("spark://ukko080:7077")
        .setMaster("local[1]")
        .setAppName("pyApp")
        .set("spark.driver.memory", "10g")
        .set("spark.worker.memory", "4g")
        .set("spark.executor.memory", "4g")
        .set("spark.rdd.compress", "true")
        .set("spark.broadcast.compress", "true")
        .set("spark.task.maxFailures", 100)
        .set("spark.local.dir", "/cs/taatto/group/cone/tmp/"))

sc = SparkContext(conf = conf, pyFiles=[os.path.realpath(__file__), '/cs/taatto/group/cone/lxwang/kvasir/python/utils.py'])

RAW_FEATURE = 100000


def import_tools():
    import numpy
    import scipy.linalg
    import scipy.sparse
    from scipy.sparse import sparsetools
    pass


def test_big():
    data = sc.textFile('/data/pyraw/part-00001')
    data = data.map(lambda s: mapfunc_to_list1(s), preservesPartitioning=True)
    stochastic_svd(data, RAW_FEATURE, 1000, 0)
    pass

def test_small():
    data = sc.textFile('/data/test_matrix')
    data = data.map(lambda s: mapfunc_to_list2(s), preservesPartitioning=True)
    stochastic_svd(data, 1000, 100, 3)
    pass


def mapfunc_to_list1(line):
    import_tools()
    tL = []
    for v in re.finditer('\(([\.\d]+),([\.\d]+)\)', line):
        x, y = v.group(1,2)
        tL.append( (int(x),float(y)) )
    return tL

def mapfunc_to_list2(line):
    import_tools()
    tL = []
    for i, v in enumerate(line.split(',')):
        tL.append( (int(i),float(v)) )
    return tL


def stochastic_svd(data, raw_feature, rank, niter):
    over_sample = int(rank * 0.2)
    e_rank = rank + over_sample

    A = data.glom().map(lambda s: mapfunc_to_csc(s, raw_feature), preservesPartitioning=True)
    y = A.mapPartitions(lambda s: mapfunc_sampling(s, e_rank))
    materialize(y)
    y = y.reduceByKey(mapfunc_add).collect()
    y = [ vector2matrix(y) ]
    q, _ = qr_decomposition(y)

    for i in xrange(niter):
        print "+"*50, "#", i, "power iteration"
        b_Q = nfs_broadcast_var(q)
        y = A.mapPartitions(lambda s: mapfunc_power_iter(s, b_Q))
        materialize(y)
        y = y.reduceByKey(mapfunc_add).collect()
        y = [ vector2matrix(y) ]
        q, _ = qr_decomposition(y)

    b = matrix_multiply(q.T, A)
    c = covariance_matrix(b)[0]
    u, s, vt = scipy.linalg.svd(c)
    del b, c, vt

    import time

    s = numpy.sqrt(s)
    print  time.ctime(), 'clip_spectrum'
    k = clip_spectrum(s**2, rank)
    print  time.ctime(), 'u = u[k]'
    u = u[:, :k].copy()
    print  time.ctime(), 's = ..'
    s = s[:k]
    print  time.ctime(), 'numpy'
    u = numpy.dot(q, u)
    print  time.ctime(), 'done'
    print s

    return u.astype(numpy.float64), s.astype(numpy.float64)


def mapfunc_add(v1, v2):
    v1 = v1 + v2
    return v1


def mapfunc_to_csc(it, num_rows):
    import_tools()
    r = chunk2csc(it, num_rows)
    return r


def mapfunc_sampling(it, rank):
    import_tools()
    mtx = it.next()
    m, n = mtx.shape
    y = numpy.zeros(dtype=mtx.dtype, shape=(m, rank))
    o = numpy.random.normal(0.0, 1.0, (n, rank)).astype(y.dtype)
    sparsetools.csc_matvecs(m, n, rank, mtx.indptr, mtx.indices,
                            mtx.data, o.ravel(), y.ravel())
    del o
    return enumerate(y.T)


def chunk2csc(chunk, num_rows):
    nnz, data, indices, indptr = 0, [], [], [0]
    for index, col in enumerate(chunk):
        indices.extend([row_id for row_id, _ in col])
        data.extend([v for _, v in col])
        nnz += len(col)
        indptr.append(nnz)
    num_cols = len(indptr) - 1
    data = numpy.asarray(data, dtype=numpy.float64)
    indices = numpy.asarray(indices)
    result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_rows, num_cols), dtype=numpy.float64)
    return result


def matrix_multiply(A, B):
    """A is small-dense, and B is fat-short."""
    b_A = nfs_broadcast_var(A)
    r = B.map(lambda s: scipy.sparse.csc_matrix(nfs_load_var(b_A)) * s)
    return r


def mapfunc_power_iter(it, b_Q):
    """One power iteration, A is fat-short and Q is dense-samll."""
    import_tools()
    A = it.next()
    y = A.T * nfs_load_var(b_Q)
    y = A * y
    return enumerate(y.T)


def covariance_matrix(A):
    """A is fat-short, return a small-dense matrix."""
    r = A.map(lambda s: (s * s.T).todense()).reduce(add)
    return [r]


def materialize(A):
    """Tweak: materialize a RDD by simple count."""
    A.count()
    return None


if __name__=='__main__':
    ###test_small()
    test_big()
    sys.exit(0)
