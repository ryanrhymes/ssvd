#!/usr/bin/env python
#
# Liang Wang @ CS Dept, Helsinki University, Finland
# 2014.05.01
#

import cPickle
import math
import numpy
import scipy.sparse
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs

try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu


def qr_decomposition(la):
    """Reduced QR decomposition."""
    print "+"*100, "Performing reduced QR decomposition ..."
    a = numpy.asfortranarray(la[0])
    del la[0], la
    m, n = a.shape
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n:
        qr = qr[:, :m]
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    q, work, info = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, "qr failed"
    assert q.flags.f_contiguous
    return q, r


def vector2matrix(vecs):
    """Convert vector array to dense ndarray."""
    m, n = vecs[0][1].size, len(vecs)
    r = numpy.zeros(shape=(m,n)).astype(numpy.float64)
    for i, v in vecs:
        r[::,i] = v
    return r


def clip_spectrum(s, k, discard=1e-6):
    rel_spectrum = numpy.abs(1.0 - numpy.cumsum(s / numpy.sum(s)))
    small = 1 + len(numpy.where(rel_spectrum > min(discard, 1.0 / k))[0])
    k = min(k, small) # clip against k
    print("keeping %i (discarding %.3f%% of energy)" %
          (k, 100 * rel_spectrum[k - 1]))
    return k


def nfs_broadcast_var(v):
    """Tweak, pyspark cannot handle large var currently."""
    path = '/cs/taatto/group/cone/tmp/'
    r = '%s%i.tmp' % (path, numpy.random.randint(1e6))
    f = open(r, 'wb')
    cPickle.dump(v, f, 2)
    return r


def nfs_load_var(p):
    f = open(p, 'rb')
    return cPickle.load(f)
