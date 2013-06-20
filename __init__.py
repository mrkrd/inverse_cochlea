#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import warnings


try:
    from isgram_reconstr import ISgramReconstructor
except ImportError:
    warnings.warn("ISgram reconstruction not loaded (missing pytave or oct2py)")
from isgram_reconstr import ISgramReconstructor


from mlp_reconstr import MlpReconstructor

from common import (
    run_ear,
    ANF,
    Signal
)

from sgram import Spectrogram
from cochlea import set_dbspl

import cPickle as pickle

def load(fname):
    r = pickle.load(open(fname,'r'))
    return r
