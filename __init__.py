#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"


try:
    from isgram_reconstr import ISgramReconstructor
except ImportError:
    print "ISgram reconstruction not loaded (probably pytave missing)"


from mlp_reconstr import MlpReconstructor

from common import run_ear, band_pass_filter

import cPickle as pickle

def load(fname):
    return pickle.load(open(fname,'r'))
